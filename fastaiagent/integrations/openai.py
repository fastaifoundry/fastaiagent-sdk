"""Auto-tracing for the OpenAI SDK."""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_original_create = None
_original_acreate = None
_enabled = False


def _serialize_messages(messages: Any) -> str | None:
    if messages is None:
        return None
    try:
        return json.dumps(messages, default=str)
    except Exception:
        logger.debug("Failed to serialize OpenAI messages for trace", exc_info=True)
        return None


def _extract_response_payload(result: Any) -> tuple[str | None, str | None, str | None]:
    """Pull (content, tool_calls_json, finish_reason) from an OpenAI completion."""
    content: str | None = None
    tool_calls_json: str | None = None
    finish_reason: str | None = None
    try:
        choice = result.choices[0]
        msg = choice.message
        content = getattr(msg, "content", None)
        tcs = getattr(msg, "tool_calls", None)
        if tcs:
            tool_calls_json = json.dumps(
                [
                    {
                        "id": getattr(t, "id", None),
                        "name": getattr(getattr(t, "function", None), "name", None),
                        "arguments": getattr(getattr(t, "function", None), "arguments", None),
                    }
                    for t in tcs
                ],
                default=str,
            )
        finish_reason = getattr(choice, "finish_reason", None)
    except Exception:
        logger.debug("Failed to extract OpenAI response payload for trace", exc_info=True)
    return content, tool_calls_json, finish_reason


class _ChunkObserver:
    """Accumulates a streamed chat completion for the span (1.89.0).

    Usage arrives only with ``stream_options={"include_usage": True}``, on a
    final chunk whose ``choices`` is empty. Tool calls arrive as deltas keyed by
    ``index``: the first carries ``id`` and ``function.name``, the rest append
    to ``function.arguments``.
    """

    def __init__(self) -> None:
        self.content: list[str] = []
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def observe(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.input_tokens = getattr(usage, "prompt_tokens", None)
            self.output_tokens = getattr(usage, "completion_tokens", None)
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            text = getattr(delta, "content", None)
            if text:
                self.content.append(text)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = self.tool_calls.setdefault(
                    getattr(tc, "index", 0) or 0, {"id": None, "name": None, "arguments": ""}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn: Any = getattr(tc, "function", None)
                name = getattr(fn, "name", None)
                if name:
                    slot["name"] = name
                arguments = getattr(fn, "arguments", None)
                if arguments:
                    slot["arguments"] += arguments
            if getattr(choice, "finish_reason", None):
                self.finish_reason = choice.finish_reason

    def apply(self, span: Any) -> None:
        from fastaiagent.trace.span import set_genai_attributes

        tool_calls = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        set_genai_attributes(
            span,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            response_content="".join(self.content) if self.content else None,
            response_tool_calls=json.dumps(tool_calls, default=str) if tool_calls else None,
            finish_reason=self.finish_reason,
        )


def _open_span(kwargs: dict[str, Any]) -> Any:
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_genai_attributes

    model = kwargs.get("model", "unknown")
    span = get_tracer("fastaiagent.integrations.openai").start_span(f"openai.chat.{model}")
    set_genai_attributes(
        span,
        system="openai",
        model=model,
        temperature=kwargs.get("temperature"),
        max_tokens=kwargs.get("max_tokens"),
        request_messages=_serialize_messages(kwargs.get("messages")),
        request_tools=_serialize_messages(kwargs.get("tools")),
    )
    return span


def _record_completion(span: Any, result: Any) -> None:
    from fastaiagent.trace.span import set_genai_attributes

    if hasattr(result, "usage") and result.usage:
        set_genai_attributes(
            span,
            input_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.completion_tokens,
        )
    content, tool_calls_json, finish_reason = _extract_response_payload(result)
    set_genai_attributes(
        span,
        response_content=content,
        response_tool_calls=tool_calls_json,
        finish_reason=finish_reason,
    )


def _settle(
    span: Any, result: Any, kwargs: dict[str, Any], secrets: list[str], is_async: bool
) -> None:
    """After ``create`` returned: hand the span to the stream, or end it now."""
    from fastaiagent.integrations._stream_span import wrap_stream

    if kwargs.get("stream") is True:
        if wrap_stream(result, span, _ChunkObserver(), secrets, is_async=is_async):
            return
        span.end()
        return
    try:
        _record_completion(span, result)
    finally:
        span.end()


def enable() -> None:
    """Enable auto-tracing for OpenAI SDK calls, sync and async.

    1.89.0: ``stream=True`` calls keep their span open until the stream ends,
    so usage is recorded and an error raised part-way through ends the span
    ERROR (scrubbed of credentials). ``AsyncOpenAI`` is traced too.
    """
    global _original_create, _original_acreate, _enabled
    if _enabled:
        return

    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI SDK is required. Install with: pip install fastaiagent[openai]")

    from fastaiagent.integrations._stream_span import (
        client_secrets,
        mark_error_scrubbed,
        use_span,
    )

    completions_mod = openai.resources.chat.completions
    _original_create = completions_mod.Completions.create
    _original_acreate = completions_mod.AsyncCompletions.create
    original_create = _original_create
    original_acreate = _original_acreate

    @functools.wraps(original_create)
    def traced_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        span = _open_span(kwargs)
        secrets = client_secrets(self_inner)
        try:
            with use_span(span):
                result = original_create(self_inner, *args, **kwargs)
        except Exception as exc:
            mark_error_scrubbed(span, exc, secrets)
            span.end()
            raise
        except BaseException:
            span.end()
            raise
        _settle(span, result, kwargs, secrets, is_async=False)
        return result

    @functools.wraps(original_acreate)
    async def traced_acreate(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        span = _open_span(kwargs)
        secrets = client_secrets(self_inner)
        try:
            with use_span(span):
                result = await original_acreate(self_inner, *args, **kwargs)
        except Exception as exc:
            mark_error_scrubbed(span, exc, secrets)
            span.end()
            raise
        except BaseException:
            span.end()
            raise
        _settle(span, result, kwargs, secrets, is_async=True)
        return result

    completions_mod.Completions.create = traced_create  # type: ignore[method-assign,assignment]
    completions_mod.AsyncCompletions.create = traced_acreate  # type: ignore[method-assign,assignment]
    _enabled = True


def disable() -> None:
    """Disable auto-tracing for OpenAI SDK calls."""
    global _original_create, _original_acreate, _enabled
    if not _enabled:
        return

    try:
        import openai

        completions_mod = openai.resources.chat.completions
        if _original_create:
            completions_mod.Completions.create = _original_create  # type: ignore[method-assign]
        if _original_acreate:
            completions_mod.AsyncCompletions.create = _original_acreate  # type: ignore[method-assign]
    except ImportError:
        logger.debug("OpenAI SDK not available during disable", exc_info=True)

    _enabled = False
    _original_create = None
    _original_acreate = None
