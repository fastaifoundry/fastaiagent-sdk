"""Auto-tracing for the Anthropic SDK."""

from __future__ import annotations

import functools
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_original_create = None
_original_acreate = None
_original_stream = None
_original_astream = None
_enabled = False


def _serialize_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception:
        logger.debug("Failed to serialize Anthropic payload for trace", exc_info=True)
        return None


def _extract_response_payload(result: Any) -> tuple[str | None, str | None, str | None]:
    """Pull (text_content, tool_calls_json, stop_reason) from an Anthropic response."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    try:
        for block in getattr(result, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", None),
                    }
                )
    except Exception:
        logger.debug("Failed to extract Anthropic response payload for trace", exc_info=True)
    content = "".join(text_parts) if text_parts else None
    tool_calls_json = json.dumps(tool_calls, default=str) if tool_calls else None
    stop_reason = getattr(result, "stop_reason", None)
    return content, tool_calls_json, stop_reason


class _EventObserver:
    """Accumulates a streamed message for the span (1.89.0).

    ``message_start`` carries the input tokens; ``message_delta`` carries the
    cumulative output tokens and the stop reason. Text and tool input arrive as
    ``content_block_delta`` events against the block opened by
    ``content_block_start``.
    """

    def __init__(self) -> None:
        self.blocks: dict[int, dict[str, Any]] = {}
        self.stop_reason: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

    def _usage(self, usage: Any) -> None:
        if usage is None:
            return
        if getattr(usage, "input_tokens", None) is not None:
            self.input_tokens = usage.input_tokens
        if getattr(usage, "output_tokens", None) is not None:
            self.output_tokens = usage.output_tokens

    def observe(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            self._usage(getattr(getattr(event, "message", None), "usage", None))
        elif kind == "content_block_start":
            block = getattr(event, "content_block", None)
            self.blocks[getattr(event, "index", 0)] = {
                "type": getattr(block, "type", None),
                "text": getattr(block, "text", "") or "",
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", None),
                "json": "",
            }
        elif kind == "content_block_delta":
            slot = self.blocks.setdefault(
                getattr(event, "index", 0), {"type": "text", "text": "", "json": ""}
            )
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", None) == "text_delta":
                slot["text"] += getattr(delta, "text", "") or ""
            elif getattr(delta, "type", None) == "input_json_delta":
                slot["json"] += getattr(delta, "partial_json", "") or ""
        elif kind == "message_delta":
            self._usage(getattr(event, "usage", None))
            reason = getattr(getattr(event, "delta", None), "stop_reason", None)
            if reason:
                self.stop_reason = reason

    def apply(self, span: Any) -> None:
        from fastaiagent.trace.span import set_genai_attributes

        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for i in sorted(self.blocks):
            block = self.blocks[i]
            if block.get("type") == "tool_use":
                try:
                    args: Any = json.loads(block["json"]) if block["json"] else {}
                except ValueError:
                    args = block["json"]
                tool_calls.append({"id": block.get("id"), "name": block.get("name"), "input": args})
            elif block.get("text"):
                texts.append(block["text"])
        set_genai_attributes(
            span,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            response_content="".join(texts) if texts else None,
            response_tool_calls=json.dumps(tool_calls, default=str) if tool_calls else None,
            finish_reason=self.stop_reason,
        )


def _open_span(kwargs: dict[str, Any]) -> Any:
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_genai_attributes

    model = kwargs.get("model", "unknown")
    span = get_tracer("fastaiagent.integrations.anthropic").start_span(
        f"anthropic.messages.{model}"
    )
    set_genai_attributes(
        span,
        system="anthropic",
        model=model,
        temperature=kwargs.get("temperature"),
        max_tokens=kwargs.get("max_tokens"),
        request_messages=_serialize_json(kwargs.get("messages")),
        request_tools=_serialize_json(kwargs.get("tools")),
    )
    return span


def _record_message(span: Any, result: Any) -> None:
    from fastaiagent.trace.span import set_genai_attributes

    if hasattr(result, "usage") and result.usage:
        set_genai_attributes(
            span,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )
    content, tool_calls_json, stop_reason = _extract_response_payload(result)
    set_genai_attributes(
        span,
        response_content=content,
        response_tool_calls=tool_calls_json,
        finish_reason=stop_reason,
    )


def _settle(
    span: Any, result: Any, kwargs: dict[str, Any], secrets: list[str], is_async: bool
) -> None:
    """After ``create`` returned: hand the span to the stream, or end it now."""
    from fastaiagent.integrations._stream_span import wrap_stream

    if kwargs.get("stream") is True:
        if wrap_stream(result, span, _EventObserver(), secrets, is_async=is_async):
            return
        span.end()
        return
    try:
        _record_message(span, result)
    finally:
        span.end()


def enable() -> None:
    """Enable auto-tracing for Anthropic SDK calls, sync and async.

    1.89.0: ``messages.create(stream=True)`` keeps its span open until the
    stream ends, so usage is recorded and an error raised part-way through ends
    the span ERROR (scrubbed of credentials). ``AsyncAnthropic`` is traced too,
    and so is the ``messages.stream(...)`` helper (sync and async), which posts
    the request itself rather than through ``create``: its span opens when the
    ``with`` block enters and follows the same contract.
    """
    global _original_create, _original_acreate, _original_stream, _original_astream, _enabled
    if _enabled:
        return

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK is required. Install with: pip install fastaiagent[anthropic]"
        )

    from fastaiagent.integrations._stream_span import (
        client_secrets,
        mark_error_scrubbed,
        use_span,
        wrap_stream,
    )

    messages_mod = anthropic.resources.messages
    _original_create = messages_mod.Messages.create
    _original_acreate = messages_mod.AsyncMessages.create
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

    _original_stream = messages_mod.Messages.stream
    _original_astream = messages_mod.AsyncMessages.stream
    original_stream = _original_stream
    original_astream = _original_astream

    @functools.wraps(original_stream)
    def traced_stream(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        manager = original_stream(self_inner, *args, **kwargs)
        attr = "_MessageStreamManager__api_request"
        api_request = getattr(manager, attr, None)
        if api_request is None:  # pragma: no cover - a future SDK shape
            logger.debug("anthropic messages.stream(): request not found; not traced")
            return manager
        secrets = client_secrets(self_inner)

        def request() -> Any:
            span = _open_span(kwargs)
            try:
                with use_span(span):
                    raw = api_request()
            except Exception as exc:
                mark_error_scrubbed(span, exc, secrets)
                span.end()
                raise
            except BaseException:
                span.end()
                raise
            if not wrap_stream(raw, span, _EventObserver(), secrets, is_async=False):
                span.end()
            return raw

        setattr(manager, attr, request)
        return manager

    @functools.wraps(original_astream)
    def traced_astream(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        manager = original_astream(self_inner, *args, **kwargs)
        attr = "_AsyncMessageStreamManager__api_request"
        api_request = getattr(manager, attr, None)
        if api_request is None:  # pragma: no cover - a future SDK shape
            logger.debug("anthropic messages.stream(): request not found; not traced")
            return manager
        secrets = client_secrets(self_inner)

        async def request() -> Any:
            span = _open_span(kwargs)
            try:
                with use_span(span):
                    raw = await api_request
            except Exception as exc:
                mark_error_scrubbed(span, exc, secrets)
                span.end()
                raise
            except BaseException:
                span.end()
                raise
            if not wrap_stream(raw, span, _EventObserver(), secrets, is_async=True):
                span.end()
            return raw

        # The manager awaits this attribute in ``__aenter__``.
        setattr(manager, attr, request())
        return manager

    messages_mod.Messages.create = traced_create  # type: ignore[method-assign,assignment]
    messages_mod.AsyncMessages.create = traced_acreate  # type: ignore[method-assign,assignment]
    messages_mod.Messages.stream = traced_stream  # type: ignore[method-assign,assignment]
    messages_mod.AsyncMessages.stream = traced_astream  # type: ignore[method-assign,assignment]
    _enabled = True


def disable() -> None:
    """Disable auto-tracing."""
    global _original_create, _original_acreate, _original_stream, _original_astream, _enabled
    if not _enabled:
        return
    try:
        import anthropic

        messages_mod = anthropic.resources.messages
        if _original_create:
            messages_mod.Messages.create = _original_create  # type: ignore[method-assign]
        if _original_acreate:
            messages_mod.AsyncMessages.create = _original_acreate  # type: ignore[method-assign]
        if _original_stream:
            messages_mod.Messages.stream = _original_stream  # type: ignore[method-assign]
        if _original_astream:
            messages_mod.AsyncMessages.stream = _original_astream  # type: ignore[method-assign]
    except ImportError:
        logger.debug("Anthropic SDK not available during disable", exc_info=True)
    _enabled = False
    _original_create = None
    _original_acreate = None
    _original_stream = None
    _original_astream = None
