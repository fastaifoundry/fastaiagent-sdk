"""Auto-tracing for the OpenAI SDK."""

from __future__ import annotations

import functools
import json
from typing import Any

_original_create = None
_original_acreate = None
_enabled = False


def _serialize_messages(messages: Any) -> str | None:
    if messages is None:
        return None
    try:
        return json.dumps(messages, default=str)
    except Exception:
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
        pass
    return content, tool_calls_json, finish_reason


def enable() -> None:
    """Enable auto-tracing for OpenAI SDK calls."""
    global _original_create, _original_acreate, _enabled
    if _enabled:
        return

    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI SDK is required. Install with: pip install fastaiagent[openai]")

    completions_cls = openai.resources.chat.completions.Completions

    _original_create = completions_cls.create

    @functools.wraps(_original_create)
    def traced_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import set_genai_attributes

        tracer = get_tracer("fastaiagent.integrations.openai")
        model = kwargs.get("model", "unknown")
        with tracer.start_as_current_span(f"openai.chat.{model}") as span:
            set_genai_attributes(
                span,
                system="openai",
                model=model,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                request_messages=_serialize_messages(kwargs.get("messages")),
                request_tools=_serialize_messages(kwargs.get("tools")),
            )
            result = _original_create(self_inner, *args, **kwargs)
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
            return result

    completions_cls.create = traced_create  # type: ignore[assignment]
    _enabled = True


def disable() -> None:
    """Disable auto-tracing for OpenAI SDK calls."""
    global _original_create, _enabled
    if not _enabled:
        return

    try:
        import openai

        if _original_create:
            openai.resources.chat.completions.Completions.create = _original_create  # type: ignore[method-assign]
    except ImportError:
        pass

    _enabled = False
    _original_create = None
