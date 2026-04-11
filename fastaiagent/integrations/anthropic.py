"""Auto-tracing for the Anthropic SDK."""

from __future__ import annotations

import functools
import json
from typing import Any

_original_create = None
_enabled = False


def _serialize_json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except Exception:
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
        pass
    content = "".join(text_parts) if text_parts else None
    tool_calls_json = json.dumps(tool_calls, default=str) if tool_calls else None
    stop_reason = getattr(result, "stop_reason", None)
    return content, tool_calls_json, stop_reason


def enable() -> None:
    """Enable auto-tracing for Anthropic SDK calls."""
    global _original_create, _enabled
    if _enabled:
        return

    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "Anthropic SDK is required. Install with: pip install fastaiagent[anthropic]"
        )

    messages_cls = anthropic.resources.messages.Messages
    _original_create = messages_cls.create

    @functools.wraps(_original_create)
    def traced_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import set_genai_attributes

        tracer = get_tracer("fastaiagent.integrations.anthropic")
        model = kwargs.get("model", "unknown")
        with tracer.start_as_current_span(f"anthropic.messages.{model}") as span:
            set_genai_attributes(
                span,
                system="anthropic",
                model=model,
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                request_messages=_serialize_json(kwargs.get("messages")),
                request_tools=_serialize_json(kwargs.get("tools")),
            )
            result = _original_create(self_inner, *args, **kwargs)
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
            return result

    messages_cls.create = traced_create
    _enabled = True


def disable() -> None:
    """Disable auto-tracing."""
    global _original_create, _enabled
    if not _enabled:
        return
    try:
        import anthropic

        if _original_create:
            anthropic.resources.messages.Messages.create = _original_create
    except ImportError:
        pass
    _enabled = False
    _original_create = None
