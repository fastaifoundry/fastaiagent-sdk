"""Auto-tracing for the Anthropic SDK."""

from __future__ import annotations

import functools
from typing import Any

_original_create = None
_enabled = False


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
            set_genai_attributes(span, system="anthropic", model=model)
            result = _original_create(self_inner, *args, **kwargs)
            if hasattr(result, "usage") and result.usage:
                set_genai_attributes(
                    span,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
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
