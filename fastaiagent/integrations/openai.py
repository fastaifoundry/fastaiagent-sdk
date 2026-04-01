"""Auto-tracing for the OpenAI SDK."""

from __future__ import annotations

import functools
from typing import Any

_original_create = None
_original_acreate = None
_enabled = False


def enable() -> None:
    """Enable auto-tracing for OpenAI SDK calls."""
    global _original_create, _original_acreate, _enabled
    if _enabled:
        return

    try:
        import openai  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "OpenAI SDK is required. Install with: pip install fastaiagent[openai]"
        )

    completions_cls = openai.resources.chat.completions.Completions

    _original_create = completions_cls.create

    @functools.wraps(_original_create)
    def traced_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import set_genai_attributes

        tracer = get_tracer("fastaiagent.integrations.openai")
        model = kwargs.get("model", "unknown")
        with tracer.start_as_current_span(f"openai.chat.{model}") as span:
            set_genai_attributes(span, system="openai", model=model)
            result = _original_create(self_inner, *args, **kwargs)
            if hasattr(result, "usage") and result.usage:
                set_genai_attributes(
                    span,
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                )
            return result

    completions_cls.create = traced_create
    _enabled = True


def disable() -> None:
    """Disable auto-tracing for OpenAI SDK calls."""
    global _original_create, _enabled
    if not _enabled:
        return

    try:
        import openai  # type: ignore[import-not-found]

        if _original_create:
            openai.resources.chat.completions.Completions.create = _original_create
    except ImportError:
        pass

    _enabled = False
    _original_create = None
