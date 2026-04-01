"""Auto-tracing for LangChain via callback handler."""

from __future__ import annotations

from typing import Any

_enabled = False


def enable() -> None:
    """Enable auto-tracing for LangChain by registering a callback handler."""
    global _enabled
    if _enabled:
        return

    try:
        import langchain_core.callbacks  # noqa: F401
    except ImportError:
        raise ImportError(
            "langchain-core is required. Install with: pip install fastaiagent[langchain]"
        )

    _enabled = True


def disable() -> None:
    """Disable auto-tracing for LangChain."""
    global _enabled
    _enabled = False


def get_callback_handler() -> Any:
    """Get the FastAIAgent LangChain callback handler."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        raise ImportError(
            "langchain-core is required. Install with: pip install fastaiagent[langchain]"
        )

    class FastAIAgentCallbackHandler(BaseCallbackHandler):  # type: ignore[misc]
        """LangChain callback handler that emits OTel spans."""

        def on_llm_start(
            self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
        ) -> None:
            from fastaiagent.trace.otel import get_tracer

            tracer = get_tracer("fastaiagent.integrations.langchain")
            model = serialized.get("name", "langchain_llm")
            span = tracer.start_span(f"langchain.llm.{model}")
            kwargs.setdefault("_fastai_spans", []).append(span)

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            spans = kwargs.get("_fastai_spans", [])
            if spans:
                spans[-1].end()

        def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
            from fastaiagent.trace.otel import get_tracer

            tracer = get_tracer("fastaiagent.integrations.langchain")
            tool_name = serialized.get("name", "tool")
            span = tracer.start_span(f"langchain.tool.{tool_name}")
            kwargs.setdefault("_fastai_spans", []).append(span)

        def on_tool_end(self, output: str, **kwargs: Any) -> None:
            spans = kwargs.get("_fastai_spans", [])
            if spans:
                spans[-1].end()

    return FastAIAgentCallbackHandler()
