"""Auto-tracing for the OpenAI and Anthropic SDKs. Each submodule is lazy-loaded,
so installing ``fastaiagent`` does not import either provider SDK until you
use that integration.

Used like::

    import fastaiagent
    fastaiagent.integrations.openai.enable()

Agent frameworks (LangChain/LangGraph, CrewAI, Pydantic AI) have no adapter
here since 1.96.0: trace them through their own OpenTelemetry instrumentation
and ``fastaiagent.enable_otel_capture()``, and use guardrails and evals as
plain functions. See docs/integrations/frameworks.md.
"""

from __future__ import annotations

from typing import Any

_SUBMODULES = ["anthropic", "openai"]

__all__ = list(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"fastaiagent.integrations.{name}")
    raise AttributeError(
        f"module 'fastaiagent.integrations' has no attribute {name!r}"
    )
