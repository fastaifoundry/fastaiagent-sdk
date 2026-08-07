"""Optional adapters that bolt FastAIAgent tracing onto third-party agent
frameworks. Each submodule is lazy-loaded so installing
``fastaiagent`` doesn't pull in ``langchain`` / ``crewai`` / etc. unless
the user actually touches that integration.

Used like::

    import fastaiagent
    fastaiagent.integrations.langchain.enable()
"""

from __future__ import annotations

from typing import Any

# ``agent_name`` is a plain context manager with no third-party dependency, so
# it is imported eagerly — it is the framework-agnostic way to name a run for
# control-plane linkage when you are not using ``with_guardrails(name=...)``.
from fastaiagent.integrations._identity import agent_name

_SUBMODULES = ["langchain", "crewai", "pydanticai", "anthropic", "openai"]

__all__ = [*_SUBMODULES, "agent_name"]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"fastaiagent.integrations.{name}")
    raise AttributeError(
        f"module 'fastaiagent.integrations' has no attribute {name!r}"
    )
