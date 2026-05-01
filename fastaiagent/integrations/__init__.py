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

__all__ = ["langchain", "crewai", "anthropic", "openai"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import importlib

        return importlib.import_module(f"fastaiagent.integrations.{name}")
    raise AttributeError(
        f"module 'fastaiagent.integrations' has no attribute {name!r}"
    )
