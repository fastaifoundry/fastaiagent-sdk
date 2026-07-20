"""Runtime prompt provenance (Gap 4).

A context-local carrier for "which registry prompt is this run using," set by
the agent around its LLM invocation and read by ``LLMClient`` to stamp
``fastaiagent.prompt.*`` on the ``llm_call`` span. A ContextVar keeps it
async-task-local (concurrent runs don't cross-contaminate) with no changes to
the LLM call signatures.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any

# {"slug": str, "version": int | None, "environment": str | None} or None.
_current_prompt_provenance: ContextVar[dict[str, Any] | None] = ContextVar(
    "fastaiagent_prompt_provenance", default=None
)


def set_prompt_provenance(provenance: dict[str, Any] | None) -> Token[dict[str, Any] | None]:
    """Set the active prompt provenance; returns a token for :func:`reset`."""
    return _current_prompt_provenance.set(provenance)


def reset_prompt_provenance(token: Token[dict[str, Any] | None]) -> None:
    """Restore the previous provenance (call in a ``finally``)."""
    _current_prompt_provenance.reset(token)


def get_prompt_provenance() -> dict[str, Any] | None:
    """Return the active prompt provenance, if any."""
    return _current_prompt_provenance.get()
