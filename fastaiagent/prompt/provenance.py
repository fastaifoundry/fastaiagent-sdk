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

# {"name": str, "version": int | None} — plus "slug" and "environment" for a
# control-plane prompt — or None when the running agent uses no registry prompt.
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


def stamp_prompt_provenance(span: Any) -> None:
    """Stamp ``fastaiagent.prompt.*`` on an ``llm.*`` span from the active provenance.

    ``prompt.name`` and ``prompt.version`` for any registry prompt — the Local
    UI's prompt lineage finds a prompt's traces by ``prompt.name`` — plus
    ``prompt.slug`` and ``prompt.environment`` for a control-plane one, which
    Prompt Analytics reads. A no-op when the running agent uses no registry
    prompt. Shared by ``LLMClient`` and the offline test models, so a run is
    attributed the same way whichever model served it.
    """
    from fastaiagent.trace.span import set_fastaiagent_attributes

    prov = _current_prompt_provenance.get()
    if prov and (prov.get("name") or prov.get("slug")):
        set_fastaiagent_attributes(
            span,
            **{
                "prompt.name": prov.get("name"),
                "prompt.slug": prov.get("slug"),
                "prompt.version": prov.get("version"),
                "prompt.environment": prov.get("environment"),
            },
        )
