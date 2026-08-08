"""Runtime agent identity for foreign-framework runs.

A context-local carrier for "which agent is this run," set by the harness
proxies (``with_guardrails``) or explicitly by the user, and read by each
integration when it opens a *root* span so it can stamp ``agent.name``.

Why this exists: the control plane resolves a trace to an agent by name,
checking ``agent.name`` / ``chain.name`` / ``swarm.name`` / ``workflow.name`` on
the root span first and only then falling back to prefix-stripping the span
name. Foreign root spans are named from the framework's own serialization
(``langchain.chain``, ``crewai.crew.crew``, ``pydanticai.agent.self``), which
carries no agent identity — so without this the plane derives a generic,
colliding name and the trace never links.

Native agents already set ``agent.name`` directly (``agent/agent.py``); this
brings foreign runs to parity.

A ContextVar keeps it async-task-local so concurrent runs don't cross-
contaminate, with no change to any framework's call signatures. Setting it
*outside* the delegated call is safe under LCEL's ``copy_context().run(...)``
step isolation: a copied context inherits values set before the copy.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

logger = logging.getLogger(__name__)

_current_agent_name: ContextVar[str | None] = ContextVar(
    "fastaiagent_agent_name", default=None
)

# One warning per process per framework — never per run.
_warned: set[str] = set()


def set_agent_name(name: str | None) -> Token[str | None]:
    """Set the active agent name; returns a token for :func:`reset_agent_name`."""
    return _current_agent_name.set(name)


def reset_agent_name(token: Token[str | None]) -> None:
    """Restore the previous agent name (call in a ``finally``)."""
    _current_agent_name.reset(token)


def get_agent_name() -> str | None:
    """Return the active agent name, if any."""
    return _current_agent_name.get()


@contextmanager
def agent_name(name: str) -> Iterator[None]:
    """Name the agent for every root span opened inside this block.

    For code that does not go through ``with_guardrails(...)``::

        from fastaiagent.integrations import agent_name

        with agent_name("support-bot"):
            graph.invoke({"messages": [("user", "hi")]})
    """
    token = set_agent_name(name)
    try:
        yield
    finally:
        reset_agent_name(token)


def stamp_agent_name(span: Any, framework: str) -> None:
    """Stamp ``agent.name`` on a root span from the active context.

    No-ops when no name is set. Warns once per framework when the SDK is
    connected to a control plane but the run is unnamed, because that is
    exactly the case where the trace will not link to an agent.
    """
    name = get_agent_name()
    if name:
        try:
            span.set_attribute("agent.name", name)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Could not stamp agent.name", exc_info=True)
        return

    if framework in _warned:
        return
    try:
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return
    except Exception:  # pragma: no cover - defensive
        return
    _warned.add(framework)
    logger.warning(
        "%s run is not named, so its trace cannot be linked to an agent on the "
        "control plane (the root span carries no agent.name and the plane will "
        "fall back to a generic derived name). Pass name= to "
        "fastaiagent.integrations.%s.with_guardrails(...), or wrap the run in "
        "fastaiagent.integrations.agent_name('my-agent').",
        framework,
        framework,
    )


def reset_warned_for_tests() -> None:
    """Clear the once-per-process warning state (tests only)."""
    _warned.clear()
