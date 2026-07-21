"""Push agent/chain definitions to the plane so they become governed console
objects (``managed_by=sdk``).

This is the SDK-owned registration path — it replaces hand-written
``httpx.post(...)`` calls in user code. Three entry points, all routing through
:func:`push_agent`:

* ``agent.push()`` / ``fa.push(agent)`` — explicit (CI/deploy or on-demand).
* ``connect()`` flush — registers every agent that already exists at connect
  time (see :func:`flush_pending_registrations`).
* first-run auto-register — an agent born after ``connect()`` registers itself
  the first time it runs (see :func:`auto_register_async`).

Registration is **idempotent per process** (guarded by a name-keyed set) and
**best-effort** on the auto paths — a failure logs once and never breaks a run.
The plane upserts by ``name`` within the project, so re-runs never duplicate and
a changed definition on the next process resyncs automatically.
"""

from __future__ import annotations

import logging
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastaiagent.agent.agent import Agent

logger = logging.getLogger(__name__)

# Process-global registration state. Guarded by ``_lock`` because agents can be
# constructed / run from multiple threads (e.g. a runner worker pool).
_lock = threading.Lock()
# Agents that have registered themselves on construction. Weak so holding them
# here never keeps a short-lived agent alive.
_agent_registry: weakref.WeakSet[Agent] = weakref.WeakSet()
# Names already pushed this process → the push result. Name-keyed (not
# per-object) so a server that builds a fresh Agent("X") per request pushes
# only on the first request. See the plan's Gap 2 idempotency table.
_pushed: dict[str, PushResult] = {}
# Names we've already warned about (unregistered-while-connected) — warn once.
_warned: set[str] = set()


@dataclass
class PushResult:
    """Outcome of registering an agent/chain with the plane."""

    agent_id: str | None
    name: str
    version: int | None = None
    url: str | None = None
    skipped: bool = False  # True when returned from the idempotency cache


def track_agent(agent: Agent) -> None:
    """Record a freshly-constructed agent so ``connect()`` can flush it.

    Called from ``Agent.__init__``. Cheap, additive, and never raises — a
    tracking failure must not break agent construction.
    """
    try:
        _agent_registry.add(agent)
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not track agent for registration", exc_info=True)


def _console_base() -> str | None:
    """Resolve the console origin for building deep links.

    Defaults to the connection ``target`` (correct in production, where the API
    and console share an origin). A split-origin dev setup (console :20000 vs
    API :20001) sets ``connect(console_url=…)`` / ``FASTAIAGENT_CONSOLE_URL``.
    """
    import os

    from fastaiagent.client import _connection

    base = getattr(_connection, "console_url", None) or os.environ.get(
        "FASTAIAGENT_CONSOLE_URL"
    )
    base = base or getattr(_connection, "target", None)
    return base.rstrip("/") if base else None


def _console_url_for_agent(agent_id: str | None) -> str | None:
    """Build the console deep link for a pushed agent.

    Route verified against the Enterprise console (React Router mounted at
    ``/next/*``, agent detail ``agents/:agentId``).
    """
    base = _console_base()
    if not base:
        return None
    if agent_id:
        return f"{base}/next/agents/{agent_id}"
    return f"{base}/next/agents"


def _has_agent_write_scope() -> bool:
    """Whether the connected key can create agents. ``scopes`` may be empty when
    the plane didn't return them (older plane) — treat unknown as allowed so we
    still attempt and let the 403 path handle it."""
    from fastaiagent.client import _connection

    scopes = getattr(_connection, "scopes", None) or []
    if not scopes:
        return True
    return "agent:write" in scopes


def push_agent(
    agent: Agent,
    *,
    force: bool = False,
    best_effort: bool = False,
) -> PushResult | None:
    """Register (upsert) an agent definition with the plane.

    Idempotent per process: if the agent's name was already pushed and
    ``force`` is False, returns the cached :class:`PushResult` (``skipped=True``)
    without a network call — so the 2nd..Nth run costs nothing.

    ``force=True`` (used by explicit ``agent.push()``) always re-pushes and
    refreshes the cache — the escape hatch for a same-process definition change.

    ``best_effort=True`` (the auto paths) swallows errors, logs once, and
    returns ``None`` so a run is never broken. ``best_effort=False`` (explicit
    push) lets platform errors propagate so the caller/CLI sees them.
    """
    from fastaiagent._platform.api import get_platform_api

    name = agent.name
    with _lock:
        if not force and name in _pushed:
            cached = _pushed[name]
            return PushResult(
                agent_id=cached.agent_id,
                name=cached.name,
                version=cached.version,
                url=cached.url,
                skipped=True,
            )

    try:
        payload = agent.to_dict()
        resp = get_platform_api().post("/public/v1/sdk/agents", payload)
        agent_id = resp.get("id") or resp.get("agent_id")
        result = PushResult(
            agent_id=agent_id,
            name=resp.get("name", name),
            version=resp.get("version"),
            url=_console_url_for_agent(agent_id),
        )
        # Link the live agent so governance tool-gating lights up immediately.
        if agent_id and not getattr(agent, "agent_id", None):
            try:
                agent.agent_id = agent_id
            except Exception:  # pragma: no cover - defensive
                logger.debug("Could not stamp agent_id back onto agent", exc_info=True)
        with _lock:
            _pushed[name] = result
        logger.info(
            "Registered agent %r with the plane%s",
            name,
            f" → {result.url}" if result.url else "",
        )
        return result
    except Exception as exc:
        if best_effort:
            _warn_once(
                name,
                f"Agent {name!r} could not be registered with the plane ({exc}). "
                "Its traces won't be grouped by agent. It will retry on the next "
                "run; or call agent.push() / run `fastaiagent push`.",
            )
            return None
        raise


def auto_register_async(agent: Agent) -> None:
    """Kick a best-effort first-run registration on a daemon thread.

    Non-blocking so it never adds latency to ``run()``. Fired as early as
    possible in the traced run; the platform trace exporter batches spans, so
    this quick POST typically lands before the run's trace is ingested (giving
    it an ``agent_id``). No-op when disconnected, opted out, or already pushed.
    """
    from fastaiagent.client import _connection

    if not _connection.is_connected or not getattr(_connection, "auto_register", True):
        # Not auto-registering — nudge the user once so silence never hides it.
        if _connection.is_connected:
            _warn_once(
                agent.name,
                f"Agent {agent.name!r} ran but auto-registration is off, so its "
                "traces won't be grouped by agent. Enable connect(auto_register=True), "
                "call agent.push(), or run `fastaiagent push`.",
            )
        return
    with _lock:
        if agent.name in _pushed:
            return
    if not _has_agent_write_scope():
        _warn_once(
            agent.name,
            f"Agent {agent.name!r} can't register: the API key lacks the "
            "'agent:write' scope. Traces won't be grouped by agent.",
        )
        return

    def _run() -> None:
        push_agent(agent, best_effort=True)

    try:
        threading.Thread(target=_run, daemon=True).start()
    except Exception:  # pragma: no cover - defensive
        logger.debug("Could not kick first-run auto-register", exc_info=True)


def flush_pending_registrations() -> list[str]:
    """Register every agent that exists at ``connect()`` time.

    The strong path for the "define agents → connect() → run" ordering:
    registration completes before any run, so every trace is linked from the
    first. Returns the pushed ``agent_id``s so ``connect()`` can feed
    governance enrollment (closing the ``governed_agent_ids`` gap). Best-effort
    throughout — never raises into ``connect()``.
    """
    from fastaiagent.client import _connection

    if not getattr(_connection, "auto_register", True):
        return []
    if not _has_agent_write_scope():
        agents = list(_agent_registry)
        if agents:
            _warn_once(
                "*",
                "auto_register is on but the API key lacks the 'agent:write' "
                "scope — agents won't be registered. Traces won't group by agent.",
            )
        return []

    agent_ids: list[str] = []
    seen_names: set[str] = set()
    for agent in list(_agent_registry):
        # Soft-warn on same-name collisions: the plane upserts by name, so two
        # distinct live agents sharing a name would clobber each other.
        if agent.name in seen_names:
            logger.warning(
                "Two live agents share the name %r — the plane upserts by name, "
                "so they will overwrite each other. Give them distinct names.",
                agent.name,
            )
        seen_names.add(agent.name)
        result = push_agent(agent, best_effort=True)
        if result and result.agent_id and not result.skipped:
            agent_ids.append(result.agent_id)
    return agent_ids


def _warn_once(key: str, message: str) -> None:
    """Emit ``message`` at most once per process for ``key``."""
    with _lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message)


def reset_registration_state() -> None:
    """Clear process-global push/warn state (on disconnect and for tests).

    Does not clear the weak agent registry — those objects still exist and a
    subsequent connect() should re-register them.
    """
    with _lock:
        _pushed.clear()
        _warned.clear()


# Back-compat/test alias.
reset_for_testing = reset_registration_state
