"""Execute a runner command. v1 handles ``live_playground`` only.

The command payload carries the agent **config** (``Agent.to_dict()`` — prompt,
model ref, tool *specs*, KB refs) plus the input — never tools or keys. The
runner reconstructs via ``Agent.from_dict`` and runs inside a ``job_scope`` so
the agent binds the customer's LOCAL tools / keys / KB in the runner's own
boundary. That is the runner's whole reason to exist.

A ``guarded_live_rerun`` (a fast-follow, not handled here) must gate on the 2.1
``replay_class`` and NEVER execute a ``side_effecting`` tool.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


class CommandResult(NamedTuple):
    status: str  # "completed" | "failed"
    result: Any | None
    trace_id: str | None
    error: str | None


async def execute_command(cmd: dict[str, Any]) -> CommandResult:
    """Run a single command and return a :class:`CommandResult`.

    Never raises: any failure is returned as ``status="failed"`` so the daemon
    reports it and keeps serving.
    """
    ctype = cmd.get("type")
    if ctype != "live_playground":
        return CommandResult("failed", None, None, f"unsupported command type: {ctype!r}")

    payload = cmd.get("payload") or {}
    agent_config = payload.get("agent")
    user_input = payload.get("input", "")
    if not agent_config:
        return CommandResult("failed", None, None, "command payload missing 'agent' config")

    # Imported lazily so importing the runner package is cheap.
    from fastaiagent import job_scope
    from fastaiagent.agent.agent import Agent

    tenant = cmd.get("tenant")
    try:
        agent = Agent.from_dict(agent_config)
        # Scope the job to its tenant/project; the agent binds the runner's
        # local tools/keys. One job == one asyncio task (the daemon guarantees
        # this) so the ContextVar isolation holds.
        with job_scope(project=tenant):
            result = await agent.arun(user_input)
        return CommandResult(
            "completed", result.output, getattr(result, "trace_id", None), None
        )
    except Exception as e:  # noqa: BLE001 — report as failed, never crash the daemon
        logger.exception("live_playground command %s failed", cmd.get("command_id"))
        return CommandResult("failed", None, None, str(e))
