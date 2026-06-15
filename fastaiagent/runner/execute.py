"""Execute a runner command — ``live_playground`` and ``eval_run`` (v1).

The command payload carries the agent **config** (``Agent.to_dict()`` — prompt,
model ref, tool *specs*, KB refs) plus the input — never tools or keys. The
runner reconstructs via ``Agent.from_dict`` and runs inside a ``job_scope`` so
the agent binds the customer's LOCAL tools / keys / KB in the runner's own
boundary. That is the runner's whole reason to exist.

``eval_run`` runs the agent once per case and returns the per-case outputs +
trace ids; the plane scores centrally from each case's ``criteria`` (the runner
does NOT score). A ``guarded_live_rerun`` / ``tool_exec`` is not handled here.
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
    if ctype == "live_playground":
        return await _run_live_playground(cmd)
    if ctype == "eval_run":
        return await _run_eval_run(cmd)
    return CommandResult("failed", None, None, f"unsupported command type: {ctype!r}")


async def _run_live_playground(cmd: dict[str, Any]) -> CommandResult:
    payload = cmd.get("payload") or {}
    agent_config = payload.get("agent")
    user_input = payload.get("input", "")
    if not agent_config:
        return CommandResult("failed", None, None, "command payload missing 'agent' config")

    # Imported lazily so importing the runner package is cheap.
    from fastaiagent import job_scope
    from fastaiagent.agent.agent import Agent

    try:
        agent = Agent.from_dict(agent_config)
        # job_scope isolates the per-job tool registry (one job == one asyncio
        # task, which the daemon guarantees). We do NOT override the project: the
        # plane routes traces by the runner's API key, and the trace exporter
        # drains on a background thread that can't see a per-job ContextVar, so a
        # per-job project would hide the spans from the drain. Letting the
        # process-global project stand keeps the span stamp and the drain filter
        # consistent.
        with job_scope():
            result = await agent.arun(user_input)
        return CommandResult(
            "completed", result.output, getattr(result, "trace_id", None), None
        )
    except Exception as e:  # noqa: BLE001 — report as failed, never crash the daemon
        logger.exception("live_playground command %s failed", cmd.get("command_id"))
        return CommandResult("failed", None, None, str(e))


async def _run_eval_run(cmd: dict[str, Any]) -> CommandResult:
    """Run the agent once per case; return per-case outputs + trace ids.

    The plane scores centrally from each case's ``criteria`` — the runner only
    executes. Result shape (frozen): ``{"outputs": [{"case_id", "output",
    "trace_id"}, ...]}`` in case order. A single case failing doesn't fail the
    command (its output is empty + logged); a missing agent config does.
    """
    payload = cmd.get("payload") or {}
    agent_config = payload.get("agent")
    cases = payload.get("cases") or []
    if not agent_config:
        return CommandResult("failed", None, None, "command payload missing 'agent' config")

    from fastaiagent import job_scope
    from fastaiagent.agent.agent import Agent

    try:
        agent = Agent.from_dict(agent_config)
    except Exception as e:  # noqa: BLE001 — can't build the agent → whole command fails
        logger.exception("eval_run command %s failed to build agent", cmd.get("command_id"))
        return CommandResult("failed", None, None, str(e))

    outputs: list[dict[str, Any]] = []
    first_trace_id: str | None = None
    for case in cases:
        case_id = case.get("case_id")
        trace_id: str | None = None
        try:
            with job_scope():  # see _run_live_playground re: no project override
                result = await agent.arun(case.get("input", ""))
            trace_id = getattr(result, "trace_id", None)
            outputs.append({"case_id": case_id, "output": result.output, "trace_id": trace_id})
        except Exception:  # noqa: BLE001 — one bad case shouldn't fail the suite
            logger.exception(
                "eval_run command %s case %s failed", cmd.get("command_id"), case_id
            )
            outputs.append({"case_id": case_id, "output": "", "trace_id": None})
        if first_trace_id is None and trace_id:
            first_trace_id = trace_id

    return CommandResult("completed", {"outputs": outputs}, first_trace_id, None)
