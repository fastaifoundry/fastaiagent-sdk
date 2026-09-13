"""Execute a runner command — ``live_playground`` and ``eval_run`` (v1).

The command payload carries the agent **config** (``Agent.to_dict()`` — prompt,
model ref, tool *specs*, KB refs) plus the input — never tools or keys. The
runner reconstructs via ``Agent.from_dict`` and runs inside a ``job_scope`` so
the agent binds the customer's LOCAL tools / keys / KB in the runner's own
boundary. That is the runner's whole reason to exist.

``eval_run`` runs the agent once per case and returns the per-case outputs +
trace ids; the plane scores centrally from each case's ``criteria`` (the runner
does NOT score). ``tool_exec`` runs one LOCAL connector/tool the plane dispatches
(the SaaS + ``customer_private`` case): it resolves the tool by ``exposed_name``
in the ToolRegistry and runs it with the operator's own creds. A
``guarded_live_rerun`` is not handled here.

**Durability (audit D4).** Agent-running commands get a per-job checkpointer and
run under the plane's own ``command_id`` as their ``execution_id``, so a
platform-initiated run survives a crash on this host and shows up in the plane's
Durability view. Before this the whole package contained no checkpoint code:
every playground click and eval case was non-durable, and the plane could
dispatch work and then hold no record of its state. ``tool_exec`` is deliberately
excluded — one tool call, no agent loop, nothing to check point.

What this does **not** buy is cross-runner resume: if the runner process dies,
another cannot pick the run up. The runner channel documents "no cross-runner
reassignment in v1" and the frozen command payload carries no execution id, so
that half needs a plane wire bump. Same-runner resume and visibility are the
whole of it.
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


def _job_checkpointer() -> Any | None:
    """A checkpointer for one platform-initiated job, or None when disabled.

    Until this existed ``fastaiagent/runner/`` contained no checkpoint code at
    all (audit D4): ``Agent.from_dict`` drops the checkpointer and ``arun`` was
    called with no ``execution_id``, so every run the plane dispatched — live
    playground, eval suites — was non-durable and invisible in the plane's
    Durability view. The plane could dispatch work and then hold no record of
    its state.

    ⚠ **This turns on local disk writes and plane ingest on hosts that had
    neither**, so it has an off switch: ``FASTAIAGENT_RUNNER_CHECKPOINTS=0``.
    Default-on because a runner that loses a run on a crash is the thing being
    fixed; the switch is for operators who want the old footprint back.

    Best-effort: a checkpointer that cannot be built must not stop the job. A
    non-durable run is worse than a durable one, and far better than none.
    """
    import os

    if os.environ.get("FASTAIAGENT_RUNNER_CHECKPOINTS") == "0":
        return None
    try:
        from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

        store = SQLiteCheckpointer()
        store.setup()
        return store
    except Exception:
        logger.warning(
            "Could not open a checkpointer for this job — it will run without "
            "durability. Set FASTAIAGENT_RUNNER_CHECKPOINTS=0 to silence this.",
            exc_info=True,
        )
        return None


def _execution_id_for(cmd: dict[str, Any], suffix: str = "") -> str | None:
    """Use the plane's own ``command_id`` as the run's ``execution_id``.

    This is what makes a runner's checkpoints joinable to the command that
    caused them **with no wire change**. ``CommandResult`` is the frozen shape
    reported back to the plane, and adding a field to it would be a
    plane-observable change; reusing an id the plane already knows costs
    nothing and says the same thing.

    Returns None when the command carries no id — then the agent mints its own
    UUID exactly as before, and the run is durable locally but not joinable.
    """
    command_id = cmd.get("command_id")
    if not command_id:
        return None
    # The plane's execution_id column holds 255 characters; a UUID plus a case
    # id is nowhere near it, but a caller-supplied case id is not ours to trust.
    return f"{command_id}{suffix}"[:255]


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
    if ctype == "tool_exec":
        return await _run_tool_exec(cmd)
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
        # Durability for a platform-initiated run (audit D4). The execution_id is
        # the plane's own command_id, so the replica joins back to the command
        # without any wire change.
        agent = Agent.from_dict(agent_config, checkpointer=_job_checkpointer())
        # job_scope isolates the per-job tool registry (one job == one asyncio
        # task, which the daemon guarantees). We do NOT override the project: the
        # plane routes traces by the runner's API key, and the trace exporter
        # drains on a background thread that can't see a per-job ContextVar, so a
        # per-job project would hide the spans from the drain. Letting the
        # process-global project stand keeps the span stamp and the drain filter
        # consistent — and the checkpoint outbox drains on the same terms.
        with job_scope():
            result = await agent.arun(user_input, execution_id=_execution_id_for(cmd))
        return CommandResult("completed", result.output, getattr(result, "trace_id", None), None)
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
        agent = Agent.from_dict(agent_config, checkpointer=_job_checkpointer())
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
                # One execution per CASE, not per command: a suite is N runs, and
                # collapsing them onto one id would interleave their checkpoints
                # into a history no resume could read.
                result = await agent.arun(
                    case.get("input", ""),
                    execution_id=_execution_id_for(cmd, f"-{case_id}" if case_id else ""),
                )
            trace_id = getattr(result, "trace_id", None)
            outputs.append({"case_id": case_id, "output": result.output, "trace_id": trace_id})
        except Exception:  # noqa: BLE001 — one bad case shouldn't fail the suite
            logger.exception("eval_run command %s case %s failed", cmd.get("command_id"), case_id)
            outputs.append({"case_id": case_id, "output": "", "trace_id": None})
        if first_trace_id is None and trace_id:
            first_trace_id = trace_id

    return CommandResult("completed", {"outputs": outputs}, first_trace_id, None)


async def _run_tool_exec(cmd: dict[str, Any]) -> CommandResult:
    """Run one LOCAL connector/tool the plane dispatched.

    Payload: ``{"tool_exec": {"tool_type", "connector": {"instance_id", "action",
    "fixed_params"}, "exposed_name", "arguments"}, "hosted_server_id"}``. We
    resolve the tool by ``exposed_name`` in the runner's ToolRegistry (the
    operator registered it locally with its own creds), run it inside a traced
    ``tool.<name>`` span, and report ``{"success", "result"}`` (the plane reads
    ``result["success"]``). ``status`` is ``completed`` whenever the tool ran —
    even if it returned an error (``success=false``) — and ``failed`` only when
    the runner can't resolve/process the command. The span is pushed like any
    other job's trace and linked by the reported ``trace_id``.
    """
    import json

    payload = cmd.get("payload") or {}
    spec = payload.get("tool_exec") or {}
    tool_type = spec.get("tool_type")
    if tool_type != "connector":
        return CommandResult(
            "failed", None, None, f"unsupported tool_exec tool_type: {tool_type!r}"
        )

    exposed_name = spec.get("exposed_name")
    if not exposed_name:
        return CommandResult("failed", None, None, "tool_exec payload missing 'exposed_name'")
    arguments = spec.get("arguments") or {}
    fixed_params = (spec.get("connector") or {}).get("fixed_params") or {}
    # Operator-fixed params win over the LLM-supplied arguments on a key collision.
    call_args = {**arguments, **fixed_params}

    from fastaiagent import job_scope
    from fastaiagent.tool.registry import ToolRegistry
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import trace_payloads_enabled

    tool = ToolRegistry.get(exposed_name)
    if tool is None:
        return CommandResult(
            "failed",
            None,
            None,
            f"no local tool registered for exposed_name {exposed_name!r} "
            "(register it before starting the runner, e.g. via --tools)",
        )

    tracer = get_tracer("fastaiagent.runner.tool_exec")
    try:
        # job_scope() keeps the span's project consistent with the exporter's
        # background-thread drain (same rationale as _run_live_playground).
        with job_scope(), tracer.start_as_current_span(f"tool.{exposed_name}") as span:
            span.set_attribute("tool.name", exposed_name)
            span.set_attribute("tool.origin", getattr(tool, "origin", "unknown"))
            span.set_attribute("fastaiagent.runner.type", "tool")
            span.set_attribute(
                "fastaiagent.tool.replay_class", getattr(tool, "replay_class", "side_effecting")
            )
            if trace_payloads_enabled():
                span.set_attribute("tool.args", json.dumps(call_args, default=str))
            try:
                tool_result = await tool.aexecute(call_args, context=None)
                success = bool(tool_result.success)
                output: Any = tool_result.output if success else (tool_result.error or "tool error")
                span.set_attribute("tool.status", "ok" if success else "error")
                if not success:
                    span.set_attribute("tool.error", str(tool_result.error))
            except Exception as e:  # noqa: BLE001 — a connector failure → success=false, not a crash
                logger.exception("tool_exec command %s tool raised", cmd.get("command_id"))
                success, output = False, f"{type(e).__name__}: {e}"
                span.set_attribute("tool.status", "error")
                span.set_attribute("tool.error", str(e))
            if trace_payloads_enabled():
                span.set_attribute("tool.result", str(output))
            trace_id = format(span.get_span_context().trace_id, "032x")

        # Keep the reported result JSON-serializable for the results channel.
        try:
            json.dumps(output)
        except (TypeError, ValueError):
            output = str(output)
        return CommandResult("completed", {"success": success, "result": output}, trace_id, None)
    except Exception as e:  # noqa: BLE001 — runner-level failure (never crash the daemon)
        logger.exception("tool_exec command %s failed", cmd.get("command_id"))
        return CommandResult("failed", None, None, str(e))
