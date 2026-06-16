"""Runner ``tool_exec`` execution (Task B) — real local tool, no mocks, no LLM.

A locally-registered ``FunctionTool`` is resolved by its ``exposed_name`` and run
by the runner's ``execute_command`` for a dispatched ``tool_exec`` command. We
assert the frozen ``{"success", "result"}`` result shape, that the tool call is
traced as a ``tool.<name>`` span, ``fixed_params`` merging, and the failure modes.
No LLM/keys needed, so this runs in the normal CI lane.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastaiagent.runner.execute import execute_command
from fastaiagent.tool.function import FunctionTool


def _cmd(exposed_name: str, arguments: dict, fixed_params: dict | None = None) -> dict:
    return {
        "command_id": "te-1",
        "type": "tool_exec",
        "tenant": "dom-1",
        "payload": {
            "tool_exec": {
                "tool_type": "connector",
                "connector": {
                    "instance_id": "inst-1",
                    "action": "run",
                    "fixed_params": fixed_params or {},
                },
                "exposed_name": exposed_name,
                "arguments": arguments,
            },
            "hosted_server_id": "srv-1",
        },
    }


def _run(cmd: dict) -> Any:
    async def drive() -> Any:
        # Own asyncio task — the daemon's per-job invariant.
        return await asyncio.create_task(execute_command(cmd))

    return asyncio.run(drive())


def test_tool_exec_runs_local_tool_and_traces(isolated_local_db) -> None:
    from fastaiagent.trace import otel
    from fastaiagent.trace.storage import TraceStore

    otel.reset()  # fresh provider → spans land in the isolated db

    def echo_upper(text: str, suffix: str = "") -> str:
        return text.upper() + suffix

    FunctionTool(name="echo_upper", fn=echo_upper)  # auto-registers in ToolRegistry

    res = _run(_cmd("echo_upper", {"text": "hi"}, fixed_params={"suffix": "!"}))

    assert res.status == "completed", res
    assert res.result == {"success": True, "result": "HI!"}  # fixed_params merged in
    assert res.trace_id and len(res.trace_id) == 32

    # The tool call was traced (and would be pushed by the exporter like any job).
    store = TraceStore()
    rows = store._db.fetchall(
        "SELECT name FROM spans WHERE trace_id = ?", (res.trace_id,)
    )
    assert any(r["name"] == "tool.echo_upper" for r in rows), [dict(r) for r in rows]
    store.close()


def test_tool_exec_tool_error_is_completed_with_success_false(isolated_local_db) -> None:
    from fastaiagent.trace import otel

    otel.reset()

    def boom(x: int) -> str:
        raise ValueError("kaboom")

    FunctionTool(name="boom", fn=boom)

    res = _run(_cmd("boom", {"x": 1}))

    # The runner processed the command (completed); the connector failed (success=false).
    assert res.status == "completed", res
    assert res.result["success"] is False
    assert "kaboom" in str(res.result["result"])


def test_tool_exec_unknown_tool_fails(isolated_local_db) -> None:
    res = _run(_cmd("nonexistent_tool", {"x": 1}))
    assert res.status == "failed"
    assert "no local tool" in (res.error or "")


def test_tool_exec_unsupported_tool_type_fails(isolated_local_db) -> None:
    cmd = _cmd("echo_upper", {})
    cmd["payload"]["tool_exec"]["tool_type"] = "rest"
    res = _run(cmd)
    assert res.status == "failed"
    assert "tool_type" in (res.error or "")
