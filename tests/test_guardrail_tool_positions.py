"""Regression tests for tool_call / tool_result guardrail positions.

These positions are already wired in the agent tool-execution loop
(``fastaiagent/agent/executor.py`` — ``_invoke_tool_with_span`` runs
``execute_guardrails`` at GuardrailPosition.tool_call before a tool runs and
GuardrailPosition.tool_result after). These tests pin that behavior so it can't
silently regress. Real TestModel/FunctionModel agent + real tool, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent._internal.config import get_config
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.agent.agent import Agent
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult, GuardrailType
from fastaiagent.testing.models import FunctionModel
from fastaiagent.tool import FunctionTool
from fastaiagent.ui.db import init_local_db


def _tool_calling_llm(tool_name: str, args: dict) -> FunctionModel:
    """First turn requests the tool; second turn answers — avoids an infinite loop."""
    state = {"n": 0}

    def responder(messages):
        state["n"] += 1
        if state["n"] == 1:
            return "", [{"name": tool_name, "arguments": args}]
        return "all done"

    return FunctionModel(responder)


async def test_tool_call_guardrail_blocks() -> None:
    """A blocking guardrail at tool_call stops execution before the tool runs."""
    ran = {"tool": False}

    def my_tool(payload: str) -> str:
        ran["tool"] = True
        return "tool output"

    tool = FunctionTool(name="my_tool", fn=my_tool, description="echo")

    def block_danger(text: str) -> GuardrailResult:
        if "danger" in text:
            return GuardrailResult(passed=False, message="blocked tool args")
        return GuardrailResult(passed=True)

    guard = Guardrail(
        name="tc_guard",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.tool_call,
        blocking=True,
        fn=block_danger,
    )

    agent = Agent(
        name="tc-agent",
        llm=_tool_calling_llm("my_tool", {"payload": "danger zone"}),
        tools=[tool],
        guardrails=[guard],
    )

    with pytest.raises(GuardrailBlockedError):
        await agent.arun("go", trace=False)
    assert ran["tool"] is False  # tool never executed


async def test_tool_result_guardrail_non_blocking_logs_event(tmp_path: Path) -> None:
    """A non-blocking guardrail at tool_result warns (and logs an event)."""
    db_file = tmp_path / "local.db"
    init_local_db(db_file).close()

    def leaky_tool() -> str:
        return "here is a leak of data"

    tool = FunctionTool(name="leaky", fn=leaky_tool, description="leaks")

    def flag_leak(text: str) -> GuardrailResult:
        if "leak" in text:
            return GuardrailResult(passed=False, message="leak detected")
        return GuardrailResult(passed=True)

    guard = Guardrail(
        name="tr_guard",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.tool_result,
        blocking=False,
        fn=flag_leak,
    )

    agent = Agent(
        name="tr-agent",
        llm=_tool_calling_llm("leaky", {}),
        tools=[tool],
        guardrails=[guard],
    )

    cfg = get_config()
    prev_enabled, prev_path = cfg.ui_enabled, cfg.local_db_path
    cfg.ui_enabled = True
    cfg.local_db_path = str(db_file)
    try:
        result = await agent.arun("go", trace=False)
        assert result.output == "all done"  # completed despite the warning
    finally:
        cfg.ui_enabled, cfg.local_db_path = prev_enabled, prev_path

    db = init_local_db(db_file)
    try:
        rows = db.fetchall(
            "SELECT * FROM guardrail_events WHERE guardrail_name = ?", ("tr_guard",)
        )
        assert len(rows) >= 1
        assert rows[0]["position"] == "tool_result"
        assert rows[0]["outcome"] == "warned"
    finally:
        db.close()
