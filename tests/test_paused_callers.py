"""Callers of ``Agent.arun()`` never take a paused run as its answer (1.78.0).

Since 1.74.0 ``arun()`` returns a pause (``status="paused"``, ``output=""``) when a
managed approval policy or an ``interrupt()`` stops a tool call, and an agent with
no checkpointer raises the bare ``InterruptSignal`` instead. Several callers inside
the SDK still read ``.output``:

* ``simulate()`` recorded ``""`` as the agent's turn, let the simulated user answer
  the silence, and judged the conversation that never happened — and one agent
  with no checkpointer crashed the whole run, every scenario with it;
* ``evaluate()`` and the pytest eval plugin scored ``""`` as the case's answer;
* replay let the bare ``InterruptSignal`` escape ``arerun()``.

Each now reports the pause the way it already reports "did not finish": an errored
scenario, an errored eval case, a ``ReplayError``. None of them approves or rejects
on anyone's behalf. The MCP server's half is in ``tests/test_mcp_server_pauses.py``.

Real ``Agent`` / ``SQLiteCheckpointer`` / SQLite ``local.db``; the agent, the
simulated user and the judge are driven by ``FunctionModel`` / ``TestModel`` (real
``LLMClient`` subclasses), so nothing touches the network.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, interrupt
from fastaiagent._internal.errors import ReplayError
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.eval import evaluate
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.simulate import Scenario, SimulatedUser, simulate
from fastaiagent.testing.models import FunctionModel, TestModel
from fastaiagent.trace.replay import Replay
from fastaiagent.trace.storage import SpanData, TraceData
from fastaiagent.ui.db import init_local_db

pytest_plugins = ["pytester"]


def _asks_manager(amount: int) -> str:
    interrupt("manager_approval", {"amount": amount})
    return f"refunded {amount}"


def _route(messages: list[Any]) -> Any:
    """Refund requests call the tool; anything else is answered directly."""
    last = messages[-1]
    if last.role.value == "tool":
        return f"done: {last.content}"
    if "refund" in str(last.content):
        return "", [{"name": "refund", "arguments": {"amount": 40}}]
    return "hi"


def _agent(ckpt: Path | None) -> Agent:
    return Agent(
        name="support",
        llm=FunctionModel(_route),
        tools=[FunctionTool(name="refund", fn=_asks_manager)],
        checkpointer=SQLiteCheckpointer(str(ckpt)) if ckpt else None,
    )


def _passing_judge() -> LLMJudge:
    return LLMJudge(llm=TestModel(response=json.dumps({"score": 1.0, "reasoning": "ok"})))


# --- simulate() -------------------------------------------------------------------


def test_a_paused_turn_ends_the_scenario_unjudged(tmp_path: Path) -> None:
    scenario = Scenario(
        name="refund",
        user=SimulatedUser(script=["please refund my order", "thanks"]),
        success_criteria=["The agent refunded the order."],
    )
    results = simulate(
        scenario, _agent(tmp_path / "ckpt.db"), judge=_passing_judge(), persist=False
    )
    r = results.results[0]

    # Before 1.78.0: an assistant turn of "" followed by "thanks", judged a pass.
    assert [t.role for t in r.transcript] == ["user"]
    assert r.passed is False
    assert r.verdicts == []
    assert r.error is not None and "manager_approval" in r.error
    assert "execution_id=" in r.error
    assert "errored" in results.summary()


def test_a_pause_without_a_checkpointer_errors_only_its_scenario() -> None:
    scenarios = [
        Scenario(name="greet", user=SimulatedUser(script=["hello"]), success_criteria=["ok"]),
        Scenario(name="refund", user=SimulatedUser(script=["refund it"]), success_criteria=["ok"]),
    ]
    # Before 1.78.0 the InterruptSignal escaped gather() and simulate() raised.
    results = simulate(scenarios, _agent(None), judge=_passing_judge(), persist=False)
    by_name = {r.scenario_name: r for r in results.results}

    assert by_name["greet"].passed is True and by_name["greet"].error is None
    refund = by_name["refund"]
    assert refund.passed is False
    assert refund.error is not None and "checkpointer" in refund.error


def test_an_adapter_returning_a_pause_is_treated_the_same(tmp_path: Path) -> None:
    agent = _agent(tmp_path / "ckpt.db")

    async def adapter(messages: list[Any]) -> Any:
        return await agent.arun(messages[-1].content)

    scenario = Scenario(
        name="refund", user=SimulatedUser(script=["refund it"]), success_criteria=["ok"]
    )
    r = simulate(scenario, adapter, judge=_passing_judge(), persist=False).results[0]
    assert r.passed is False
    assert r.error is not None and "manager_approval" in r.error


def test_the_pause_is_persisted_with_the_scenario(tmp_path: Path) -> None:
    scenario = Scenario(
        name="refund", user=SimulatedUser(script=["refund it"]), success_criteria=["ok"]
    )
    results = simulate(
        scenario, _agent(tmp_path / "ckpt.db"), judge=_passing_judge(), persist=False
    )
    run_id = results.persist_local(db_path=tmp_path / "local.db")

    db = init_local_db(tmp_path / "local.db")
    try:
        row = db.fetchone("SELECT passed, error FROM sim_cases WHERE run_id = ?", (run_id,))
    finally:
        db.close()
    assert row is not None
    assert row["passed"] == 0
    assert "manager_approval" in row["error"]


# --- evaluate() ---------------------------------------------------------------------


def test_a_paused_eval_case_is_errored_not_scored(tmp_path: Path) -> None:
    agent = _agent(tmp_path / "ckpt.db")
    results = evaluate(
        agent.arun,
        [
            {"input": "hello", "expected": "hi"},
            {"input": "refund 40", "expected": "done: refunded 40"},
        ],
        scorers=["exact_match"],
        persist=False,
        concurrency=1,
    )

    # Before 1.78.0 the paused case was scored on "" — a wrong answer, not a pause.
    assert results.errored_count == 1
    paused = next(c for c in results.cases if c.error)
    assert paused.input == "refund 40"
    assert paused.actual_output is None and paused.per_scorer == {}
    assert "manager_approval" in paused.error
    scored = next(c for c in results.cases if not c.error)
    assert scored.per_scorer["exact_match"]["passed"] is True


_PAUSED_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent import Agent, FunctionTool, interrupt
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.eval import case
from fastaiagent.testing.models import FunctionModel


def _asks_manager(amount: int) -> str:
    interrupt("manager_approval", {{"amount": amount}})
    return "refunded"


def _route(messages):
    if messages[-1].role.value == "tool":
        return "done"
    return "", [{{"name": "refund", "arguments": {{"amount": 40}}}}]


@case(input="refund 40", expected="done")
def test_refund(evaluate_one):
    agent = Agent(
        name="support",
        llm=FunctionModel(_route),
        tools=[FunctionTool(name="refund", fn=_asks_manager)],
        checkpointer=SQLiteCheckpointer(r"{ckpt}"),
    )
    evaluate_one(agent.run, scorers=["exact_match"])
"""


def test_a_paused_pytest_eval_case_fails_as_errored(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    import os

    from fastaiagent._internal.config import reset_config

    db = tmp_path / "local.db"
    pytester.makepyfile(_PAUSED_SUITE.format(db=db, ckpt=tmp_path / "ckpt.db"))
    try:
        result = pytester.runpytest("--no-header")
    finally:
        os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
        reset_config()

    # Before 1.78.0 the case was scored on "" and failed as a wrong answer.
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*errored (infra, not scored)*manager_approval*"])
    with sqlite3.connect(db) as conn:
        errors = [r[0] for r in conn.execute("SELECT error FROM eval_cases")]
    assert errors and errors[0] and "manager_approval" in errors[0]


# --- replay ---------------------------------------------------------------------------


def _refund_trace() -> TraceData:
    """A captured run that called ``refund`` once, then answered."""
    root = SpanData(
        span_id="root",
        trace_id="trace_paused_replay",
        name="agent.support",
        start_time="2026-09-24T00:00:00Z",
        end_time="2026-09-24T00:00:02Z",
        attributes={
            "agent.name": "support",
            "agent.input": "refund 40",
            "agent.output": "done",
            "agent.system_prompt": "",
            "agent.config": json.dumps({"max_iterations": 3}),
            "agent.tools": json.dumps([]),
            "agent.guardrails": json.dumps([]),
            "agent.llm.provider": "openai",
            "agent.llm.model": "gpt-4o-mini",
            "agent.llm.config": json.dumps(
                {"provider": "openai", "model": "gpt-4o-mini", "api_key": "not-used"}
            ),
        },
    )

    def llm(i: int, content: str, tool_calls: list[dict[str, Any]] | None) -> SpanData:
        attrs: dict[str, Any] = {
            "gen_ai.system": "openai",
            "gen_ai.response.content": content,
            "gen_ai.response.finish_reason": "tool_calls" if tool_calls else "stop",
        }
        if tool_calls:
            attrs["gen_ai.response.tool_calls"] = json.dumps(tool_calls)
        return SpanData(
            span_id=f"llm{i}",
            trace_id="trace_paused_replay",
            parent_span_id="root",
            name="llm.openai.gpt-4o-mini",
            start_time=f"2026-09-24T00:00:0{i}Z",
            end_time=f"2026-09-24T00:00:0{i}Z",
            attributes=attrs,
        )

    spans = [
        root,
        llm(0, "", [{"id": "c1", "name": "refund", "arguments": {"amount": 40}}]),
        llm(1, "done", None),
    ]
    return TraceData(
        trace_id="trace_paused_replay",
        name="support",
        start_time=root.start_time,
        end_time=root.end_time,
        spans=spans,
    )


@pytest.mark.asyncio
async def test_a_replay_that_pauses_raises_replay_error() -> None:
    forked = (
        Replay(_refund_trace())
        .fork_at(step=0)
        .with_determinism("recorded")
        .with_tool_override("refund", FunctionTool(name="refund", fn=_asks_manager))
    )
    # Before 1.78.0 the bare InterruptSignal escaped arerun().
    with pytest.raises(ReplayError, match="manager_approval") as exc:
        await forked.arerun()
    assert "with_tool_override" in str(exc.value)
