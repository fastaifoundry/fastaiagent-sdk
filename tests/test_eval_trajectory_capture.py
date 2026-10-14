"""``evaluate()`` hands the agent's tool calls to the trajectory scorers (1.89.0).

Before, ``evaluate()`` passed no ``actual_trajectory`` / ``actual_tool_calls`` /
``actual_steps`` and no per-row expectation, so every trajectory scorer compared
an empty record with an empty expectation and **passed**: an agent that called
the wrong tool, or no tool, scored 1.0. Now:

* a row's ``expected_trajectory`` (etc.) is forwarded, and the agent's
  ``AgentResult.tool_calls`` become the actual record: a real verdict;
* a row without an expectation is **skipped** (does not apply);
* a callable that returns a bare string has no record: **errored**.

Swept across ``evaluate`` / ``aevaluate`` × ``agent.run`` / ``agent.arun``. Real
``Agent``, real ``@tool``, a ``FunctionModel`` that calls the tool the prompt
names; no mocks.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fastaiagent.agent import Agent
from fastaiagent.eval import evaluate
from fastaiagent.eval.compare import case_outcome
from fastaiagent.eval.evaluate import aevaluate
from fastaiagent.eval.trajectory import (
    PathCorrectness,
    StepEfficiency,
    ToolCallCorrectness,
    ToolUsageAccuracy,
)
from fastaiagent.llm.message import MessageRole
from fastaiagent.testing import FunctionModel
from fastaiagent.tool import tool


@tool()
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id}: shipped"


@tool()
def issue_refund(order_id: str) -> str:
    """Refund an order."""
    return f"refunded {order_id}"


def _responder(messages: list[Any]) -> Any:
    """Calls the tool the user names once, then answers."""
    if messages[-1].role == MessageRole.tool:
        return ("done", [])
    text = str(messages[-1].content)
    name = "issue_refund" if "refund" in text else "lookup_order"
    return ("", [{"name": name, "arguments": {"order_id": "A1"}}])


def _agent() -> Agent:
    return Agent(name="traj-bot", llm=FunctionModel(_responder), tools=[lookup_order, issue_refund])


DATASET = [
    # The agent looks the order up: matches.
    {"input": "where is A1", "expected_trajectory": ["lookup_order"]},
    # The agent refunds instead of looking up: a real fail (it used to pass).
    {"input": "refund A1", "expected_trajectory": ["lookup_order"]},
    # No expectation: does not apply.
    {"input": "where is A1"},
]


def _scorers() -> list[Any]:
    return [ToolUsageAccuracy(), PathCorrectness()]


def _run(mode: str) -> Any:
    agent = _agent()
    if mode == "evaluate/run":
        return evaluate(agent.run, DATASET, scorers=_scorers(), persist=False)
    if mode == "evaluate/arun":
        return evaluate(agent.arun, DATASET, scorers=_scorers(), persist=False)
    return asyncio.run(aevaluate(agent.arun, DATASET, scorers=_scorers(), persist=False))


@pytest.mark.parametrize("mode", ["evaluate/run", "evaluate/arun", "aevaluate/arun"])
def test_trajectory_is_scored_from_the_agents_tool_calls(mode: str) -> None:
    results = _run(mode)
    outcomes = [c.per_scorer["tool_usage_accuracy"] for c in results.cases]
    assert outcomes[0]["passed"] is True and outcomes[0]["score"] == 1.0
    assert outcomes[1]["passed"] is False, "the wrong tool scored as correct"
    assert outcomes[2].get("skipped") is True
    assert [case_outcome({"per_scorer": c.per_scorer}) for c in results.cases] == [
        "passed",
        "failed",
        "errored",  # nothing applied: unscored, never a pass
    ]
    details = outcomes[1].get("details") or {}
    assert details.get("actual") == ["issue_refund"], details


def test_a_bare_string_agent_has_no_record() -> None:
    results = evaluate(
        lambda q: "done",
        [{"input": "where is A1", "expected_trajectory": ["lookup_order"]}],
        scorers=_scorers(),
        persist=False,
    )
    v = results.cases[0].per_scorer["tool_usage_accuracy"]
    assert v.get("errored") is True and v["passed"] is False, v


def test_row_expectation_wins_over_a_run_wide_kwarg() -> None:
    results = evaluate(
        _agent().run,
        [{"input": "refund A1", "expected_trajectory": ["issue_refund"]}, {"input": "where"}],
        scorers=[ToolUsageAccuracy()],
        persist=False,
        expected_trajectory=["lookup_order"],
    )
    first, second = (c.per_scorer["tool_usage_accuracy"] for c in results.cases)
    assert first["passed"] is True, "the row's own expectation applies"
    assert second["passed"] is True, "the run-wide kwarg still applies to rows without one"


def test_tool_call_arguments_and_step_count_reach_their_scorers() -> None:
    results = evaluate(
        _agent().run,
        [
            {
                "input": "where is A1",
                "expected_tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "A1"}}],
                "expected_steps": 1,
            }
        ],
        scorers=[ToolCallCorrectness(), StepEfficiency()],
        persist=False,
    )
    per = results.cases[0].per_scorer
    assert per["tool_call_correctness"]["passed"] is True, per
    assert per["step_efficiency"]["score"] == 1.0, per
    assert per["step_efficiency"]["details"] == {"expected": 1, "actual": 1}, per

    # The control that makes the pass mean something: wrong arguments and a
    # wrong step count are real fails (both passed vacuously before 1.89.0).
    row = {
        "input": "where is A1",
        "expected_tool_calls": [{"name": "lookup_order", "arguments": {"order_id": "Z9"}}],
        "expected_steps": 3,
    }
    scorers = [ToolCallCorrectness(), StepEfficiency()]
    wrong = evaluate(_agent().run, [row], scorers=scorers, persist=False).cases[0].per_scorer
    assert wrong["tool_call_correctness"]["passed"] is False, wrong
    assert wrong["step_efficiency"]["details"] == {"expected": 3, "actual": 1}, wrong


def test_no_case_content_in_trajectory_details() -> None:
    """``details`` holds tool names only; arguments are case content."""
    results = evaluate(
        _agent().run,
        [{"input": "where is A1", "expected_trajectory": ["lookup_order"]}],
        scorers=[ToolUsageAccuracy()],
        persist=False,
    )
    details = results.cases[0].per_scorer["tool_usage_accuracy"].get("details")
    assert details == {"expected": ["lookup_order"], "actual": ["lookup_order"]}, details
    assert "A1" not in str(details)
