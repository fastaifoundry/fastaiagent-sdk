"""A Supervisor's result lists what its workers ran (1.99.0).

``result.tool_calls`` held only the lead's ``delegate_to_*`` calls: a trajectory
scorer over ``supervisor.run`` never saw ``lookup_order``. A Swarm's result has
always listed every hop's calls tagged with the agent. Now each worker's calls
follow the ``delegate_to_<role>`` call that ran them, tagged ``agent`` and
``worker``, and the lead's own calls are tagged with the lead.

No mocks: the offline ``FunctionModel`` / ``TestModel`` from ``fastaiagent.testing``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.tool import FunctionTool


def _uses(tool: str, answer: str) -> FunctionModel:
    def route(messages: list[Any]) -> Any:
        if messages[-1].role.value == "tool":
            return answer
        return "", [{"name": tool, "arguments": {"q": "x"}}]

    return FunctionModel(route)


def _lead() -> FunctionModel:
    plan = iter(["delegate_to_orders", "delegate_to_policy"])

    def route(messages: list[Any]) -> Any:
        nxt = next(plan, None)
        if nxt is None:
            return "team answer"
        return "", [{"name": nxt, "arguments": {"task": "go"}}]

    return FunctionModel(route)


def _team(**kw: Any) -> fa.Supervisor:
    orders = fa.Agent(
        name="orders-agent",
        llm=_uses("lookup_order", "order A100"),
        tools=[FunctionTool(name="lookup_order", fn=lambda q: "A100: annual plan")],
    )
    policy = fa.Agent(
        name="policy-agent",
        llm=_uses("search_policy", "fee $17.40"),
        tools=[FunctionTool(name="search_policy", fn=lambda q: "fee $17.40")],
    )
    return fa.Supervisor(
        name="calls-lead",
        llm=_lead(),
        workers=[fa.Worker(orders, role="orders"), fa.Worker(policy, role="policy")],
        **kw,
    )


@pytest.mark.parametrize("path", ["run", "arun"])
def test_worker_calls_follow_the_delegation_that_ran_them(path: str) -> None:
    team = _team()
    result = team.run("refund?") if path == "run" else asyncio.run(team.arun("refund?"))

    assert [(c["tool_name"], c["agent"], c.get("worker")) for c in result.tool_calls] == [
        ("delegate_to_orders", "calls-lead", None),
        ("lookup_order", "orders-agent", "orders"),
        ("delegate_to_policy", "calls-lead", None),
        ("search_policy", "policy-agent", "policy"),
    ]


def test_a_trajectory_scorer_sees_the_workers_tools() -> None:
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.trajectory import ToolUsageAccuracy

    results = evaluate(
        _team().run,
        [{"input": "refund?", "expected_trajectory": ["lookup_order", "search_policy"]}],
        scorers=[ToolUsageAccuracy()],
        persist=False,
    )
    assert results.cases[0].per_scorer["tool_usage_accuracy"]["score"] == 1.0


def test_every_validation_attempt_is_listed() -> None:
    reviews = iter(['{"approved": false, "feedback": "again"}', '{"approved": true}'])

    def lead(messages: list[Any]) -> Any:
        if messages[0].content.startswith("You are an output reviewer"):
            return next(reviews)
        if messages[-1].role.value == "tool":
            return "done"
        return "", [{"name": "delegate_to_orders", "arguments": {"task": "go"}}]

    orders = fa.Agent(
        name="orders-agent",
        llm=_uses("lookup_order", "order A100"),
        tools=[FunctionTool(name="lookup_order", fn=lambda q: "A100")],
    )
    team = fa.Supervisor(
        name="calls-lead",
        llm=FunctionModel(lead),
        workers=[fa.Worker(orders, role="orders")],
        validate_outputs=True,
    )
    names = [c["tool_name"] for c in team.run("go").tool_calls]
    assert names == ["delegate_to_orders", "lookup_order", "lookup_order"]


def test_a_routed_result_tags_the_workers_calls() -> None:
    route = {
        "answers": [
            {
                "type": "choice",
                "name": "worker",
                "choice": "orders",
                "probabilities": [
                    {"value": "orders", "probability": 0.9},
                    {"value": "other", "probability": 0.1},
                ],
                "confidence": 0.9,
            }
        ]
    }
    orders = fa.Agent(
        name="orders-agent",
        llm=_uses("lookup_order", "order A100"),
        tools=[FunctionTool(name="lookup_order", fn=lambda q: "A100")],
    )
    team = fa.Supervisor(
        name="desk",
        llm=TestModel("unused"),
        workers=[
            fa.Worker(orders, role="orders"),
            fa.Worker(fa.Agent(name="o", llm=TestModel("x")), role="other"),
        ],
        routing="decisions",
        router_llm=TestModel(decisions=route),
        fallback_worker="other",
    )
    calls = team.run("where is A100?").tool_calls
    assert [(c["tool_name"], c["agent"], c["worker"]) for c in calls] == [
        ("lookup_order", "orders-agent", "orders")
    ]
