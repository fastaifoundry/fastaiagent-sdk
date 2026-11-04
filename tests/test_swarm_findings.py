"""A Swarm handoff tells the next agent what the team already found (1.99.0).

The briefing carried the handoff reason, the shared blackboard and the original
request — and nothing the handing-off agent's tools had returned unless its model
copied it into ``reason`` or ``context``. Live: tech found the router fix, billing
never saw it, judged the tech issue open and handed back, and tech searched again;
in a re-run the ping-pong hit ``max_handoffs``.

No mocks: the offline ``FunctionModel`` from ``fastaiagent.testing``; the briefing
is read off the messages the next agent's model actually received.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.testing import FunctionModel
from fastaiagent.tool import FunctionTool

FIX = "Hold reset for 15 seconds, then re-pair. Escalation code TCH-77."


def _swarm(seen: list[str], *, checkpointer: Any = None, result: str = FIX) -> fa.Swarm:
    def tech(messages: list[Any]) -> Any:
        if messages[-1].role.value == "tool":
            return "", [{"name": "handoff_to_billing", "arguments": {"reason": "billing next"}}]
        return "", [{"name": "search_tech_kb", "arguments": {"query": "E-311"}}]

    def billing(messages: list[Any]) -> Any:
        seen.append(str(messages[-1].content))
        return "Fee is $17.40."

    return fa.Swarm(
        name="findings",
        agents=[
            fa.Agent(
                name="tech",
                llm=FunctionModel(tech),
                tools=[FunctionTool(name="search_tech_kb", fn=lambda query: result)],
            ),
            fa.Agent(name="billing", llm=FunctionModel(billing)),
        ],
        entrypoint="tech",
        checkpointer=checkpointer,
    )


@pytest.mark.parametrize("path", ["run", "astream"])
def test_the_next_agent_is_told_what_was_found(path: str) -> None:
    seen: list[str] = []
    swarm = _swarm(seen)
    if path == "run":
        swarm.run("My router shows E-311 and I want to cancel.")
    else:

        async def consume() -> None:
            async for _ in swarm.astream("My router shows E-311 and I want to cancel."):
                pass

        asyncio.run(consume())

    assert len(seen) == 1
    assert f"tech ran search_tech_kb: {FIX}" in seen[0]
    assert "handoff_to_billing" not in seen[0].split("found so far")[1]


def test_a_long_result_is_cut_in_the_briefing_and_says_so() -> None:
    seen: list[str] = []
    _swarm(seen, result="x" * 9000).run("go")
    assert "[… 5000 more characters]" in seen[0]


def test_findings_survive_in_the_checkpoint(tmp_path: Any) -> None:
    """``resume`` restores them, so a resumed run's briefings still carry them."""
    from fastaiagent.agent.swarm import _restore_state

    store = SQLiteCheckpointer(str(tmp_path / "ckpt.db"))
    result = _swarm([], checkpointer=store).run("go")
    handoffs = [
        cp for cp in store.list(result.execution_id, limit=50) if cp.node_id.startswith("handoff:")
    ]
    restored = _restore_state(handoffs[-1].state_snapshot)
    assert restored.findings == [{"agent": "tech", "tool": "search_tech_kb", "result": FIX}]
