"""``@idempotent`` inside real runs: a side effect before an ``interrupt()`` must
not fire again when the paused run resumes.

Before 1.92.0 an ``async def`` wrapped in ``@idempotent`` never ran at all: the
sync wrapper cached the un-awaited coroutine, raised, the model was told the tool
failed, and the run still finished normally. Swept across the places that bind the
cache: ``Agent.arun``, ``Agent.astream`` and ``Chain.aexecute``, each resumed.

Real ``Agent`` / ``Chain`` / ``SQLiteCheckpointer``; the model is ``FunctionModel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, Chain, FunctionTool, Resume, idempotent, interrupt
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.llm.stream import Paused
from fastaiagent.testing.models import FunctionModel


def _make_pay(charges: list[int]):
    @idempotent
    async def charge(amount: int) -> dict:
        charges.append(amount)
        return {"charge_id": f"ch_{len(charges)}"}

    async def pay(amount: int) -> str:
        receipt = await charge(amount)
        interrupt("confirm_receipt", {"charge_id": receipt["charge_id"]})
        return f"paid {receipt['charge_id']}"

    return pay


def _route(messages: list[Any]) -> Any:
    last = messages[-1]
    if last.role.value == "tool":
        return f"done: {last.content}"
    return "", [{"name": "pay", "arguments": {"amount": 40}}]


def _agent(tmp_path: Path, charges: list[int]) -> Agent:
    return Agent(
        name="payer",
        llm=FunctionModel(_route),
        tools=[FunctionTool(name="pay", fn=_make_pay(charges))],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "cp.db")),
    )


@pytest.mark.asyncio
async def test_agent_arun_then_aresume_charges_once(tmp_path: Path) -> None:
    charges: list[int] = []
    agent = _agent(tmp_path, charges)

    paused = await agent.arun("pay 40", execution_id="pay-arun")
    assert paused.status == "paused"
    assert charges == [40]

    done = await agent.aresume("pay-arun", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert "paid ch_1" in done.output
    assert charges == [40]


@pytest.mark.asyncio
async def test_agent_astream_then_aresume_charges_once(tmp_path: Path) -> None:
    charges: list[int] = []
    agent = _agent(tmp_path, charges)

    events = [e async for e in agent.astream("pay 40", execution_id="pay-stream")]
    assert any(isinstance(e, Paused) for e in events)
    assert charges == [40]

    done = await agent.aresume("pay-stream", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert charges == [40]


@pytest.mark.asyncio
async def test_chain_aexecute_then_aresume_charges_once(tmp_path: Path) -> None:
    charges: list[int] = []
    chain = Chain(
        "pay-chain",
        checkpoint_enabled=True,
        checkpointer=SQLiteCheckpointer(str(tmp_path / "cp.db")),
    )
    chain.add_node(
        "pay",
        type="tool",
        tool=FunctionTool(name="pay", fn=_make_pay(charges)),
        input_mapping={"amount": "{{input.amount}}"},
    )

    paused = await chain.aexecute({"amount": 40}, execution_id="pay-chain-1")
    assert paused.status == "paused"
    assert charges == [40]

    done = await chain.aresume("pay-chain-1", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert charges == [40]
