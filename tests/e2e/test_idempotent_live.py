"""Live: a real model drives a tool that charges (async ``@idempotent``) and then
pauses; the run is resumed, and the charge must have happened exactly once.

Covers the 1.92.0 fixes end to end with gpt-5-mini choosing the tool: an async
``@idempotent`` body ran 0 times before, and inside an Agent node of a checkpointed
Chain it ran again on every Chain resume. The offline matrix is
``tests/test_idempotent_runs.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fastaiagent import Agent, Chain, FunctionTool, LLMClient, Resume, idempotent, interrupt
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

pytestmark = pytest.mark.e2e

_PROMPT = "Refund order 1042 for 40 dollars using the refund tool."


@pytest.fixture(autouse=True)
def _needs_openai() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def _agent(charges: list[str], checkpointer: SQLiteCheckpointer | None) -> Agent:
    @idempotent
    async def charge(order: str, amount: int) -> dict:
        charges.append(order)
        return {"charge_id": f"ch_{len(charges)}"}

    async def refund(order: str, amount: int) -> str:
        """Refund an order. A manager confirms after the charge."""
        receipt = await charge(order, amount)
        interrupt("manager_confirm", {"charge_id": receipt["charge_id"]})
        return f"refunded, {receipt['charge_id']}"

    return Agent(
        name="refunds",
        system_prompt="Always use the refund tool for refunds. Reply in one short sentence.",
        llm=LLMClient(provider="openai", model="gpt-5-mini", reasoning="minimal"),
        tools=[FunctionTool(name="refund", fn=refund)],
        checkpointer=checkpointer,
    )


async def test_an_agent_charges_once_across_resume(tmp_path: Path) -> None:
    charges: list[str] = []
    agent = _agent(charges, SQLiteCheckpointer(str(tmp_path / "cp.db")))

    paused = await agent.arun(_PROMPT, execution_id="live-agent", trace=False)
    assert paused.status == "paused"
    assert charges == ["1042"]

    done = await agent.aresume("live-agent", resume_value=Resume(approved=True), trace=False)
    assert done.status == "completed"
    assert charges == ["1042"]


async def test_an_agent_node_in_a_chain_charges_once_across_resume(tmp_path: Path) -> None:
    charges: list[str] = []
    chain = Chain(
        "live-refund-chain",
        checkpoint_enabled=True,
        checkpointer=SQLiteCheckpointer(str(tmp_path / "cp.db")),
    )
    chain.add_node("refund", type="agent", agent=_agent(charges, None))

    paused = await chain.aexecute({"input": _PROMPT}, execution_id="live-chain")
    assert paused.status == "paused"
    assert charges == ["1042"]

    done = await chain.aresume("live-chain", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert charges == ["1042"]
