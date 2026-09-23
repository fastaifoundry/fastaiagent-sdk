"""A policy-gated tool call is resolved by the calling application (1.74.0).

The plane decided on 2026-09-23 that it never approves a runtime tool call: it
distributes the policy, records each pause and its resolution, and flags one that
outlives its timeout. The application that started the run is the approver. These
tests pin what that means inside the SDK:

* a rejected approval **never runs the tool** — through ``Agent.aresume``, the
  deprecated blocking wait (rejected, or expired at the poll ceiling), a Chain,
  a Swarm and a Supervisor. Until 1.74.0 every agent-level path ran it (audit H1),
  and a governed agent inside a Swarm or Supervisor was never gated at all;
* ``arun()`` returns the pause by default, and the pause carries the arguments
  the app needs to ask its user;
* the HITL ledger records a policy pause as ``kind="approval"`` with the resolver,
  while an ``interrupt()`` in user code stays ``kind="interrupt"``.

Real ``Agent`` / ``Chain`` / ``Swarm`` / ``Supervisor``, real
``SQLiteCheckpointer``, and a real HTTP server standing in for the plane's
governance endpoints (``tests/_governance_plane.py``). The model is scripted: what
it *says* is irrelevant here. What matters is whether the tool ran and what the
model was told. The same paths against a real ``gpt-4o-mini`` live in
``tests/e2e/test_governance_e2e.py``.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, governance, interrupt
from fastaiagent.chain.interrupt import Resume
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from tests._governance_plane import AGENT_ID, GovPlane, reset_connection, serve

TRANSFER = ToolCall(id="call_1", name="transfer_funds", arguments={"amount": 500, "to": "Bob"})
REFUSAL = governance.denied("transfer_funds")


class _Script(LLMClient):
    """Calls ``first`` until the conversation holds a tool result, then answers.

    Keyed on the conversation, not a call counter, so a resume that re-issues a
    call gets the same answer. Keeps every conversation it was handed.
    """

    def __init__(self, first: ToolCall) -> None:
        super().__init__(provider="mock", model="mock")
        self._first = first
        self.seen: list[list[Any]] = []

    async def acomplete(self, messages: Any, tools: Any = None, **_: Any) -> LLMResponse:
        self.seen.append(list(messages))
        results = [m for m in messages if m.role.value == "tool"]
        if not results:
            return LLMResponse(content=None, tool_calls=[self._first], finish_reason="tool_calls")
        return LLMResponse(content=f"Tool said: {results[-1].content}", finish_reason="stop")

    def told(self) -> list[str]:
        """Every tool result in the last conversation the model saw."""
        return [m.content for m in self.seen[-1] if m.role.value == "tool"]


def _bank() -> tuple[list[dict[str, Any]], FunctionTool]:
    """A ``transfer_funds`` tool that records every time it actually runs."""
    ran: list[dict[str, Any]] = []

    def transfer_funds(amount: int, to: str) -> str:
        ran.append({"amount": amount, "to": to})
        return f"Transferred ${amount} to {to}."

    return ran, FunctionTool(name="transfer_funds", fn=transfer_funds)


def _banker(llm: LLMClient, tool: FunctionTool, ckpt: Path | None) -> Agent:
    return Agent(
        name="banker",
        agent_id=AGENT_ID,
        system_prompt="Move money with transfer_funds.",
        llm=llm,
        tools=[tool],
        checkpointer=SQLiteCheckpointer(str(ckpt)) if ckpt else None,
    )


@pytest.fixture
def plane(isolated_local_db: Path) -> Iterator[GovPlane]:
    import fastaiagent

    with serve() as (state, url):
        fastaiagent.connect(api_key="fa_k_gov_test", target=url)
        try:
            yield state
        finally:
            reset_connection()


def _ledger(plane: GovPlane, run_id: str) -> list[tuple[Any, ...]]:
    from fastaiagent.trace.hitl_export import get_hitl_exporter

    get_hitl_exporter().export([])  # synchronous drain; the emit's own drain may race it
    return [(e["event_type"], e["kind"], e["status"], e["resolver"]) for e in plane.ledger(run_id)]


# --- the pause goes to the app, with what it needs to ask ---------------------


def test_arun_returns_the_pause_with_the_tool_and_its_arguments(
    plane: GovPlane, tmp_path: Path
) -> None:
    ran, tool = _bank()
    agent = _banker(_Script(TRANSFER), tool, tmp_path / "ckpt.db")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-1"))

    assert res.status == "paused", res
    assert res.pending_interrupt is not None
    assert res.pending_interrupt["reason"] == governance.APPROVAL_REASON
    context = res.pending_interrupt["context"]
    assert context["tool"] == "transfer_funds"
    assert context["tool_input"] == {"amount": 500, "to": "Bob"}
    assert ran == []
    # Nothing waited on the plane, and the default is not the deprecated mode.
    assert plane.pending_polls == 0
    assert not [w for w in caught if "wait_for_approval" in str(w.message)]
    # The plane still gets the pause, linked to its approval request.
    assert plane.pending_posts[0]["body"]["kind"] == "approval"


# --- the fix: no means no -----------------------------------------------------


def test_a_rejected_approval_does_not_run_the_tool(plane: GovPlane, tmp_path: Path) -> None:
    ran, tool = _bank()
    llm = _Script(TRANSFER)
    agent = _banker(llm, tool, tmp_path / "ckpt.db")
    asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-1"))

    final = asyncio.run(agent.aresume("run-1", resume_value=Resume(approved=False)))

    assert final.status == "completed", final
    assert ran == [], "a rejected approval executed the tool"
    assert llm.told() == [REFUSAL]


def test_an_approved_approval_runs_the_tool_once(plane: GovPlane, tmp_path: Path) -> None:
    ran, tool = _bank()
    llm = _Script(TRANSFER)
    agent = _banker(llm, tool, tmp_path / "ckpt.db")
    asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-1"))

    final = asyncio.run(agent.aresume("run-1", resume_value=Resume(approved=True)))

    assert final.status == "completed", final
    assert ran == [{"amount": 500, "to": "Bob"}]
    assert llm.told() == ["Transferred $500 to Bob."]


def test_the_deprecated_blocking_wait_still_honours_a_console_approval(
    plane: GovPlane, tmp_path: Path
) -> None:
    """The plane keeps the console decision working for its transition window."""
    plane.resolve_as = "approved"  # the deprecated console approve
    ran, tool = _bank()
    agent = _banker(_Script(TRANSFER), tool, tmp_path / "ckpt.db")

    with pytest.warns(DeprecationWarning, match="wait_for_approval"):
        final = asyncio.run(
            agent.arun("Transfer $500 to Bob.", execution_id="run-1", wait_for_approval=True)
        )

    assert final.status == "completed", final
    assert ran == [{"amount": 500, "to": "Bob"}]


def test_the_deprecated_blocking_wait_refuses_a_rejection(plane: GovPlane, tmp_path: Path) -> None:
    plane.resolve_as = "rejected"  # the deprecated console deny
    ran, tool = _bank()
    llm = _Script(TRANSFER)
    agent = _banker(llm, tool, tmp_path / "ckpt.db")

    with pytest.warns(DeprecationWarning, match="wait_for_approval"):
        final = asyncio.run(
            agent.arun("Transfer $500 to Bob.", execution_id="run-1", wait_for_approval=True)
        )

    assert final.status == "completed", final
    assert ran == []
    assert llm.told() == [REFUSAL]


def test_the_deprecated_blocking_wait_refuses_at_the_ceiling(
    plane: GovPlane, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nobody decides — the plane never does now — so the wait expires. Expiry is a no."""
    monkeypatch.setattr(governance, "_POLL_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(governance, "_POLL_INTERVAL_SECONDS", 0.05)
    ran, tool = _bank()
    llm = _Script(TRANSFER)
    agent = _banker(llm, tool, tmp_path / "ckpt.db")

    with pytest.warns(DeprecationWarning, match="wait_for_approval"):
        final = asyncio.run(
            agent.arun("Transfer $500 to Bob.", execution_id="run-1", wait_for_approval=True)
        )

    assert plane.pending_polls >= 2, "the wait should have polled until the ceiling"
    assert final.status == "completed", final
    assert ran == []
    assert llm.told() == [REFUSAL]


def test_the_refusal_is_scoped_to_policy_pauses(plane: GovPlane, tmp_path: Path) -> None:
    """An ``interrupt()`` in user code is the tool's own business: on a "no" the
    tool is re-entered and decides for itself, as before 1.74.0."""
    answers: list[bool] = []

    def ask_human(question: str) -> str:
        decision = interrupt(reason="need_sign_off", context={"question": question})
        answers.append(decision.approved)
        return "signed" if decision.approved else "declined"

    llm = _Script(ToolCall(id="call_1", name="ask_human", arguments={"question": "ok?"}))
    agent = Agent(
        name="clerk",
        agent_id=AGENT_ID,
        system_prompt="Ask a human.",
        llm=llm,
        tools=[FunctionTool(name="ask_human", fn=ask_human)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    paused = asyncio.run(agent.arun("Get sign-off.", execution_id="run-u"))
    assert paused.status == "paused" and paused.pending_interrupt is not None
    assert paused.pending_interrupt["reason"] == "need_sign_off"

    final = asyncio.run(agent.aresume("run-u", resume_value=Resume(approved=False)))

    assert final.status == "completed", final
    assert answers == [False]
    assert llm.told() == ["declined"]
    assert _ledger(plane, "run-u") == [
        ("paused", "interrupt", None, None),
        ("resolved", "interrupt", "rejected", None),
    ]


# --- the evidence: kind and resolver ------------------------------------------


def test_the_ledger_records_an_approval_and_who_resolved_it(
    plane: GovPlane, tmp_path: Path
) -> None:
    ran, tool = _bank()
    agent = _banker(_Script(TRANSFER), tool, tmp_path / "ckpt.db")
    asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-1"))
    asyncio.run(
        agent.aresume(
            "run-1",
            resume_value=Resume(approved=False, metadata={"resolver": "carol@bank.example"}),
        )
    )

    assert _ledger(plane, "run-1") == [
        ("paused", "approval", None, None),
        ("resolved", "approval", "rejected", "carol@bank.example"),
    ]
    # No outcome the plane has not shipped yet (``expired``) ever leaves.
    assert {e["status"] for e in plane.ledger("run-1")} <= {None, "approved", "rejected"}


# --- the same guarantee one level up ------------------------------------------


def test_a_chain_refuses_a_rejected_approval(plane: GovPlane, tmp_path: Path) -> None:
    """An agent with no checkpointer of its own inside a Chain: the chain owns the
    pause, and on resume the gate's own resume branch refuses."""
    from fastaiagent import Chain

    ran, tool = _bank()
    llm = _Script(TRANSFER)
    chain = Chain("desk", checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")))
    chain.add_node("bank", agent=_banker(llm, tool, None))

    paused = asyncio.run(chain.aexecute({"input": "Transfer $500 to Bob."}, execution_id="run-c"))
    assert paused.status == "paused", paused
    assert ran == []

    final = asyncio.run(chain.aresume("run-c", resume_value=Resume(approved=False)))

    assert final.status == "completed", final
    assert ran == []
    assert llm.told() == [REFUSAL]
    assert [row[:2] for row in _ledger(plane, "run-c")] == [
        ("paused", "approval"),
        ("resolved", "approval"),
    ]


def test_a_chain_does_not_report_a_self_checkpointed_agents_pause_as_done(
    plane: GovPlane, tmp_path: Path
) -> None:
    """An agent with its OWN checkpointer returns the pause instead of raising.
    Before 1.74.0 the chain took its empty output as the node's result and
    reported ``completed`` — past a decision nobody had made. The pause belongs
    to the agent, so the chain says so rather than pretending the step ran."""
    from fastaiagent import Chain
    from fastaiagent._internal.errors import ChainError

    ran, tool = _bank()
    agent = _banker(_Script(TRANSFER), tool, tmp_path / "agent.db")
    chain = Chain("desk", checkpointer=SQLiteCheckpointer(str(tmp_path / "chain.db")))
    chain.add_node("bank", agent=agent)

    with pytest.raises(ChainError, match="paused .*policy_approval_required.* own checkpointer"):
        asyncio.run(chain.aexecute({"input": "Transfer $500 to Bob."}, execution_id="run-c"))
    assert ran == []


def test_a_parallel_node_does_not_count_a_paused_child_as_a_success(
    plane: GovPlane, tmp_path: Path
) -> None:
    from fastaiagent import Chain
    from fastaiagent.chain.node import NodeType

    ran, tool = _bank()
    chain = Chain("desk", checkpoint_enabled=False)
    chain.add_node(
        "fan",
        type=NodeType.parallel,
        agents=[_banker(_Script(TRANSFER), tool, tmp_path / "agent.db")],
    )

    result = asyncio.run(chain.aexecute({"input": "Transfer $500 to Bob."}))

    [outcome] = result.node_results["fan"]["outputs"]
    assert "output" not in outcome, outcome
    assert "paused" in outcome["error"] and "policy_approval_required" in outcome["error"]
    assert ran == []


def test_a_swarm_gates_and_refuses_its_agents_tool(plane: GovPlane, tmp_path: Path) -> None:
    from fastaiagent.agent.swarm import Swarm

    ran, tool = _bank()
    llm = _Script(TRANSFER)
    swarm = Swarm(
        name="desk",
        agents=[_banker(llm, tool, None)],
        entrypoint="banker",
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    paused = asyncio.run(swarm.arun("Transfer $500 to Bob.", execution_id="run-s"))
    assert paused.status == "paused", f"the policy never gated the swarm: {paused}"
    assert ran == []

    final = asyncio.run(swarm.aresume("run-s", resume_value=Resume(approved=False)))

    assert final.status == "completed", final
    assert ran == []
    assert llm.told() == [REFUSAL]


def test_a_supervisor_gates_and_refuses_its_workers_tool(plane: GovPlane, tmp_path: Path) -> None:
    from fastaiagent.agent.team import Supervisor, Worker

    ran, tool = _bank()
    worker_llm = _Script(TRANSFER)
    supervisor = Supervisor(
        name="desk",
        llm=_Script(
            ToolCall(id="sup_1", name="delegate_to_banker", arguments={"task": "Pay Bob $500."})
        ),
        workers=[Worker(agent=_banker(worker_llm, tool, None), role="banker", description="Pays")],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    paused = asyncio.run(supervisor.arun("Transfer $500 to Bob.", execution_id="run-v"))
    assert paused.status == "paused", f"the policy never gated the worker: {paused}"
    assert ran == []

    final = asyncio.run(supervisor.aresume("run-v", resume_value=Resume(approved=False)))

    assert final.status == "completed", final
    assert ran == []
    assert worker_llm.told() == [REFUSAL]
