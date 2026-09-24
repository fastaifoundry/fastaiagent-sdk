"""A policy-gated tool call is resolved by the calling application (1.74.0).

The plane decided on 2026-09-23 that it never approves a runtime tool call: it
distributes the policy, records each pause and its resolution, and flags one that
outlives its timeout. The application that started the run is the approver. These
tests pin what that means inside the SDK:

* a rejected approval **never runs the tool** — through ``Agent.aresume``, a
  Chain, a Swarm and a Supervisor. Until 1.74.0 every agent-level path ran it
  (audit H1), and a governed agent inside a Swarm or Supervisor was never gated;
* ``arun()`` returns the pause, and the pause carries the arguments the app needs
  to ask its user. The blocking ``wait_for_approval=True`` is gone (1.76.0);
* the HITL ledger records a policy pause as ``kind="approval"`` with the resolver,
  while an ``interrupt()`` in user code stays ``kind="interrupt"``;
* a policy resolution names the pending run it resolves (``context.pending_id``,
  1.76.0) — ``null`` when registration failed — so the plane matches it exactly.

Real ``Agent`` / ``Chain`` / ``Swarm`` / ``Supervisor``, real
``SQLiteCheckpointer``, and a real HTTP server standing in for the plane's
governance endpoints (``tests/_governance_plane.py``). The model is scripted: what
it *says* is irrelevant here. What matters is whether the tool ran and what the
model was told. The same paths against a real ``gpt-4o-mini`` live in
``tests/e2e/test_governance_e2e.py``.
"""

from __future__ import annotations

import asyncio
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

    res = asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-1"))

    assert res.status == "paused", res
    assert res.pending_interrupt is not None
    assert res.pending_interrupt["reason"] == governance.APPROVAL_REASON
    context = res.pending_interrupt["context"]
    assert context["tool"] == "transfer_funds"
    assert context["tool_input"] == {"amount": 500, "to": "Bob"}
    assert context["pending_id"] == "pr-run-1"  # the id the plane returned for this pause
    assert ran == []
    # Nothing waits on the plane.
    assert plane.pending_polls == 0
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


def test_wait_for_approval_true_is_removed_and_runs_nothing(
    plane: GovPlane, tmp_path: Path
) -> None:
    """Removed in 1.76.0: the plane retired the console decision it waited for
    (enterprise PR #199). It fails before anything runs — no model call, no pause
    registered, no tool — rather than wait on nothing and refuse at the end."""
    ran, tool = _bank()
    llm = _Script(TRANSFER)
    agent = _banker(llm, tool, tmp_path / "ckpt.db")

    with pytest.raises(ValueError, match=r"wait_for_approval=True\) was removed .*aresume"):
        asyncio.run(
            agent.arun("Transfer $500 to Bob.", execution_id="run-1", wait_for_approval=True)
        )

    assert llm.seen == [] and plane.pending_posts == [] and ran == []
    # False is still accepted, and is the pause-returning default.
    res = asyncio.run(
        agent.arun("Transfer $500 to Bob.", execution_id="run-2", wait_for_approval=False)
    )
    assert res.status == "paused", res


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
    # The app's own reason travels as-is; only the policy reason makes a pause an approval.
    assert {e["reason"] for e in plane.ledger("run-u")} == {"need_sign_off"}
    # No pending run exists for an interrupt(), so its resolution names none.
    assert [e["context"] for e in plane.ledger("run-u")] == [None, None]


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
    # The plane matches a resolution to a pause by these two values (2026-09-23), and on SDKs
    # up to 1.73.0 by the reason alone. They are wire, so they are pinned as LITERALS here —
    # a test written against ``governance.APPROVAL_REASON`` would pass through a rename.
    assert {(e["kind"], e["reason"]) for e in plane.ledger("run-1")} == {
        ("approval", "policy_approval_required")
    }


# --- exact matching: the resolution names its pending run (1.76.0) ------------
#
# The plane's contract (enterprise PR #199, wire v1.1): a policy pause's
# ``resolved`` event carries ``context: {"pending_id": <id | null>}``. A value
# closes exactly that pause; ``null`` closes nothing; NO key means an older SDK
# and a match by position — the guess this replaces, so the key is never omitted
# for a pause this SDK registered.


def test_a_resolution_names_the_pending_run_it_resolves(plane: GovPlane, tmp_path: Path) -> None:
    ran, tool = _bank()
    agent = _banker(_Script(TRANSFER), tool, tmp_path / "ckpt.db")
    asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-1"))
    asyncio.run(agent.aresume("run-1", resume_value=Resume(approved=True)))

    _ledger(plane, "run-1")  # drain
    paused, resolved = plane.ledger("run-1")
    assert resolved["event_type"] == "resolved"
    # Exactly the id the plane handed back from POST /runs/run-1/pending, and nothing else:
    # the pause's own context also holds ``tool_input``, which must never ride this channel.
    assert resolved["context"] == {"pending_id": "pr-run-1"}
    # The plane reads it only on the resolution.
    assert paused["context"] is None


def test_a_failed_registration_is_reported_not_left_to_a_guess(
    plane: GovPlane, tmp_path: Path
) -> None:
    """The pause still happens when ``POST /runs/{id}/pending`` fails, and the tool
    still waits for the app. Its resolution says there is no pending run — ``null``,
    key present — so the plane closes nothing instead of shifting it onto the
    run's next pause."""
    plane.fail_pending_post = True
    ran, tool = _bank()
    llm = _Script(TRANSFER)
    agent = _banker(llm, tool, tmp_path / "ckpt.db")

    paused = asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-1"))
    assert paused.status == "paused", paused
    assert paused.pending_interrupt is not None
    assert paused.pending_interrupt["context"]["pending_id"] is None
    asyncio.run(agent.aresume("run-1", resume_value=Resume(approved=False)))

    _ledger(plane, "run-1")  # drain
    resolved = plane.ledger("run-1")[-1]
    assert resolved["context"] == {"pending_id": None}
    assert ran == [] and llm.told() == [REFUSAL]


def test_a_pause_saved_before_1_76_keeps_the_positional_match() -> None:
    """A pause checkpointed by 1.75.0 or earlier has no ``pending_id`` in its saved
    context. Sending ``null`` for it would tell the plane there is no pending run
    when there is one; sending nothing keeps the match it was paused under."""
    before_1_76 = {"tool": "transfer_funds", "run_id": "r", "approval_request_id": "a"}
    assert governance.resolution_context(governance.APPROVAL_REASON, before_1_76) is None
    assert governance.resolution_context(governance.APPROVAL_REASON, None) is None
    # And a user interrupt never carries one, whatever its context holds.
    assert governance.resolution_context("need_sign_off", {"pending_id": "x"}) is None


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
    # One run id from pause to resolution: the Chain's. The agent's own run id is
    # thrown away — on resume the Chain re-runs the agent under a new one — so the
    # pending run registered under it could never be closed (fixed 1.76.0).
    assert [p["run_id"] for p in plane.pending_posts] == ["run-c"]
    assert paused.pending_interrupt is not None
    assert paused.pending_interrupt["context"]["run_id"] == "run-c"  # what the app resumes
    assert plane.ledger("run-c")[-1]["context"] == {"pending_id": "pr-run-c"}


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
