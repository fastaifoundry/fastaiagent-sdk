"""The Chain approval gate (``NodeType.hitl``) means what it says (1.77.0, audit M9).

Three defects the approvals audit found, each reproduced on 1.73.0:

* **A rejection did not stop the chain.** The gate recorded ``approved=False`` and
  the next node ran anyway — the rejected work happened unless the author had wired
  a condition node to catch it. A rejection now stops the chain at the gate:
  ``status="rejected"``, nothing after it runs.
* **An ``async def`` handler was never awaited.** Its coroutine is truthy, so the
  gate approved itself, the next node ran, and only then did the run crash
  pickling the coroutine. It is awaited now.
* **A pause with nowhere to be saved reported ``paused``.** With
  ``checkpoint_enabled=False`` an ``interrupt()`` returned ``status="paused"`` with
  nothing persisted, and the resume it promised failed. It rises instead, like an
  Agent's pause with no checkpointer.

Real ``Chain``, real ``SQLiteCheckpointer``, real ``FunctionTool`` nodes; no model.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Chain, FunctionTool, interrupt
from fastaiagent.chain.interrupt import AlreadyResumed, InterruptSignal, Resume
from fastaiagent.chain.node import NodeType
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer


def _gated_chain(tmp_path: Path, sent: list[str], **chain_kw: Any) -> Chain:
    """``review`` (approval gate) → ``send`` (a side effect that must not run on a no)."""

    def send() -> str:
        sent.append("sent")
        return "sent"

    if "checkpoint_enabled" not in chain_kw:
        chain_kw["checkpointer"] = SQLiteCheckpointer(str(tmp_path / "chain.db"))
    chain = Chain("gate", **chain_kw)
    chain.add_node("review", type=NodeType.hitl)
    chain.add_node("send", tool=FunctionTool(name="send", fn=send))
    chain.connect("review", "send")
    return chain


def test_a_rejection_stops_the_chain_at_the_gate(tmp_path: Path) -> None:
    sent: list[str] = []
    chain = _gated_chain(tmp_path, sent)

    result = asyncio.run(chain.aexecute({}, execution_id="run-1", hitl_handler=lambda *_: False))

    assert result.status == "rejected", result
    assert sent == [], "the node after a rejected gate ran"
    assert result.node_results["review"] == {"approved": False}
    assert "send" not in result.node_results
    assert result.output is None


def test_an_approval_lets_the_chain_continue(tmp_path: Path) -> None:
    sent: list[str] = []
    chain = _gated_chain(tmp_path, sent)

    result = asyncio.run(chain.aexecute({}, execution_id="run-1", hitl_handler=lambda *_: True))

    assert result.status == "completed", result
    assert sent == ["sent"]
    assert result.node_results["review"] == {"approved": True}


def test_a_gate_that_did_not_say_yes_did_not_approve(tmp_path: Path) -> None:
    """A handler that forgets to return (``None``) is not an approval."""
    sent: list[str] = []
    chain = _gated_chain(tmp_path, sent)

    result = asyncio.run(chain.aexecute({}, execution_id="run-1", hitl_handler=lambda *_: None))

    assert result.status == "rejected", result
    assert sent == []


@pytest.mark.parametrize(
    ("answer", "status", "ran"), [(False, "rejected", []), (True, "completed", ["sent"])]
)
def test_an_async_handler_is_awaited(
    tmp_path: Path, answer: bool, status: str, ran: list[str]
) -> None:
    """Before 1.77.0 the coroutine itself was the answer: truthy, so ``send`` ran, and
    then the run crashed with ``TypeError: cannot pickle 'coroutine' object``."""
    sent: list[str] = []
    chain = _gated_chain(tmp_path, sent)

    async def handler(node: Any, context: Any, state: Any) -> bool:
        await asyncio.sleep(0)
        return answer

    result = asyncio.run(chain.aexecute({}, execution_id="run-1", hitl_handler=handler))

    assert result.status == status, result
    assert sent == ran
    assert result.node_results["review"] == {"approved": answer}


def test_a_rejected_run_is_over_and_cannot_be_resumed(tmp_path: Path) -> None:
    """``rejected`` ends the run: it gets the same end-of-run checkpoint a completed run
    does (status ``completed`` on the wire), so a resume refuses rather than replaying."""
    sent: list[str] = []
    chain = _gated_chain(tmp_path, sent)
    asyncio.run(chain.aexecute({}, execution_id="run-1", hitl_handler=lambda *_: False))

    with pytest.raises(AlreadyResumed):
        asyncio.run(chain.aresume("run-1", resume_value=Resume(approved=True)))
    assert sent == []


def test_a_pause_that_cannot_be_saved_is_not_reported_as_paused(tmp_path: Path) -> None:
    """``checkpoint_enabled=False``: nothing can hold the pause, so it rises to the
    caller instead of a ``paused`` result whose resume would fail."""

    def ask() -> str:
        decision = interrupt(reason="need_sign_off", context={})
        return "ok" if decision.approved else "no"

    chain = Chain("gate", checkpoint_enabled=False)
    chain.add_node("ask", tool=FunctionTool(name="ask", fn=ask))

    with pytest.raises(InterruptSignal):
        asyncio.run(chain.aexecute({}, execution_id="run-1"))
