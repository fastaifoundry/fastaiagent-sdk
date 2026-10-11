"""A tool re-invoked by crash recovery may pause, and the pause is recorded (1.88.0).

The process dies inside a tool after its pre-tool checkpoint, before the tool
reached its ``interrupt()``. ``resume()`` re-invokes the saved tool, which now
pauses. The tool loop turns that ``InterruptSignal`` into a recorded pause;
``Agent.aresume``'s direct re-invocation did not, so until 1.88.0 the signal
escaped ``resume()`` with nothing checkpointed, and every later resume re-ran the
tool and raised again.

Real crash: a child process killed with ``os._exit`` inside the tool, on a real
SQLite checkpointer. No mocks.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from fastaiagent import Agent, FunctionTool, Resume, SQLiteCheckpointer, interrupt
from fastaiagent.testing import FunctionModel

charges: list[int] = []


def refund(order: str, amount: int) -> dict:
    """Refund an order; large amounts need a manager."""
    decision = interrupt(reason="manager_approval", context={"order": order, "amount": amount})
    if not decision.approved:
        return {"approved": False}
    charges.append(amount)
    return {"approved": True}


def responder(messages):
    if any(m.role.value == "tool" for m in messages):
        return "Refund done."
    return "", [{"name": "refund", "arguments": {"order": "1042", "amount": 50000}}]


def _agent(db: Path) -> Agent:
    return Agent(
        name="refund-bot",
        llm=FunctionModel(responder),
        tools=[FunctionTool(name="refund", fn=refund)],
        checkpointer=SQLiteCheckpointer(str(db)),
    )


CHILD = textwrap.dedent(
    """
    import os, sys
    from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
    from fastaiagent.testing import FunctionModel

    def refund(order: str, amount: int) -> dict:
        os._exit(2)   # the process dies inside the tool, after its checkpoint

    def responder(messages):
        return "", [{"name": "refund", "arguments": {"order": "1042", "amount": 50000}}]

    Agent(name="refund-bot", llm=FunctionModel(responder),
          tools=[FunctionTool(name="refund", fn=refund)],
          checkpointer=SQLiteCheckpointer(sys.argv[1])).run("Refund 1042", execution_id="crash-1")
    """
)


def _crash(db: Path, tmp_path: Path) -> None:
    run = subprocess.run(
        [sys.executable, "-c", CHILD, str(db)], capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert run.returncode == 2, run.stderr[-1500:]


def test_a_pause_during_crash_recovery_is_recorded_and_resumable(tmp_path: Path) -> None:
    db = tmp_path / "ckpt.db"
    _crash(db, tmp_path)
    store = SQLiteCheckpointer(str(db))
    assert store.get_last("crash-1").node_id == "turn:0/tool:refund"

    paused = _agent(db).resume("crash-1")

    assert paused.status == "paused"
    assert paused.pending_interrupt["reason"] == "manager_approval"
    assert paused.pending_interrupt["node_id"] == "turn:0/tool:refund"
    assert [p.execution_id for p in store.list_pending_interrupts()] == ["crash-1"]
    last = store.get_last("crash-1")
    assert (last.status, last.step_type) == ("interrupted", "hitl_pause")

    charges.clear()
    done = _agent(db).resume("crash-1", resume_value=Resume(approved=True))
    assert done.status == "completed"
    assert done.output == "Refund done."
    assert charges == [50000]


def test_a_second_crash_resume_does_not_re_pause(tmp_path: Path) -> None:
    """Once the pause is recorded, a resume without a decision is refused as an
    interrupted run, rather than re-running the tool into a second pause."""
    import pytest

    from fastaiagent._internal.errors import ChainCheckpointError

    db = tmp_path / "ckpt.db"
    _crash(db, tmp_path)
    _agent(db).resume("crash-1")
    with pytest.raises(ChainCheckpointError, match="suspended on interrupt"):
        _agent(db).resume("crash-1")
    store = SQLiteCheckpointer(str(db))
    assert len(store.list_pending_interrupts()) == 1
