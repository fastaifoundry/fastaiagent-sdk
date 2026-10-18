"""``SQLiteCheckpointer(":memory:")`` works (1.90.0, F9).

The LangGraph migration guide recommends it for tests (it is our
``MemorySaver``). It never worked: ``setup()`` ran the migrations on a helper it
then closed, and an in-memory SQLite database lives only as long as its
connection — so the next call opened a fresh, empty one and raised
``no such table``. Real SQLite, real Agent pause and resume.
"""

from __future__ import annotations

import threading

from fastaiagent import Agent, FunctionTool, Resume, SQLiteCheckpointer, interrupt
from fastaiagent.testing import FunctionModel


def test_setup_then_reads_work() -> None:
    cp = SQLiteCheckpointer(":memory:")
    cp.setup()
    assert cp.list_pending_interrupts() == []


def test_a_paused_agent_resumes_from_memory() -> None:
    ran: list[str] = []

    def refund(order: str) -> str:
        """Refund an order; a manager approves."""
        if interrupt(reason="manager_approval", context={"order": order}).approved:
            ran.append(order)
            return f"refunded {order}"
        return "declined"

    def respond(messages):
        if any(m.role.value == "tool" for m in messages):
            return "Done."
        return "", [{"name": "refund", "arguments": {"order": "1042"}}]

    cp = SQLiteCheckpointer(":memory:")
    agent = Agent(
        name="refund-bot",
        llm=FunctionModel(respond),
        tools=[FunctionTool(name="refund", fn=refund)],
        checkpointer=cp,
    )

    first = agent.run("Refund 1042", execution_id="mem-1", trace=False)
    assert first.status == "paused"
    assert [p.execution_id for p in cp.list_pending_interrupts()] == ["mem-1"]

    done = agent.resume("mem-1", resume_value=Resume(approved=True), trace=False)
    assert done.status == "completed"
    assert ran == ["1042"]


def test_two_in_memory_checkpointers_are_separate() -> None:
    a, b = SQLiteCheckpointer(":memory:"), SQLiteCheckpointer(":memory:")
    a.setup()
    b.setup()
    from fastaiagent.chain.checkpoint import Checkpoint

    a.put(Checkpoint(chain_name="c", execution_id="only-in-a", node_id="n", status="completed"))
    assert a.list("only-in-a") and not b.list("only-in-a")


def test_an_in_memory_store_leaves_the_working_directory_alone(tmp_path, monkeypatch) -> None:
    """``Path(":memory:").parent`` is ``.``, and the helper tightened the database
    file's parent to 0o700 — so on 1.89.0 opening an in-memory checkpointer
    stripped group/other access from the caller's current directory."""
    import os
    import stat

    work = tmp_path / "project"
    work.mkdir()
    os.chmod(work, 0o755)
    monkeypatch.chdir(work)

    cp = SQLiteCheckpointer(":memory:")
    cp.setup()
    cp.list_pending_interrupts()

    assert stat.S_IMODE(os.stat(work).st_mode) == 0o755
    assert not any(p.name.startswith((":memory:", "file:")) for p in work.iterdir())


def test_another_thread_sees_the_same_database() -> None:
    """The plane replica drains on a daemon thread; it must see the same rows."""
    from fastaiagent.chain.checkpoint import Checkpoint

    cp = SQLiteCheckpointer(":memory:")
    cp.put(Checkpoint(chain_name="c", execution_id="x", node_id="n", status="completed"))
    seen: list[int] = []
    t = threading.Thread(target=lambda: seen.append(len(cp.list("x"))))
    t.start()
    t.join()
    assert seen == [1]
