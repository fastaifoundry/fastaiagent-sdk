"""``Tool(timeout=)`` applies to sync functions too (1.90.0).

``Tool.ainvoke`` enforces ``timeout`` with ``asyncio.wait_for``. A
``FunctionTool`` called its function directly on the event loop, so a sync
function blocked the loop and ``wait_for`` could never fire: a sync tool with
``timeout=0.1`` that took ten seconds took ten seconds and returned its result.

Since 1.90.0 a sync function with a ``timeout`` runs in a worker thread (the
caller's context vars are copied, so ``interrupt()`` / ``@idempotent`` still
work), and the wait is bounded. A thread cannot be killed: a timed-out function
finishes in the background, its side effects included. Without ``timeout`` the
function still runs on the calling thread, exactly as before.

Real FunctionTool, real Agent loop, ``FunctionModel``.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, Resume, SQLiteCheckpointer, interrupt
from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.agent.context import RunContext
from fastaiagent.testing import FunctionModel


def _slow(seconds: float) -> str:
    """Sleep, then answer."""
    time.sleep(seconds)
    return "finished"


class _Blocked:
    """A sync tool body that blocks until released, and says whether it finished.

    The timeout tests assert the call came back while this was still blocked: an
    event, not a wall-clock bound, which a slow CI runner broke (0.88 s against
    0.8 s on macOS). The 30 s cap only keeps a regression from hanging the suite;
    if it is ever reached, ``finished`` is set and the test fails.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self) -> str:
        self.release.wait(30)
        self.finished.set()
        return "finished"


def test_a_sync_tool_times_out() -> None:
    blocked = _Blocked()
    tool = FunctionTool(name="slow", fn=blocked, timeout=0.1)
    try:
        with pytest.raises(ToolExecutionError, match="timed out"):
            asyncio.run(tool.ainvoke({}))
        assert not blocked.finished.is_set(), "the timeout did not bound the call"
    finally:
        blocked.release.set()


def test_a_sync_tool_within_its_timeout_answers() -> None:
    tool = FunctionTool(name="quick", fn=lambda: _slow(0.0), timeout=2.0)
    assert asyncio.run(tool.ainvoke({})).output == "finished"


def test_the_agent_loop_hands_the_model_the_timeout() -> None:
    seen: list[str] = []

    def respond(messages: list[Any]):
        results = [str(m.content) for m in messages if m.role.value == "tool"]
        if results:
            seen.extend(results)
            return "gave up"
        return "", [{"name": "slow", "arguments": {}}]

    blocked = _Blocked()
    tool = FunctionTool(name="slow", fn=blocked, timeout=0.1)
    agent = Agent(name="t", llm=FunctionModel(respond), tools=[tool])
    try:
        assert agent.run("go", trace=False).output == "gave up"
        assert not blocked.finished.is_set(), "the run waited for the timed-out tool"
    finally:
        blocked.release.set()
    assert seen and "timed out" in seen[0]


def test_without_a_timeout_a_sync_tool_still_runs_on_the_calling_thread() -> None:
    ran_on: list[int] = []

    def where() -> str:
        """Record the thread."""
        ran_on.append(threading.get_ident())
        return "ok"

    asyncio.run(FunctionTool(name="where", fn=where).ainvoke({}))
    assert ran_on == [threading.get_ident()]


def test_run_context_reaches_a_timed_sync_tool() -> None:
    def whoami(ctx: RunContext[dict]) -> str:
        """Read the run context."""
        return ctx.state["user"]

    tool = FunctionTool(name="whoami", fn=whoami, timeout=2.0)
    result = asyncio.run(tool.ainvoke({}, context=RunContext(state={"user": "ada"})))
    assert result.output == "ada"


def test_interrupt_and_resume_work_inside_a_timed_sync_tool(tmp_path) -> None:
    """``interrupt()`` reads the execution's context vars; the worker thread
    must see them, on the pause and on the resume."""
    ran: list[str] = []

    def refund(order: str) -> str:
        """Refund an order; a manager approves."""
        if interrupt(reason="manager_approval", context={"order": order}).approved:
            ran.append(order)
            return f"refunded {order}"
        return "declined"

    def respond(messages: list[Any]):
        if any(m.role.value == "tool" for m in messages):
            return "Done."
        return "", [{"name": "refund", "arguments": {"order": "1042"}}]

    agent = Agent(
        name="refund-bot",
        llm=FunctionModel(respond),
        tools=[FunctionTool(name="refund", fn=refund, timeout=5.0)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "c.db")),
    )
    exec_id = f"t-{uuid.uuid4().hex[:6]}"
    first = agent.run("Refund 1042", execution_id=exec_id, trace=False)
    assert first.status == "paused", first
    done = agent.resume(exec_id, resume_value=Resume(approved=True), trace=False)
    assert done.status == "completed"
    assert ran == ["1042"]
