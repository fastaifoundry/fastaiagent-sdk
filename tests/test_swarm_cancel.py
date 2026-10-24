"""Swarm runs can be cancelled and timed out, and record how they ended (1.93.0).

Before 1.93.0 ``Swarm.arun`` could not be cancelled (documented) while
``Swarm.astream`` could, by accident; neither Swarm nor Supervisor had a run time
limit; a hop's timeout left ``run_reason`` empty; and a streamed swarm wrote no
run-end marker of its own, while each hop abandoned at its handoff wrote a
``failed`` one (``GeneratorExit``) — so a ``SwarmError`` was never recorded.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, Supervisor, Swarm, Worker
from fastaiagent._internal.errors import AgentTimeoutError
from fastaiagent.agent.swarm import SwarmError
from fastaiagent.llm.stream import StreamDone
from fastaiagent.testing import TestModel
from tests._cancel_helpers import cancel_when

#: Set when the slow tool starts; a cancel waits for it (a fixed delay raced
#: the run on slow CI runners).
_IN_TOOL = threading.Event()


async def aslow() -> str:
    """Take a while."""
    _IN_TOOL.set()
    await asyncio.sleep(5)
    return "late"


def _slow_swarm(store: SQLiteCheckpointer, **kwargs: Any) -> Swarm:
    agent = Agent(
        name="a",
        llm=TestModel(response=["", "done"], tool_calls=[{"name": "aslow", "arguments": {}}]),
        tools=[FunctionTool(name="aslow", fn=aslow)],
    )
    return Swarm(
        name="s", agents=[agent], entrypoint="a", handoffs={"a": []}, checkpointer=store, **kwargs
    )


def _handoff_swarm(store: SQLiteCheckpointer, max_handoffs: int = 2) -> Swarm:
    a = Agent(
        name="a",
        llm=TestModel(
            response=["", "unused"],
            tool_calls=[{"name": "handoff_to_b", "arguments": {"reason": "b knows"}}],
        ),
    )
    b = Agent(name="b", llm=TestModel(response="b answered"))
    return Swarm(
        name="s",
        agents=[a, b],
        entrypoint="a",
        handoffs={"a": ["b"], "b": []},
        max_handoffs=max_handoffs,
        checkpointer=store,
    )


def _stream(swarm: Swarm, execution_id: str) -> list[Any]:
    async def drain() -> list[Any]:
        return [e async for e in swarm.astream("go", execution_id=execution_id)]

    return asyncio.run(drain())


@pytest.fixture
def store(tmp_path: Path) -> SQLiteCheckpointer:
    return SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))


def _run_end(store: SQLiteCheckpointer, execution_id: str) -> Any:
    last = store.get_last(execution_id)
    assert last is not None and last.node_id == "run_end", last
    return last


def test_swarm_arun_can_be_cancelled(store: SQLiteCheckpointer) -> None:
    swarm = _slow_swarm(store)
    _IN_TOOL.clear()
    cancel_when(_IN_TOOL, swarm.cancel, "sw-1")
    started = time.monotonic()
    result = swarm.run("go", execution_id="sw-1")
    assert time.monotonic() - started < 2
    assert result.status == "cancelled" and result.stop_reason == "cancelled"
    end = _run_end(store, "sw-1")
    assert (end.status, end.state_snapshot["run_reason"]) == ("failed", "cancelled")


def test_swarm_astream_can_be_cancelled(store: SQLiteCheckpointer) -> None:
    swarm = _slow_swarm(store)
    _IN_TOOL.clear()
    cancel_when(_IN_TOOL, swarm.cancel, "sw-2")
    started = time.monotonic()
    events = _stream(swarm, "sw-2")
    assert time.monotonic() - started < 2
    assert isinstance(events[-1], StreamDone) and events[-1].finish_reason == "cancelled"
    end = _run_end(store, "sw-2")
    assert (end.status, end.state_snapshot["run_reason"]) == ("failed", "cancelled")


@pytest.mark.parametrize("method", ["arun", "astream"])
def test_a_swarm_timeout_stops_the_run(store: SQLiteCheckpointer, method: str) -> None:
    swarm = _slow_swarm(store, timeout_seconds=0.3)
    started = time.monotonic()
    with pytest.raises(AgentTimeoutError):
        if method == "arun":
            swarm.run("go", execution_id="sw-3")
        else:
            _stream(swarm, "sw-3")
    assert time.monotonic() - started < 2
    end = _run_end(store, "sw-3")
    assert (end.status, end.state_snapshot["run_reason"]) == ("failed", "timeout")


def test_a_streamed_handoff_ends_completed_with_no_false_failure(
    store: SQLiteCheckpointer,
) -> None:
    events = _stream(_handoff_swarm(store), "sw-4")
    assert any(type(e).__name__ == "HandoffEvent" for e in events)
    assert _run_end(store, "sw-4").status == "completed"
    errors = [cp.state_snapshot.get("run_error") for cp in store.list("sw-4", limit=100)]
    assert not any(e and "GeneratorExit" in e for e in errors), errors


def test_a_streamed_swarm_error_is_recorded(store: SQLiteCheckpointer) -> None:
    with pytest.raises(SwarmError, match="max_handoffs"):
        _stream(_handoff_swarm(store, max_handoffs=0), "sw-5")
    end = _run_end(store, "sw-5")
    assert end.status == "failed" and "SwarmError" in end.state_snapshot["run_error"]


def test_a_finished_swarm_cannot_be_cancelled(store: SQLiteCheckpointer) -> None:
    swarm = _handoff_swarm(store)
    assert swarm.run("go", execution_id="sw-6").output == "b answered"
    assert swarm.cancel("sw-6") is False


def test_a_supervisor_timeout_stops_the_run() -> None:
    worker = Agent(
        name="w",
        llm=TestModel(response=["", "x"], tool_calls=[{"name": "aslow", "arguments": {}}]),
        tools=[FunctionTool(name="aslow", fn=aslow)],
    )
    lead = TestModel(
        response=["", "done"],
        tool_calls=[{"name": "delegate_to_w", "arguments": {"task": "go"}}],
    )
    sup = Supervisor(name="lead", llm=lead, workers=[Worker(agent=worker)], timeout_seconds=0.3)
    started = time.monotonic()
    with pytest.raises(AgentTimeoutError):
        sup.run("go")
    assert time.monotonic() - started < 2


def test_timeout_seconds_must_be_positive(store: SQLiteCheckpointer) -> None:
    with pytest.raises(SwarmError):
        _slow_swarm(store, timeout_seconds=0)
    with pytest.raises(ValueError):
        Supervisor(name="x", workers=[], timeout_seconds=-1)
