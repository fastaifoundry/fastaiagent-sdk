"""``Supervisor.astream`` runs the same inner agent as ``arun`` (1.93.0).

It built one without the supervisor's checkpointer and opened no
``supervisor.*`` root span: a streamed supervisor run left no run-end marker
(cancelled or completed), nothing to resume, and an unrooted trace. And a
cancel waited for the step in flight — here a worker's whole tool call — since a
streamed run's tool calls were not stoppable mid-call.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer, Supervisor, Worker
from fastaiagent.llm.stream import StreamDone
from fastaiagent.testing import FunctionModel


async def aslow() -> str:
    """Take a while."""
    await asyncio.sleep(5)
    return "done"


def _supervisor(store: SQLiteCheckpointer, *, slow_worker: bool) -> Supervisor:
    worker = Agent(
        name="w",
        llm=FunctionModel(
            lambda m: ("", [{"name": "aslow", "arguments": {}}]) if slow_worker else "worked"
        ),
        tools=[FunctionTool(name="aslow", fn=aslow)],
    )
    turns = {"n": 0}

    def lead(messages: Any) -> Any:
        turns["n"] += 1
        if turns["n"] == 1:
            return "", [{"name": "delegate_to_w", "arguments": {"task": "go"}}]
        return "all done"

    return Supervisor(
        name="lead", llm=FunctionModel(lead), workers=[Worker(agent=worker)], checkpointer=store
    )


@pytest.fixture
def store(tmp_path: Path) -> SQLiteCheckpointer:
    return SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))


def _drain(sup: Supervisor, execution_id: str, cancel_after: float | None = None) -> list[Any]:
    async def main() -> list[Any]:
        events: list[Any] = []

        async def consume() -> None:
            async for event in sup.astream("go", execution_id=execution_id):
                events.append(event)

        task = asyncio.ensure_future(consume())
        if cancel_after is not None:
            await asyncio.sleep(cancel_after)
            assert Agent.cancel(sup.workers[0].agent, execution_id) is True
        await asyncio.wait_for(task, 5)
        return events

    return asyncio.run(main())


def test_a_completed_stream_leaves_a_run_end_marker(store: SQLiteCheckpointer) -> None:
    _drain(_supervisor(store, slow_worker=False), "ss-1")
    last = store.get_last("ss-1")
    assert last is not None and last.node_id == "run_end" and last.status == "completed"


def test_a_cancelled_stream_records_the_cancel(store: SQLiteCheckpointer) -> None:
    started = time.monotonic()
    events = _drain(_supervisor(store, slow_worker=True), "ss-2", cancel_after=0.3)
    # The worker's 5 s tool is stopped mid-call, not waited out.
    assert time.monotonic() - started < 2
    assert isinstance(events[-1], StreamDone) and events[-1].finish_reason == "cancelled"
    last = store.get_last("ss-2")
    assert last is not None and last.node_id == "run_end" and last.status == "failed"
    assert last.state_snapshot["run_reason"] == "cancelled"


def test_the_stream_has_a_supervisor_root_span(
    store: SQLiteCheckpointer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    otel.reset()
    try:
        _drain(_supervisor(store, slow_worker=False), "ss-3")
        otel.get_tracer_provider().force_flush()
        with sqlite3.connect(db) as conn:
            rows = conn.execute("SELECT name, parent_span_id FROM spans").fetchall()
    finally:
        otel.reset()
        reset_config()
    roots = [name for name, parent in rows if not parent]
    assert roots == ["supervisor.lead"]
    assert "agent.lead" in {name for name, _ in rows}
