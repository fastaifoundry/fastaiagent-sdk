"""A Chain run can be cancelled, and its tool nodes are traced (1.93.0).

Before 1.93.0 a Chain could not be cancelled (documented), and cancelling an
agent node by its own execution id — which its tools can read — made the chain
carry on with ``output=""`` and report ``completed``. An agent node's timeout
failed the chain with no ``run_reason``. Tool nodes emitted no span, though the
docs say every tool nests under the chain root.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, AgentConfig, Chain, FunctionTool, SQLiteCheckpointer
from fastaiagent._internal.errors import AgentTimeoutError
from fastaiagent.agent.context import RunContext
from fastaiagent.testing import TestModel


async def aslow() -> str:
    """Take a while."""
    await asyncio.sleep(5)
    return "late"


def _slow_agent(**config: Any) -> Agent:
    return Agent(
        name="worker",
        llm=TestModel(response=["", "worked"], tool_calls=[{"name": "aslow", "arguments": {}}]),
        tools=[FunctionTool(name="aslow", fn=aslow)],
        config=AgentConfig(**config),
    )


def _chain(store: SQLiteCheckpointer, agent: Agent, ran: list[str]) -> Chain:
    def after() -> str:
        """Runs after the agent node."""
        ran.append("after")
        return "after"

    chain = Chain("pipeline", checkpointer=store)
    chain.add_node("work", agent=agent)
    chain.add_node("after", tool=FunctionTool(name="after", fn=after))
    chain.connect("work", "after")
    return chain


@pytest.fixture
def store(tmp_path: Path) -> SQLiteCheckpointer:
    return SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))


def test_a_chain_can_be_cancelled_by_its_id(store: SQLiteCheckpointer) -> None:
    ran: list[str] = []
    chain = _chain(store, _slow_agent(), ran)
    threading.Timer(0.3, chain.cancel, args=("ch-1",)).start()
    started = time.monotonic()
    result = asyncio.run(chain.aexecute({"message": "go"}, execution_id="ch-1"))
    assert time.monotonic() - started < 2, "the cancel waited out the agent's tool"
    assert result.status == "cancelled" and ran == []
    last = store.get_last("ch-1")
    assert last is not None and last.node_id == "run_end" and last.status == "failed"
    assert last.state_snapshot["run_reason"] == "cancelled"


def test_cancelling_an_agent_node_stops_the_chain(store: SQLiteCheckpointer) -> None:
    ran: list[str] = []
    seen: dict[str, str] = {}

    async def note_and_wait(ctx: RunContext[None]) -> str:
        """Record the run id, then take a while."""
        seen["id"] = ctx.run.execution_id
        await asyncio.sleep(5)
        return "late"

    agent = Agent(
        name="worker",
        llm=TestModel(
            response=["", "worked"], tool_calls=[{"name": "note_and_wait", "arguments": {}}]
        ),
        tools=[FunctionTool(name="note_and_wait", fn=note_and_wait)],
    )
    chain = _chain(store, agent, ran)

    def cancel_the_node() -> None:
        while "id" not in seen:
            time.sleep(0.02)
        assert agent.cancel(seen["id"]) is True

    threading.Thread(target=cancel_the_node).start()
    result = chain.execute({"message": "go"}, context=RunContext(state=None), execution_id="ch-2")
    assert result.status == "cancelled"
    assert ran == [], "the next node ran on the cancelled node's empty output"


def test_a_blocking_tool_node_lets_no_later_node_start(store: SQLiteCheckpointer) -> None:
    ran: list[str] = []

    def blocking() -> str:
        """Blocks the loop."""
        time.sleep(0.4)
        ran.append("blocking")
        return "ok"

    def after() -> str:
        """Runs last."""
        ran.append("after")
        return "after"

    chain = Chain("blocking-chain", checkpointer=store)
    chain.add_node("first", tool=FunctionTool(name="blocking", fn=blocking))
    chain.add_node("after", tool=FunctionTool(name="after", fn=after))
    chain.connect("first", "after")
    threading.Timer(0.1, chain.cancel, args=("ch-3",)).start()
    result = chain.execute({}, execution_id="ch-3")
    assert result.status == "cancelled"
    assert ran == ["blocking"]


def test_an_agent_nodes_timeout_is_recorded(store: SQLiteCheckpointer) -> None:
    chain = _chain(store, _slow_agent(timeout_seconds=0.2), [])
    with pytest.raises(AgentTimeoutError):
        chain.execute({"message": "go"}, execution_id="ch-4")
    last = store.get_last("ch-4")
    assert last is not None and last.status == "failed"
    assert last.state_snapshot["run_reason"] == "timeout"


def test_cancel_of_an_unknown_or_finished_chain_is_false(store: SQLiteCheckpointer) -> None:
    chain = Chain("quick", checkpointer=store)
    chain.add_node("only", tool=FunctionTool(name="only", fn=lambda: "x"))
    assert chain.execute({}, execution_id="ch-5").status == "completed"
    assert chain.cancel("ch-5") is False
    assert chain.cancel("never-ran") is False


def test_tool_nodes_are_traced_under_the_chain(
    store: SQLiteCheckpointer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    otel.reset()

    def double(n: int) -> int:
        """Twice n."""
        return n * 2

    def boom() -> str:
        """Fails."""
        raise RuntimeError("tool node broke")

    chain = Chain("traced", checkpointer=store)
    chain.add_node("a", tool=FunctionTool(name="double", fn=double), input_mapping={"n": 21})
    chain.add_node("b", tool=FunctionTool(name="boom", fn=boom))
    chain.connect("a", "b")
    try:
        with pytest.raises(Exception, match="tool node broke"):
            chain.execute({})
        otel.get_tracer_provider().force_flush()
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute("SELECT * FROM spans")]
    finally:
        otel.reset()
        reset_config()
    by_name = {r["name"]: r for r in rows}
    root = by_name["chain.traced"]
    for name in ("tool.double", "tool.boom"):
        assert by_name[name]["parent_span_id"] == root["span_id"]
    assert '"tool.status": "ok"' in by_name["tool.double"]["attributes"]
    assert '"tool.result": "42"' in by_name["tool.double"]["attributes"]
    assert '"tool.status": "error"' in by_name["tool.boom"]["attributes"]
