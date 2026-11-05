"""``Supervisor(memory=...)`` — the team's own conversation (1.99.0).

A Supervisor took no memory. A follow-up ("which order was that?") worked only
when a worker happened to have memory; live, the lead delegated "the order
referenced in the conversation" without knowing which one.

No mocks: the offline ``FunctionModel`` from ``fastaiagent.testing`` and a real
``AgentMemory`` / ``SQLiteCheckpointer``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.agent.memory import AgentMemory
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.tool import FunctionTool


def _lead(seen: list[list[str]]) -> FunctionModel:
    def route(messages: list[Any]) -> Any:
        if messages[-1].role.value == "tool":
            return f"answer: {messages[-1].content}"
        seen.append([f"{m.role.value}:{m.content}" for m in messages if m.role.value != "system"])
        return "", [{"name": "delegate_to_orders", "arguments": {"task": "look it up"}}]

    return FunctionModel(route)


def _team(seen: list[list[str]], memory: AgentMemory, **kw: Any) -> fa.Supervisor:
    worker = fa.Agent(name="orders", llm=TestModel("order A100"))
    return fa.Supervisor(
        name="mem-lead",
        llm=_lead(seen),
        workers=[fa.Worker(worker, role="orders")],
        memory=memory,
        **kw,
    )


def _drive(team: fa.Supervisor, path: str, text: str) -> None:
    if path == "run":
        team.run(text)
    elif path == "stream":
        team.stream(text)
    else:

        async def consume() -> None:
            async for _ in team.astream(text):
                pass

        asyncio.run(consume())


@pytest.mark.parametrize("path", ["run", "stream", "astream"])
def test_the_next_turn_sees_the_last(path: str) -> None:
    seen: list[list[str]] = []
    memory = AgentMemory()
    team = _team(seen, memory)

    _drive(team, path, "Where is my order?")
    _drive(team, path, "Which order was that?")

    assert seen[1] == [
        "user:Where is my order?",
        "assistant:answer: order A100",
        "user:Which order was that?",
    ]
    # One request and one answer per turn: no delegate mechanics in it.
    assert [m.role.value for m in memory.get_context()] == ["user", "assistant"] * 2


def test_the_pushed_lead_says_it_has_memory() -> None:
    team = _team([], AgentMemory())
    assert team._build_inner_agent().to_dict()["memory_enabled"] is True


def test_decision_routing_refuses_memory() -> None:
    workers = [
        fa.Worker(fa.Agent(name="x", llm=TestModel()), role="x"),
        fa.Worker(fa.Agent(name="y", llm=TestModel()), role="y"),
    ]
    with pytest.raises(ValueError, match="routing='tools'"):
        fa.Supervisor(
            name="desk",
            workers=workers,
            routing="decisions",
            fallback_worker="y",
            memory=AgentMemory(),
        )


def test_resume_continues_the_current_turn_not_an_earlier_one(tmp_path: Any) -> None:
    """The lead's memory puts earlier turns ahead of this one in its checkpoint;
    resume read the FIRST user message and would re-run the old request."""
    calls: list[str] = []

    def ask(task: str) -> str:
        calls.append(task)
        if len(calls) == 2:
            interrupt("manager_approval", {"task": task})
        return f"done {task}"

    def route(messages: list[Any]) -> Any:
        if messages[-1].role.value == "tool":
            return f"answer: {messages[-1].content}"
        return "", [{"name": "delegate_to_w", "arguments": {"task": str(messages[-1].content)}}]

    worker_route = FunctionModel(
        lambda m: (
            f"worker: {m[-1].content}"
            if m[-1].role.value == "tool"
            else ("", [{"name": "ask", "arguments": {"task": str(m[-1].content)}}])
        )
    )
    worker = fa.Agent(name="w", llm=worker_route, tools=[FunctionTool(name="ask", fn=ask)])
    team = fa.Supervisor(
        name="mem-resume",
        llm=FunctionModel(route),
        workers=[fa.Worker(worker, role="w")],
        memory=AgentMemory(),
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )

    first = team.run("first request")
    assert first.status == "completed"
    paused = team.run("second request")
    assert paused.status == "paused"

    resumed = team.resume(paused.execution_id, resume_value=Resume(approved=True))

    assert resumed.status == "completed"
    assert "second request" in resumed.output and "first request" not in resumed.output
