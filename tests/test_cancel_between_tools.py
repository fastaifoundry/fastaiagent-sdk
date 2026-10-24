"""A cancel or timeout during a blocking sync tool stops the turn there (1.93.0).

A sync tool blocks the event loop, so ``arun``'s task only saw its cancel at the
next real await: the turn's remaining tools ran, and the next turn's
``before_model`` hooks fired, after the run was stopped (found against a live
model). The loop now checks before each tool call and at the top of each turn.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from fastaiagent import Agent, AgentConfig, FunctionTool
from fastaiagent._internal.errors import AgentTimeoutError
from fastaiagent.agent.middleware import AgentMiddleware
from fastaiagent.testing import TestModel
from tests._cancel_helpers import cancel_when

#: Set when the first blocking tool starts; the cancel waits for it (a fixed
#: delay raced the run on slow CI runners).
_in_tool: dict[str, threading.Event] = {}


def _agent(log: list[str], sleep: float = 0.4, **config: object) -> Agent:
    started = threading.Event()

    def slow(step: str) -> str:
        log.append(f"start {step}")
        started.set()
        time.sleep(sleep)
        log.append(f"end {step}")
        return step

    class Hooks(AgentMiddleware):
        async def before_model(self, ctx, messages):  # type: ignore[no-untyped-def]
            log.append("before_model")
            return messages

    agent = Agent(
        name="blocking",
        llm=TestModel(
            response=["", "done"],
            tool_calls=[
                {"name": "slow", "arguments": {"step": "one"}},
                {"name": "slow", "arguments": {"step": "two"}},
            ],
        ),
        tools=[FunctionTool(name="slow", fn=slow)],
        middleware=[Hooks()],
        config=AgentConfig(**config),  # type: ignore[arg-type]
    )
    _in_tool[agent.name] = started
    return agent


def _cancel_soon(agent: Agent, exec_id: str) -> None:
    cancel_when(_in_tool[agent.name], agent.cancel, exec_id)


def test_run_stops_after_the_blocking_tool() -> None:
    log: list[str] = []
    agent = _agent(log)
    _cancel_soon(agent, "c-run")
    result = agent.run("go", execution_id="c-run")
    assert result.status == "cancelled"
    assert log == ["before_model", "start one", "end one"]


def test_arun_stops_after_the_blocking_tool() -> None:
    log: list[str] = []
    agent = _agent(log)
    _cancel_soon(agent, "c-arun")
    result = asyncio.run(agent.arun("go", execution_id="c-arun"))
    assert result.status == "cancelled"
    assert log == ["before_model", "start one", "end one"]


def test_astream_stops_after_the_blocking_tool() -> None:
    log: list[str] = []
    agent = _agent(log)
    _cancel_soon(agent, "c-stream")
    result = agent.stream("go", execution_id="c-stream")
    assert result.status == "cancelled"
    assert log == ["before_model", "start one", "end one"]


def test_astream_stops_inside_a_slow_async_tool() -> None:
    """A streamed run stopped only at its next event: a cancel waited out the
    tool in flight. The tool call now runs as the run's task."""
    done: list[str] = []
    in_tool = threading.Event()

    async def aslow() -> str:
        in_tool.set()
        await asyncio.sleep(5)
        done.append("finished")
        return "late"

    agent = Agent(
        name="async-slow",
        llm=TestModel(response=["", "done"], tool_calls=[{"name": "aslow", "arguments": {}}]),
        tools=[FunctionTool(name="aslow", fn=aslow)],
    )
    cancel_when(in_tool, agent.cancel, "c-async")
    started = time.monotonic()
    result = agent.stream("go", execution_id="c-async")
    assert result.status == "cancelled"
    assert time.monotonic() - started < 2
    assert done == []


def test_a_streamed_timeout_lands_inside_a_slow_async_tool() -> None:
    async def aslow() -> str:
        await asyncio.sleep(5)
        return "late"

    agent = Agent(
        name="async-timeout",
        llm=TestModel(response=["", "done"], tool_calls=[{"name": "aslow", "arguments": {}}]),
        tools=[FunctionTool(name="aslow", fn=aslow)],
        config=AgentConfig(timeout_seconds=0.3),
    )
    started = time.monotonic()
    with pytest.raises(AgentTimeoutError):
        agent.stream("go")
    assert time.monotonic() - started < 2


def test_a_timeout_stops_after_the_blocking_tool() -> None:
    log: list[str] = []
    # Margins wide enough that a slow runner still times out inside the tool.
    agent = _agent(log, sleep=2.0, timeout_seconds=1.0)
    with pytest.raises(AgentTimeoutError):
        agent.run("go")
    assert log == ["before_model", "start one", "end one"]
