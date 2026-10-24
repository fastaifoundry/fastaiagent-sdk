"""Parallel tool execution in the agent tool-calling loop.

Runs without API keys. A scripted local LLM (a stand-in that returns
pre-baked ``LLMResponse`` objects — *not* a mock of our own code) drives the
real :func:`execute_tool_loop`, so the concurrency path, ordering, and
sequential-fallback gate are all exercised end to end.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from fastaiagent.agent.executor import execute_tool_loop
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import Message, ToolCall, UserMessage
from fastaiagent.tool.function import FunctionTool

_SLEEP = 0.25


class _ScriptedLLM:
    """Returns each pre-baked response in turn from ``acomplete``."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self._i = 0

    async def acomplete(self, messages, tools=None, **kwargs) -> LLMResponse:
        resp = self._responses[self._i]
        self._i += 1
        return resp


def _two_call_script() -> list[LLMResponse]:
    return [
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="slow", arguments={"tag": "a"}),
                ToolCall(id="call_2", name="slow", arguments={"tag": "b"}),
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="final answer", finish_reason="stop"),
    ]


#: Each tool call's (start, end), for the overlap checks. A wall-clock bound on
#: the whole loop flaked on slow CI runners (macOS: 0.567 s for two 0.25 s
#: calls that did overlap) — loop overhead, not sequencing.
_SPANS: list[tuple[float, float]] = []


def _overlapped() -> bool:
    (s1, e1), (s2, e2) = sorted(_SPANS)
    return s2 < e1


async def _run(**loop_kwargs) -> tuple[float, list[Message], list[dict]]:
    def make_tool() -> FunctionTool:
        async def slow(tag: str) -> str:
            started = time.perf_counter()
            await asyncio.sleep(_SLEEP)
            _SPANS.append((started, time.perf_counter()))
            return f"done-{tag}"

        return FunctionTool(name="slow", fn=slow)

    _SPANS.clear()
    messages: list[Message] = [UserMessage(content="go")]
    llm = _ScriptedLLM(_two_call_script())
    start = time.monotonic()
    _, all_calls = await execute_tool_loop(llm, messages, [make_tool()], **loop_kwargs)
    return time.monotonic() - start, messages, all_calls


@pytest.mark.asyncio
async def test_parallel_overlaps_and_preserves_order() -> None:
    _, messages, all_calls = await _run(parallel_tools=True, max_parallel_tools=4)

    # The two calls ran at the same time: each started before the other ended.
    assert _overlapped(), f"expected overlap: {_SPANS}"

    # Tool results are appended in call order regardless of completion order.
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["call_1", "call_2"]
    assert [c["tool_call_id"] for c in all_calls] == ["call_1", "call_2"]


@pytest.mark.asyncio
async def test_sequential_by_default() -> None:
    _, messages, _ = await _run()  # parallel_tools defaults to False
    assert not _overlapped(), f"expected sequential: {_SPANS}"
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["call_1", "call_2"]


@pytest.mark.asyncio
async def test_governance_agent_id_forces_sequential() -> None:
    # agent_id set => managed governance is order/identity-sensitive, so the
    # loop must fall back to sequential even with parallel_tools=True.
    await _run(parallel_tools=True, agent_id="agent-123")
    assert not _overlapped(), f"expected sequential fallback: {_SPANS}"


def _sync_agent(parallel: bool, spans: list[tuple[float, float]] | None = None) -> object:
    from fastaiagent import Agent, AgentConfig
    from fastaiagent.testing import TestModel

    def slow(tag: str) -> str:
        started = time.perf_counter()
        time.sleep(_SLEEP)  # a plain blocking tool, the common case
        if spans is not None:
            spans.append((started, time.perf_counter()))
        return f"done-{tag}"

    return Agent(
        name="sync-tools",
        llm=TestModel(
            response=["", "final answer"],
            tool_calls=[
                {"name": "slow", "arguments": {"tag": "a"}},
                {"name": "slow", "arguments": {"tag": "b"}},
            ],
        ),
        tools=[FunctionTool(name="slow", fn=slow)],
        config=AgentConfig(parallel_tools=parallel),
    )


@pytest.mark.parametrize("method", ["run", "stream"])
def test_plain_sync_tools_overlap_too(method: str) -> None:
    """1.93.0: sync tools ran on the event loop, so ``parallel_tools=True`` gave
    them no overlap (two 2 s tools took 4 s against a live model)."""
    spans: list[tuple[float, float]] = []
    result = getattr(_sync_agent(True, spans), method)("go")
    assert [c["output"] for c in result.tool_calls] == ["done-a", "done-b"]
    # The calls' own intervals overlap: each started before the other ended.
    # (A wall-clock bound flaked on slow CI runners: overhead, not overlap.)
    (s1, e1), (s2, e2) = sorted(spans)
    assert s2 < e1, f"sync tools ran back to back: {spans}"


def test_sync_tools_stay_sequential_without_parallel_tools() -> None:
    spans: list[tuple[float, float]] = []
    _sync_agent(False, spans).run("go")
    (s1, e1), (s2, e2) = sorted(spans)
    assert s2 >= e1
