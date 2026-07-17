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


async def _run(**loop_kwargs) -> tuple[float, list[Message], list[dict]]:
    def make_tool() -> FunctionTool:
        async def slow(tag: str) -> str:
            await asyncio.sleep(_SLEEP)
            return f"done-{tag}"

        return FunctionTool(name="slow", fn=slow)

    messages: list[Message] = [UserMessage(content="go")]
    llm = _ScriptedLLM(_two_call_script())
    start = time.monotonic()
    _, all_calls = await execute_tool_loop(llm, messages, [make_tool()], **loop_kwargs)
    return time.monotonic() - start, messages, all_calls


@pytest.mark.asyncio
async def test_parallel_overlaps_and_preserves_order() -> None:
    elapsed, messages, all_calls = await _run(parallel_tools=True, max_parallel_tools=4)

    # Two 0.25s tools running concurrently finish in ~0.25s, not ~0.5s.
    assert elapsed < 2 * _SLEEP, f"expected overlap, took {elapsed:.3f}s"

    # Tool results are appended in call order regardless of completion order.
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["call_1", "call_2"]
    assert [c["tool_call_id"] for c in all_calls] == ["call_1", "call_2"]


@pytest.mark.asyncio
async def test_sequential_by_default() -> None:
    elapsed, messages, _ = await _run()  # parallel_tools defaults to False
    assert elapsed >= 2 * _SLEEP * 0.9, f"expected sequential, took {elapsed:.3f}s"
    tool_msgs = [m for m in messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["call_1", "call_2"]


@pytest.mark.asyncio
async def test_governance_agent_id_forces_sequential() -> None:
    # agent_id set => managed governance is order/identity-sensitive, so the
    # loop must fall back to sequential even with parallel_tools=True.
    elapsed, _, _ = await _run(parallel_tools=True, agent_id="agent-123")
    assert elapsed >= 2 * _SLEEP * 0.9, f"expected sequential fallback, took {elapsed:.3f}s"
