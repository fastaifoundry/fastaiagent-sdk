"""Tests for middleware and durability support in the streaming tool loop."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fastaiagent.agent import Agent
from fastaiagent.agent.middleware import AgentMiddleware, MiddlewareContext
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import Message, ToolCall
from fastaiagent.llm.stream import TextDelta, ToolCallStart
from fastaiagent.tool.base import Tool
from fastaiagent.tool.function import FunctionTool
from tests.conftest import MockLLMClient


def _echo_tool() -> Tool:
    async def echo(text: str) -> str:
        return f"echoed:{text}"

    return FunctionTool(
        name="echo",
        fn=echo,
        description="Echo text back",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def _llm_with_one_tool_call_stream(
    tool_name: str = "echo", args: dict | None = None, final: str = "done"
) -> MockLLMClient:
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name=tool_name, arguments=args or {"text": "hi"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content=final, finish_reason="stop"),
        ]
    )


def _llm_with_n_tool_calls_stream(
    n: int, tool_name: str = "echo", final: str = "done"
) -> MockLLMClient:
    responses = []
    for i in range(n):
        responses.append(
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id=f"call_{i}", name=tool_name, arguments={"text": str(i)})
                ],
                finish_reason="tool_calls",
            )
        )
    responses.append(LLMResponse(content=final, finish_reason="stop"))
    return MockLLMClient(responses=responses)


# ---------------------------------------------------------------------------
# before_model fires during streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_before_model_fires_during_streaming():
    """Middleware before_model is called during the streaming tool loop."""
    hook_calls: list[int] = []

    class Recorder(AgentMiddleware):
        name = "recorder"

        async def before_model(self, ctx: MiddlewareContext, messages: list[Message]):
            hook_calls.append(ctx.turn)
            return messages

    agent = Agent(
        name="stream-mw",
        llm=_llm_with_one_tool_call_stream(),
        tools=[_echo_tool()],
        middleware=[Recorder()],
    )
    events = []
    async for event in agent.astream("hello", trace=False):
        events.append(event)

    # before_model should fire for iteration 0 (tool call turn) and
    # iteration 1 (final text turn)
    assert hook_calls == [0, 1]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "done"


# ---------------------------------------------------------------------------
# after_model fires during streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_model_fires_during_streaming():
    """Middleware after_model is called after each streamed response."""
    hook_calls: list[str | None] = []

    class Recorder(AgentMiddleware):
        name = "recorder"

        async def after_model(self, ctx, response):
            hook_calls.append(response.content)
            return response

    agent = Agent(
        name="stream-after",
        llm=_llm_with_one_tool_call_stream(final="final text"),
        tools=[_echo_tool()],
        middleware=[Recorder()],
    )
    events = []
    async for event in agent.astream("hello", trace=False):
        events.append(event)

    assert len(hook_calls) == 2
    # First call: tool call turn (no text content)
    assert hook_calls[0] is None
    # Second call: final text response
    assert hook_calls[1] == "final text"


# ---------------------------------------------------------------------------
# wrap_tool fires during streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wrap_tool_fires_during_streaming():
    """Middleware wrap_tool wraps tool calls during streaming."""
    wrap_calls: list[str] = []

    class ToolWrapper(AgentMiddleware):
        name = "wrapper"

        async def wrap_tool(self, ctx, tool, args, call_next):
            wrap_calls.append(tool.name)
            return await call_next(tool, args)

    agent = Agent(
        name="stream-wrap",
        llm=_llm_with_one_tool_call_stream(),
        tools=[_echo_tool()],
        middleware=[ToolWrapper()],
    )
    events = []
    async for event in agent.astream("hello", trace=False):
        events.append(event)

    assert wrap_calls == ["echo"]


# ---------------------------------------------------------------------------
# ToolBudget stops streaming after N tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_budget_stops_streaming():
    """ToolBudget middleware stops the streaming loop after max_calls."""
    from fastaiagent import ToolBudget

    agent = Agent(
        name="budgeted-stream",
        llm=_llm_with_n_tool_calls_stream(n=5, final="done"),
        tools=[_echo_tool()],
        middleware=[ToolBudget(max_calls=2, message="over budget")],
    )
    events = []
    async for event in agent.astream("run tools", trace=False):
        events.append(event)

    # The loop should stop after 2 tool calls — the generator returns
    # without yielding the "done" text.
    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert len(tool_starts) == 3  # calls 0, 1, 2 — budget fires on call 2
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    # No final text since budget stopped the run
    assert text == ""


# ---------------------------------------------------------------------------
# Checkpoint writes occur during streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_writes_during_streaming():
    """Checkpoints are written during the streaming tool loop."""
    mock_checkpointer = MagicMock()
    mock_checkpointer.setup = MagicMock()

    agent = Agent(
        name="ckpt-stream",
        llm=_llm_with_one_tool_call_stream(final="done"),
        tools=[_echo_tool()],
        checkpointer=mock_checkpointer,
    )
    events = []
    async for event in agent.astream("hello", trace=False):
        events.append(event)

    # Verify checkpointer.put was called:
    # - Turn checkpoint for iteration 0
    # - Tool checkpoint for tool call in iteration 0
    # - Turn checkpoint for iteration 1
    put_calls = mock_checkpointer.put.call_args_list
    assert len(put_calls) == 3

    # First checkpoint: turn:0
    ckpt0 = put_calls[0][0][0]
    assert ckpt0.node_id == "turn:0"
    assert ckpt0.status == "completed"

    # Second checkpoint: turn:0/tool:echo
    ckpt1 = put_calls[1][0][0]
    assert "tool:echo" in ckpt1.node_id
    assert ckpt1.status == "completed"

    # Third checkpoint: turn:1
    ckpt2 = put_calls[2][0][0]
    assert ckpt2.node_id == "turn:1"
    assert ckpt2.status == "completed"


# ---------------------------------------------------------------------------
# StopAgent in before_model stops streaming gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_agent_before_model_stops_streaming():
    """StopAgent raised in before_model stops the streaming generator."""
    from fastaiagent._internal.errors import StopAgent

    class Stopper(AgentMiddleware):
        name = "stopper"

        async def before_model(self, ctx, messages):
            raise StopAgent("stopped early")

    agent = Agent(
        name="stop-stream",
        llm=MockLLMClient(),
        middleware=[Stopper()],
    )
    events = []
    async for event in agent.astream("hello", trace=False):
        events.append(event)

    # No events should be yielded — the generator returns immediately
    assert events == []
