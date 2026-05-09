"""Tests for ``fastaiagent.testing.TestModel`` and ``FunctionModel``.

No mocking: ``TestModel``/``FunctionModel`` ARE the deterministic stand-
ins. These tests exercise their public surface end-to-end including
trace span emission.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent.agent import Agent
from fastaiagent.llm.message import MessageRole, ToolCall, UserMessage
from fastaiagent.llm.stream import TextDelta, ToolCallEnd
from fastaiagent.testing import FunctionModel, TestModel

# ---------------------------------------------------------------------------
# TestModel
# ---------------------------------------------------------------------------


def test_test_model_canned_text() -> None:
    tm = TestModel(response="hello")
    resp = tm.complete([UserMessage("anything")])
    assert resp.content == "hello"
    assert resp.finish_reason == "stop"
    assert resp.tool_calls == []


def test_test_model_round_robin() -> None:
    tm = TestModel(response=["one", "two", "three"])
    out = [tm.complete([UserMessage("x")]).content for _ in range(5)]
    # Last canned response repeats once the list is exhausted.
    assert out == ["one", "two", "three", "three", "three"]


def test_test_model_canned_tool_calls() -> None:
    tm = TestModel(
        response="",
        tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
    )
    resp = tm.complete([UserMessage("go")])
    assert resp.finish_reason == "tool_calls"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search"
    assert resp.tool_calls[0].arguments == {"q": "x"}


def test_test_model_records_calls() -> None:
    tm = TestModel(response="ok")
    tm.complete([UserMessage("first")])
    tm.complete([UserMessage("second")])
    assert len(tm.calls) == 2
    first_msg = tm.calls[0]["messages"][0]
    assert first_msg.role == MessageRole.user
    assert first_msg.content == "first"


def test_test_model_provider_is_test() -> None:
    tm = TestModel()
    assert tm.provider == "test"
    assert tm.api_key == "not-used"


def test_test_model_streaming_emits_text_delta() -> None:
    tm = TestModel(response="abc")

    async def collect() -> list[str]:
        events = []
        async for ev in tm.astream([UserMessage("x")]):
            events.append(type(ev).__name__)
        return events

    types = asyncio.run(collect())
    # TextDelta -> Usage -> StreamDone (no tool calls)
    assert "TextDelta" in types
    assert "Usage" in types
    assert types[-1] == "StreamDone"


def test_test_model_streaming_emits_tool_call_pair() -> None:
    tm = TestModel(
        response="",
        tool_calls=[{"name": "search", "arguments": {"q": "test"}}],
    )

    async def collect() -> list:
        events = []
        async for ev in tm.astream([UserMessage("go")]):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    types = [type(e).__name__ for e in events]
    assert "ToolCallStart" in types
    assert "ToolCallEnd" in types
    # The end should carry the canned arguments verbatim.
    end = next(e for e in events if isinstance(e, ToolCallEnd))
    assert end.arguments == {"q": "test"}


def test_test_model_in_agent_run() -> None:
    """End-to-end: TestModel powers a real Agent.run() with no network."""
    agent = Agent(name="hello-bot", llm=TestModel(response="hi"))
    result = agent.run("anything")
    assert result.output == "hi"


def test_test_model_usage_passes_through() -> None:
    tm = TestModel(response="ok", usage=(10, 5))
    resp = tm.complete([UserMessage("x")])
    assert resp.usage["prompt_tokens"] == 10
    assert resp.usage["completion_tokens"] == 5
    assert resp.usage["total_tokens"] == 15


# ---------------------------------------------------------------------------
# FunctionModel
# ---------------------------------------------------------------------------


def test_function_model_simple_text_responder() -> None:
    fm = FunctionModel(lambda messages: "answer")
    resp = fm.complete([UserMessage("hello")])
    assert resp.content == "answer"
    assert resp.tool_calls == []
    assert resp.finish_reason == "stop"


def test_function_model_tool_call_responder() -> None:
    fm = FunctionModel(
        lambda messages: ("", [{"name": "lookup", "arguments": {"id": 1}}])
    )
    resp = fm.complete([UserMessage("go")])
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0].name == "lookup"


def test_function_model_state_machine() -> None:
    """Two-turn flow: first call asks tool, second returns final."""
    state = {"calls": 0}

    def responder(messages):
        state["calls"] += 1
        if state["calls"] == 1:
            return ("", [{"name": "search", "arguments": {"q": "x"}}])
        return ("done", [])

    fm = FunctionModel(responder)
    r1 = fm.complete([UserMessage("ask")])
    r2 = fm.complete([UserMessage("answer")])
    assert r1.tool_calls and r1.tool_calls[0].name == "search"
    assert r2.content == "done"


def test_function_model_returns_tool_call_objects_directly() -> None:
    """Responder may return ToolCall instances instead of dicts."""
    tc = ToolCall(id="c-7", name="bare", arguments={})
    fm = FunctionModel(lambda messages: ("", [tc]))
    resp = fm.complete([UserMessage("x")])
    assert resp.tool_calls[0].id == "c-7"
    assert resp.tool_calls[0].name == "bare"


def test_function_model_async_responder() -> None:
    """Async responders are awaited automatically."""

    async def responder(messages):
        return "async-answer"

    fm = FunctionModel(responder)
    resp = fm.complete([UserMessage("hi")])
    assert resp.content == "async-answer"


def test_function_model_invalid_return_raises() -> None:
    fm = FunctionModel(lambda messages: 42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return"):
        fm.complete([UserMessage("x")])


def test_function_model_in_agent_with_tools() -> None:
    """Two-turn agent loop: tool-call then final answer."""
    state = {"calls": 0}

    def responder(messages):
        state["calls"] += 1
        if state["calls"] == 1:
            return ("", [{"name": "echo", "arguments": {"text": "boo"}}])
        return ("got: boo", [])

    from fastaiagent.tool import tool

    @tool()
    def echo(text: str) -> str:
        """Echo text back."""
        return text

    agent = Agent(name="echo-bot", llm=FunctionModel(responder), tools=[echo])
    result = agent.run("say it")
    assert result.output == "got: boo"
    assert state["calls"] == 2  # exactly two LLM iterations


# ---------------------------------------------------------------------------
# Smoke: stream events plumbed through agent
# ---------------------------------------------------------------------------


def test_test_model_astream_via_agent_streams_text() -> None:
    """Agent.astream forwards TextDelta from a TestModel-backed agent."""
    agent = Agent(name="streamer", llm=TestModel(response="streamed-ok"))

    async def collect() -> str:
        text_chunks: list[str] = []
        async for ev in agent.astream("anything"):
            if isinstance(ev, TextDelta):
                text_chunks.append(ev.text)
        return "".join(text_chunks)

    out = asyncio.run(collect())
    assert "streamed-ok" in out
