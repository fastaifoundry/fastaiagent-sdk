"""Tests for streaming support across LLMClient, executor, and Agent."""

from __future__ import annotations

import pytest

from fastaiagent.agent import Agent, AgentConfig
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import (
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)
from fastaiagent.tool import FunctionTool


# --- StreamEvent type tests ---


class TestStreamEventTypes:
    def test_text_delta(self):
        event = TextDelta(text="Hello")
        assert event.text == "Hello"

    def test_tool_call_start(self):
        event = ToolCallStart(call_id="c1", tool_name="search")
        assert event.call_id == "c1"
        assert event.tool_name == "search"

    def test_tool_call_end(self):
        event = ToolCallEnd(call_id="c1", tool_name="search", arguments={"q": "test"})
        assert event.arguments == {"q": "test"}

    def test_usage(self):
        event = Usage(prompt_tokens=10, completion_tokens=5)
        assert event.prompt_tokens == 10
        assert event.completion_tokens == 5

    def test_stream_done(self):
        event = StreamDone()
        assert isinstance(event, StreamDone)

    def test_usage_defaults(self):
        event = Usage()
        assert event.prompt_tokens == 0
        assert event.completion_tokens == 0

    def test_tool_call_end_default_args(self):
        event = ToolCallEnd(call_id="c1", tool_name="search")
        assert event.arguments == {}


# --- MockStreamingLLMClient ---


class MockStreamingLLMClient(LLMClient):
    """Mock LLM client that supports both acomplete() and astream()."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        stream_events: list[list[StreamEvent]] | None = None,
    ):
        super().__init__(provider="mock", model="mock-model")
        self._responses = responses or [
            LLMResponse(content="Hello! How can I help?", finish_reason="stop")
        ]
        self._stream_events = stream_events
        self._call_count = 0
        self._stream_call_count = 0

    async def acomplete(self, messages, tools=None, **kwargs):
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = self._responses[-1]
        self._call_count += 1
        return response

    async def astream(self, messages, tools=None, **kwargs):
        if self._stream_events is None:
            # Fallback: convert responses to stream events
            resp = self._responses[min(self._stream_call_count, len(self._responses) - 1)]
            self._stream_call_count += 1
            if resp.content:
                for char in resp.content:
                    yield TextDelta(text=char)
            for tc in resp.tool_calls:
                yield ToolCallStart(call_id=tc.id, tool_name=tc.name)
                yield ToolCallEnd(call_id=tc.id, tool_name=tc.name, arguments=tc.arguments)
            yield Usage(prompt_tokens=10, completion_tokens=5)
            yield StreamDone()
        else:
            events = self._stream_events[
                min(self._stream_call_count, len(self._stream_events) - 1)
            ]
            self._stream_call_count += 1
            for event in events:
                yield event


# --- LLMClient.stream() tests ---


class TestLLMClientStream:
    def test_stream_sync_returns_llm_response(self):
        """stream() collects events into a single LLMResponse."""
        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    TextDelta(text="Hello "),
                    TextDelta(text="world!"),
                    Usage(prompt_tokens=5, completion_tokens=3),
                    StreamDone(),
                ]
            ]
        )
        result = llm.stream([])
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello world!"
        assert result.usage["prompt_tokens"] == 5
        assert result.finish_reason == "stop"

    def test_stream_sync_with_tool_calls(self):
        """stream() collects tool calls from stream events."""
        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    ToolCallStart(call_id="c1", tool_name="search"),
                    ToolCallEnd(
                        call_id="c1", tool_name="search", arguments={"q": "weather"}
                    ),
                    Usage(prompt_tokens=10, completion_tokens=8),
                    StreamDone(),
                ]
            ]
        )
        result = llm.stream([])
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"q": "weather"}
        assert result.finish_reason == "tool_calls"


# --- LLMClient.astream() tests ---


class TestLLMClientAstream:
    @pytest.mark.asyncio
    async def test_astream_yields_text_deltas(self):
        llm = MockStreamingLLMClient(
            stream_events=[
                [TextDelta(text="Hi"), TextDelta(text="!"), StreamDone()]
            ]
        )
        events = []
        async for event in llm.astream([]):
            events.append(event)
        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_events) == 2
        assert text_events[0].text == "Hi"
        assert text_events[1].text == "!"

    @pytest.mark.asyncio
    async def test_astream_yields_tool_events(self):
        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    ToolCallStart(call_id="c1", tool_name="greet"),
                    ToolCallEnd(call_id="c1", tool_name="greet", arguments={"name": "World"}),
                    StreamDone(),
                ]
            ]
        )
        events = []
        async for event in llm.astream([]):
            events.append(event)
        starts = [e for e in events if isinstance(e, ToolCallStart)]
        ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(starts) == 1
        assert len(ends) == 1
        assert ends[0].arguments == {"name": "World"}

    @pytest.mark.asyncio
    async def test_astream_yields_usage(self):
        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    TextDelta(text="test"),
                    Usage(prompt_tokens=15, completion_tokens=7),
                    StreamDone(),
                ]
            ]
        )
        events = []
        async for event in llm.astream([]):
            events.append(event)
        usage_events = [e for e in events if isinstance(e, Usage)]
        assert len(usage_events) == 1
        assert usage_events[0].prompt_tokens == 15

    @pytest.mark.asyncio
    async def test_astream_unsupported_provider_raises(self):
        llm = LLMClient(provider="bedrock", model="test")
        with pytest.raises(Exception, match="Streaming not supported"):
            async for _ in llm.astream([]):
                pass


# --- stream_tool_loop() tests ---


class TestStreamToolLoop:
    @pytest.mark.asyncio
    async def test_simple_stream_no_tools(self):
        """Stream a simple response with no tool calls."""
        from fastaiagent.agent.executor import stream_tool_loop
        from fastaiagent.llm.message import UserMessage

        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    TextDelta(text="Hello"),
                    TextDelta(text=" world"),
                    Usage(prompt_tokens=5, completion_tokens=3),
                    StreamDone(),
                ]
            ]
        )
        events = []
        async for event in stream_tool_loop(
            llm=llm, messages=[UserMessage("Hi")], tools=[]
        ):
            events.append(event)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self):
        """Stream with one tool call iteration then final response."""
        from fastaiagent.agent.executor import stream_tool_loop
        from fastaiagent.llm.message import UserMessage

        def search(query: str) -> str:
            return f"Results for: {query}"

        tool = FunctionTool(name="search", fn=search)

        llm = MockStreamingLLMClient(
            stream_events=[
                # Iteration 1: LLM requests tool call
                [
                    ToolCallStart(call_id="c1", tool_name="search"),
                    ToolCallEnd(
                        call_id="c1", tool_name="search", arguments={"query": "weather"}
                    ),
                    Usage(prompt_tokens=10, completion_tokens=5),
                    StreamDone(),
                ],
                # Iteration 2: LLM returns final text
                [
                    TextDelta(text="The weather is sunny."),
                    Usage(prompt_tokens=20, completion_tokens=8),
                    StreamDone(),
                ],
            ]
        )
        events = []
        async for event in stream_tool_loop(
            llm=llm, messages=[UserMessage("Weather?")], tools=[tool]
        ):
            events.append(event)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "The weather is sunny."
        starts = [e for e in events if isinstance(e, ToolCallStart)]
        ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(starts) == 1
        assert len(ends) == 1
        assert starts[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_stream_max_iterations_raises(self):
        """Stream raises MaxIterationsError when loop doesn't terminate."""
        from fastaiagent._internal.errors import MaxIterationsError
        from fastaiagent.agent.executor import stream_tool_loop
        from fastaiagent.llm.message import UserMessage

        tool = FunctionTool(name="loop", fn=lambda: "again")

        # Always returns a tool call
        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    ToolCallStart(call_id="c1", tool_name="loop"),
                    ToolCallEnd(call_id="c1", tool_name="loop", arguments={}),
                    StreamDone(),
                ]
            ]
        )

        with pytest.raises(MaxIterationsError, match="2"):
            events = []
            async for event in stream_tool_loop(
                llm=llm,
                messages=[UserMessage("Loop")],
                tools=[tool],
                max_iterations=2,
            ):
                events.append(event)


# --- Agent.astream() tests ---


class TestAgentAstream:
    @pytest.mark.asyncio
    async def test_simple_astream(self):
        """Agent.astream() yields TextDelta events."""
        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    TextDelta(text="Hello"),
                    TextDelta(text="!"),
                    Usage(prompt_tokens=5, completion_tokens=2),
                    StreamDone(),
                ]
            ]
        )
        agent = Agent(name="test", system_prompt="Be helpful", llm=llm)
        events = []
        async for event in agent.astream("Hi"):
            events.append(event)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Hello!"

    @pytest.mark.asyncio
    async def test_astream_with_tools(self):
        """Agent.astream() handles tool calls during streaming."""

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        tool = FunctionTool(name="greet", fn=greet)
        llm = MockStreamingLLMClient(
            stream_events=[
                # Tool call
                [
                    ToolCallStart(call_id="c1", tool_name="greet"),
                    ToolCallEnd(
                        call_id="c1", tool_name="greet", arguments={"name": "Alice"}
                    ),
                    StreamDone(),
                ],
                # Final text
                [
                    TextDelta(text="I greeted Alice."),
                    StreamDone(),
                ],
            ]
        )
        agent = Agent(name="test", llm=llm, tools=[tool])
        events = []
        async for event in agent.astream("Greet Alice"):
            events.append(event)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "I greeted Alice."

    @pytest.mark.asyncio
    async def test_astream_input_guardrail_blocks(self):
        """Agent.astream() runs input guardrails before streaming."""
        from fastaiagent._internal.errors import GuardrailBlockedError
        from fastaiagent.guardrail import Guardrail, GuardrailPosition

        llm = MockStreamingLLMClient()
        guardrail = Guardrail(
            name="block_bad",
            position=GuardrailPosition.input,
            blocking=True,
            fn=lambda text: "bad" not in text,
        )
        agent = Agent(name="test", llm=llm, guardrails=[guardrail])

        with pytest.raises(GuardrailBlockedError, match="block_bad"):
            async for _ in agent.astream("This is bad input"):
                pass

    @pytest.mark.asyncio
    async def test_astream_output_guardrail_blocks(self):
        """Agent.astream() runs output guardrails after streaming completes."""
        from fastaiagent._internal.errors import GuardrailBlockedError
        from fastaiagent.guardrail import Guardrail, GuardrailPosition

        llm = MockStreamingLLMClient(
            stream_events=[
                [TextDelta(text="PII: 123-45-6789"), StreamDone()]
            ]
        )
        from fastaiagent.guardrail import no_pii

        agent = Agent(name="test", llm=llm, guardrails=[no_pii()])

        with pytest.raises(GuardrailBlockedError, match="PII detected"):
            async for _ in agent.astream("Show me data"):
                pass

    @pytest.mark.asyncio
    async def test_astream_stores_in_memory(self):
        """Agent.astream() stores user/assistant messages in memory."""
        from fastaiagent.agent import AgentMemory

        llm = MockStreamingLLMClient(
            stream_events=[
                [TextDelta(text="Hi there!"), StreamDone()]
            ]
        )
        mem = AgentMemory()
        agent = Agent(name="test", llm=llm, memory=mem)

        async for _ in agent.astream("Hello"):
            pass

        assert len(mem) == 2
        assert mem.messages[0].content == "Hello"
        assert mem.messages[1].content == "Hi there!"


# --- Agent.stream() sync wrapper tests ---


class TestAgentStreamSync:
    def test_stream_sync_returns_agent_result(self):
        """Agent.stream() collects streaming into AgentResult."""
        llm = MockStreamingLLMClient(
            stream_events=[
                [
                    TextDelta(text="Hello "),
                    TextDelta(text="world!"),
                    StreamDone(),
                ]
            ]
        )
        agent = Agent(name="test", llm=llm)
        result = agent.stream("Hi")
        assert result.output == "Hello world!"
        assert result.latency_ms >= 0
