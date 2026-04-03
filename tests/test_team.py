"""Tests for Supervisor/Worker — context passthrough, streaming, dynamic instructions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fastaiagent import RunContext
from fastaiagent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import StreamDone, StreamEvent, TextDelta, ToolCallEnd, ToolCallStart, Usage
from fastaiagent.tool import FunctionTool, tool


# --- Fixtures ---


@dataclass
class TeamState:
    project: str
    user_id: str


class MockLLMClient(LLMClient):
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        super().__init__(provider="mock", model="mock-model")
        self._responses = responses or [
            LLMResponse(content="Hello!", finish_reason="stop")
        ]
        self._call_count = 0
        self._calls: list[dict] = []

    async def acomplete(self, messages, tools=None, **kwargs):
        self._calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = self._responses[-1]
        self._call_count += 1
        return response

    async def astream(self, messages, tools=None, **kwargs):
        response = await self.acomplete(messages, tools=tools, **kwargs)
        if response.content:
            yield TextDelta(text=response.content)
        for tc in response.tool_calls:
            yield ToolCallStart(call_id=tc.id, tool_name=tc.name)
            yield ToolCallEnd(call_id=tc.id, tool_name=tc.name, arguments=tc.arguments)
        yield Usage(prompt_tokens=10, completion_tokens=5)
        yield StreamDone()


class MockStreamingLLMClient(LLMClient):
    """Mock LLM with explicit stream event sequences."""

    def __init__(
        self,
        responses: list[LLMResponse] | None = None,
        stream_events: list[list[StreamEvent]] | None = None,
    ):
        super().__init__(provider="mock", model="mock-model")
        self._responses = responses or [
            LLMResponse(content="Hello!", finish_reason="stop")
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


# --- Worker tests ---


class TestWorker:
    def test_explicit_description(self):
        from fastaiagent.agent.team import Worker

        agent = Agent(name="test", system_prompt="You are a tester.")
        worker = Worker(agent=agent, role="tester", description="Runs tests")
        assert worker.role == "tester"
        assert worker.description == "Runs tests"

    def test_fallback_to_system_prompt(self):
        from fastaiagent.agent.team import Worker

        prompt = "You research topics thoroughly and return detailed results."
        agent = Agent(name="researcher", system_prompt=prompt)
        worker = Worker(agent=agent, role="researcher")
        assert worker.description == prompt

    def test_fallback_truncates_long_prompt(self):
        from fastaiagent.agent.team import Worker

        prompt = "A" * 300
        agent = Agent(name="verbose", system_prompt=prompt)
        worker = Worker(agent=agent, role="verbose")
        assert len(worker.description) == 200

    def test_callable_prompt_no_crash(self):
        from fastaiagent.agent.team import Worker

        agent = Agent(name="dynamic", system_prompt=lambda ctx: "Dynamic prompt")
        worker = Worker(agent=agent, role="dynamic")
        assert worker.description == ""

    def test_role_defaults_to_agent_name(self):
        from fastaiagent.agent.team import Worker

        agent = Agent(name="my-agent", system_prompt="Be helpful.")
        worker = Worker(agent=agent)
        assert worker.role == "my-agent"


# --- Supervisor construction tests ---


class TestSupervisorConstruction:
    def test_auto_generated_prompt(self):
        from fastaiagent.agent.team import Supervisor, Worker

        agent = Agent(name="worker1", system_prompt="Does things.")
        worker = Worker(agent=agent, role="helper", description="Helps out")
        sup = Supervisor(name="lead", workers=[worker])

        assert "helper" in sup.system_prompt
        assert "Helps out" in sup.system_prompt
        assert isinstance(sup.system_prompt, str)

    def test_custom_string_prompt(self):
        from fastaiagent.agent.team import Supervisor

        sup = Supervisor(name="lead", system_prompt="Custom instructions.")
        assert sup.system_prompt == "Custom instructions."

    def test_callable_prompt_stored(self):
        from fastaiagent.agent.team import Supervisor

        fn = lambda ctx: f"Dynamic for {ctx.state['name']}"
        sup = Supervisor(name="lead", system_prompt=fn)
        assert sup.system_prompt is fn


# --- Delegation round-trip ---


class TestSupervisorDelegation:
    @pytest.mark.asyncio
    async def test_delegation_round_trip(self):
        """Supervisor delegates to worker, worker responds, supervisor synthesizes."""
        from fastaiagent.agent.team import Supervisor, Worker

        # Worker LLM: just returns a direct answer
        worker_llm = MockLLMClient(
            responses=[LLMResponse(content="Worker result: found 42 items.", finish_reason="stop")]
        )
        worker_agent = Agent(name="searcher", system_prompt="Search for things.", llm=worker_llm)
        worker = Worker(agent=worker_agent, role="searcher", description="Searches for data")

        # Supervisor LLM: first call delegates, second call synthesizes
        supervisor_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="c1", name="delegate_to_searcher", arguments={"task": "find items"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="The search found 42 items.", finish_reason="stop"),
            ]
        )

        sup = Supervisor(name="lead", llm=supervisor_llm, workers=[worker])
        result = await sup.arun("How many items are there?")

        assert result.output == "The search found 42 items."

    @pytest.mark.asyncio
    async def test_context_flows_to_worker_tools(self):
        """RunContext passes through supervisor → worker → worker's tools."""
        from fastaiagent.agent.team import Supervisor, Worker

        received_ctx = {}

        @tool(name="get_data")
        def get_data(ctx: RunContext[TeamState], key: str) -> str:
            received_ctx["state"] = ctx.state
            return f"data for {key} by {ctx.state.user_id}"

        # Worker LLM: calls get_data tool, then returns final answer
        worker_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="wc1", name="get_data", arguments={"key": "orders"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Got the data.", finish_reason="stop"),
            ]
        )
        worker_agent = Agent(name="data-agent", system_prompt="Fetch data.", llm=worker_llm, tools=[get_data])
        worker = Worker(agent=worker_agent, role="data", description="Fetches data")

        # Supervisor LLM: delegates to data worker, then synthesizes
        supervisor_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="sc1", name="delegate_to_data", arguments={"task": "get orders"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Here are your orders.", finish_reason="stop"),
            ]
        )

        sup = Supervisor(name="lead", llm=supervisor_llm, workers=[worker])
        ctx = RunContext(state=TeamState(project="acme", user_id="u-789"))
        result = await sup.arun("Show my orders", context=ctx)

        assert result.output == "Here are your orders."
        assert received_ctx["state"].user_id == "u-789"
        assert received_ctx["state"].project == "acme"

    @pytest.mark.asyncio
    async def test_no_context_backward_compatible(self):
        """Supervisor works without context — backward compatible."""
        from fastaiagent.agent.team import Supervisor, Worker

        @tool(name="search")
        def search(query: str) -> str:
            return f"Results for: {query}"

        worker_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="wc1", name="search", arguments={"query": "test"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Found results.", finish_reason="stop"),
            ]
        )
        worker_agent = Agent(name="searcher", system_prompt="Search.", llm=worker_llm, tools=[search])
        worker = Worker(agent=worker_agent, role="searcher", description="Searches")

        supervisor_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="sc1", name="delegate_to_searcher", arguments={"task": "search test"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Done.", finish_reason="stop"),
            ]
        )
        sup = Supervisor(name="lead", llm=supervisor_llm, workers=[worker])
        result = await sup.arun("Search for test")
        assert result.output == "Done."


# --- Streaming tests ---


class TestSupervisorStreaming:
    @pytest.mark.asyncio
    async def test_astream_yields_events(self):
        """Supervisor.astream() yields StreamEvent objects."""
        from fastaiagent.agent.team import Supervisor, Worker

        worker_llm = MockLLMClient()
        worker_agent = Agent(name="helper", system_prompt="Help.", llm=worker_llm)
        worker = Worker(agent=worker_agent, role="helper", description="Helps")

        supervisor_llm = MockStreamingLLMClient(
            stream_events=[
                [
                    TextDelta(text="Hello "),
                    TextDelta(text="world!"),
                    Usage(prompt_tokens=5, completion_tokens=3),
                    StreamDone(),
                ]
            ]
        )

        sup = Supervisor(name="lead", llm=supervisor_llm, workers=[worker])
        events = []
        async for event in sup.astream("Hi"):
            events.append(event)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Hello world!"

    def test_stream_returns_agent_result(self):
        """Supervisor.stream() collects events into AgentResult."""
        from fastaiagent.agent.team import Supervisor, Worker

        worker_llm = MockLLMClient()
        worker_agent = Agent(name="helper", system_prompt="Help.", llm=worker_llm)
        worker = Worker(agent=worker_agent, role="helper", description="Helps")

        supervisor_llm = MockStreamingLLMClient(
            stream_events=[
                [
                    TextDelta(text="Collected "),
                    TextDelta(text="output."),
                    Usage(prompt_tokens=5, completion_tokens=3),
                    StreamDone(),
                ]
            ]
        )

        sup = Supervisor(name="lead", llm=supervisor_llm, workers=[worker])
        result = sup.stream("Hi")

        assert isinstance(result, AgentResult)
        assert result.output == "Collected output."
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_astream_with_context(self):
        """Supervisor.astream() forwards context to workers."""
        from fastaiagent.agent.team import Supervisor, Worker

        received_ctx = {}

        @tool(name="check")
        def check(ctx: RunContext[TeamState], item: str) -> str:
            received_ctx["state"] = ctx.state
            return f"checked {item}"

        # Worker: calls tool then returns
        worker_llm = MockStreamingLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="wc1", name="check", arguments={"item": "order"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Order checked.", finish_reason="stop"),
            ]
        )
        worker_agent = Agent(name="checker", system_prompt="Check items.", llm=worker_llm, tools=[check])
        worker = Worker(agent=worker_agent, role="checker", description="Checks items")

        # Supervisor: delegates then synthesizes
        supervisor_llm = MockStreamingLLMClient(
            stream_events=[
                # Delegation
                [
                    ToolCallStart(call_id="sc1", tool_name="delegate_to_checker"),
                    ToolCallEnd(
                        call_id="sc1",
                        tool_name="delegate_to_checker",
                        arguments={"task": "check order"},
                    ),
                    StreamDone(),
                ],
                # Final synthesis
                [
                    TextDelta(text="Order is good."),
                    StreamDone(),
                ],
            ]
        )

        sup = Supervisor(name="lead", llm=supervisor_llm, workers=[worker])
        ctx = RunContext(state=TeamState(project="acme", user_id="u-1"))

        events = []
        async for event in sup.astream("Check my order", context=ctx):
            events.append(event)

        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "Order is good."
        assert received_ctx["state"].user_id == "u-1"


# --- Dynamic Instructions tests ---


class TestSupervisorDynamicInstructions:
    @pytest.mark.asyncio
    async def test_callable_prompt_resolves_with_context(self):
        """Callable system_prompt receives RunContext and resolves correctly."""
        from fastaiagent.agent.team import Supervisor, Worker

        worker_llm = MockLLMClient()
        worker_agent = Agent(name="helper", system_prompt="Help.", llm=worker_llm)
        worker = Worker(agent=worker_agent, role="helper", description="Helps")

        supervisor_llm = MockLLMClient()
        sup = Supervisor(
            name="dynamic-lead",
            llm=supervisor_llm,
            workers=[worker],
            system_prompt=lambda ctx: (
                f"You manage support for {ctx.state.project}. "
                f"Available workers:\n- helper: Helps"
            ),
        )

        ctx = RunContext(state=TeamState(project="Acme Corp", user_id="u-1"))
        await sup.arun("Help me", context=ctx)

        # Verify the resolved system message was sent to LLM
        sent_messages = supervisor_llm._calls[0]["messages"]
        assert "Acme Corp" in sent_messages[0].content

    @pytest.mark.asyncio
    async def test_callable_prompt_without_context_receives_none(self):
        """Callable prompt called without context receives None."""
        from fastaiagent.agent.team import Supervisor, Worker

        worker_llm = MockLLMClient()
        worker_agent = Agent(name="helper", system_prompt="Help.", llm=worker_llm)
        worker = Worker(agent=worker_agent, role="helper", description="Helps")

        supervisor_llm = MockLLMClient()
        sup = Supervisor(
            name="safe-lead",
            llm=supervisor_llm,
            workers=[worker],
            system_prompt=lambda ctx: (
                f"Support for {ctx.state.project}."
                if ctx else
                "You are a team supervisor."
            ),
        )

        await sup.arun("Help me")

        sent_messages = supervisor_llm._calls[0]["messages"]
        assert sent_messages[0].content == "You are a team supervisor."


# --- Top-level import test ---


class TestTopLevelImport:
    def test_import_supervisor_and_worker(self):
        """Supervisor and Worker are importable from the top-level package."""
        from fastaiagent import Supervisor, Worker

        assert Supervisor is not None
        assert Worker is not None
