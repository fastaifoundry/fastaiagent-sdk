"""Tests for Dynamic Instructions — callable system_prompt support."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fastaiagent import RunContext
from fastaiagent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import StreamEvent, TextDelta
from fastaiagent.tool import FunctionTool, tool


# --- Fixtures ---


@dataclass
class UserState:
    user_name: str
    plan_tier: str


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


# --- _resolve_system_prompt unit tests ---


class TestResolveSystemPrompt:
    def test_static_string_returned_as_is(self):
        agent = Agent(name="test", system_prompt="You are helpful.")
        assert agent._resolve_system_prompt() == "You are helpful."

    def test_callable_invoked_with_context(self):
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: f"Help {ctx.state.user_name}",
        )
        ctx = RunContext(state=UserState(user_name="Alice", plan_tier="pro"))
        assert agent._resolve_system_prompt(ctx) == "Help Alice"

    def test_callable_receives_none_when_no_context(self):
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: "Default" if ctx is None else "Custom",
        )
        assert agent._resolve_system_prompt(None) == "Default"
        assert agent._resolve_system_prompt() == "Default"

    def test_callable_that_doesnt_handle_none_raises(self):
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: f"Help {ctx.state.user_name}",
        )
        with pytest.raises(AttributeError):
            agent._resolve_system_prompt(None)

    def test_empty_string_prompt(self):
        agent = Agent(name="test", system_prompt="")
        assert agent._resolve_system_prompt() == ""


# --- _build_messages tests ---


class TestBuildMessagesWithContext:
    def test_static_prompt_build_messages(self):
        agent = Agent(name="test", system_prompt="You are helpful.")
        messages = agent._build_messages("Hello")
        assert len(messages) == 2
        assert messages[0].content == "You are helpful."
        assert messages[1].content == "Hello"

    def test_callable_prompt_build_messages_with_context(self):
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: f"Help {ctx.state['name']}",
        )
        ctx = RunContext(state={"name": "Alice"})
        messages = agent._build_messages("Hello", context=ctx)
        assert len(messages) == 2
        assert messages[0].content == "Help Alice"

    def test_callable_prompt_build_messages_without_context(self):
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: "Default" if ctx is None else "Custom",
        )
        messages = agent._build_messages("Hello", context=None)
        assert messages[0].content == "Default"

    def test_empty_string_no_system_message(self):
        agent = Agent(name="test", system_prompt="")
        messages = agent._build_messages("Hello")
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    def test_callable_returning_empty_string_no_system_message(self):
        agent = Agent(name="test", system_prompt=lambda ctx: "")
        messages = agent._build_messages("Hello")
        assert len(messages) == 1
        assert messages[0].content == "Hello"


# --- Agent.arun() with dynamic prompt ---


class TestAgentRunWithDynamicPrompt:
    @pytest.mark.asyncio
    async def test_arun_with_callable_prompt_and_context(self):
        llm = MockLLMClient()
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: f"Help {ctx.state.user_name} ({ctx.state.plan_tier})",
            llm=llm,
        )
        ctx = RunContext(state=UserState(user_name="Alice", plan_tier="pro"))
        result = await agent.arun("Hello", context=ctx)

        assert result.output == "Hello!"
        # Verify the system message sent to LLM was resolved
        sent_messages = llm._calls[0]["messages"]
        assert sent_messages[0].content == "Help Alice (pro)"

    @pytest.mark.asyncio
    async def test_arun_with_callable_prompt_no_context(self):
        llm = MockLLMClient()
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: "Fallback prompt" if ctx is None else "Custom",
            llm=llm,
        )
        result = await agent.arun("Hello")

        assert result.output == "Hello!"
        sent_messages = llm._calls[0]["messages"]
        assert sent_messages[0].content == "Fallback prompt"

    @pytest.mark.asyncio
    async def test_arun_static_prompt_backward_compatible(self):
        llm = MockLLMClient()
        agent = Agent(name="test", system_prompt="You are helpful.", llm=llm)
        result = await agent.arun("Hello")

        assert result.output == "Hello!"
        sent_messages = llm._calls[0]["messages"]
        assert sent_messages[0].content == "You are helpful."


# --- Agent.astream() with dynamic prompt ---


class TestAgentStreamWithDynamicPrompt:
    @pytest.mark.asyncio
    async def test_astream_with_callable_prompt(self):
        llm = MockLLMClient()
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: f"Help {ctx.state.user_name}",
            llm=llm,
        )
        ctx = RunContext(state=UserState(user_name="Bob", plan_tier="free"))

        collected = []
        async for event in agent.astream("Hello", context=ctx):
            if isinstance(event, TextDelta):
                collected.append(event.text)

        assert "".join(collected) == "Hello!"
        sent_messages = llm._calls[0]["messages"]
        assert sent_messages[0].content == "Help Bob"


# --- Full integration: callable prompt + context tools ---


class TestCallablePromptWithContextTools:
    @pytest.mark.asyncio
    async def test_callable_prompt_and_context_tool(self):
        """Both prompt and tool receive the same RunContext."""
        received_ctx = {}

        @tool(name="get_info")
        def get_info(ctx: RunContext[UserState], key: str) -> str:
            received_ctx["state"] = ctx.state
            return f"{key}={ctx.state.user_name}"

        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="get_info", arguments={"key": "name"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Got it.", finish_reason="stop"),
            ]
        )

        agent = Agent(
            name="test",
            system_prompt=lambda ctx: f"You help {ctx.state.user_name}",
            llm=llm,
            tools=[get_info],
        )
        ctx = RunContext(state=UserState(user_name="Alice", plan_tier="pro"))
        result = await agent.arun("Get name", context=ctx)

        # Verify prompt was resolved with context
        sent_messages = llm._calls[0]["messages"]
        assert sent_messages[0].content == "You help Alice"

        # Verify tool also received context
        assert received_ctx["state"].user_name == "Alice"
        assert result.output == "Got it."


# --- Serialization tests ---


class TestToDictWithDynamicPrompt:
    def test_to_dict_raises_on_callable(self):
        agent = Agent(
            name="test",
            system_prompt=lambda ctx: "dynamic",
        )
        with pytest.raises(ValueError, match="callable system_prompt"):
            agent.to_dict()

    def test_to_dict_works_with_static_string(self):
        agent = Agent(
            name="test",
            system_prompt="You are helpful.",
            llm=LLMClient(provider="openai", model="gpt-4o"),
        )
        d = agent.to_dict()
        assert d["system_prompt"] == "You are helpful."

    def test_to_dict_error_message_includes_agent_name(self):
        agent = Agent(
            name="my-agent",
            system_prompt=lambda ctx: "dynamic",
        )
        with pytest.raises(ValueError, match="my-agent"):
            agent.to_dict()
