"""Tests for fastaiagent.agent module."""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import GuardrailBlockedError, MaxIterationsError
from fastaiagent.agent import Agent, AgentConfig, AgentMemory, AgentResult
from fastaiagent.guardrail import Guardrail, GuardrailPosition
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.tool import FunctionTool


class MockLLMClient(LLMClient):
    """Inline mock LLM for tests that need direct instantiation."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        super().__init__(provider="mock", model="mock-model")
        self._responses = responses or [
            LLMResponse(content="Hello! How can I help?", finish_reason="stop")
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


# --- Agent construction tests ---


class TestAgentConstruction:
    def test_default_agent(self):
        agent = Agent(name="test")
        assert agent.name == "test"
        assert agent.system_prompt == ""
        assert agent.tools == []
        assert agent.guardrails == []

    def test_agent_with_config(self):
        config = AgentConfig(max_iterations=5, temperature=0.7)
        agent = Agent(name="test", config=config)
        assert agent.config.max_iterations == 5
        assert agent.config.temperature == 0.7


# --- Agent execution tests ---


class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_simple_run(self, mock_llm):
        agent = Agent(name="test", system_prompt="Be helpful", llm=mock_llm)
        result = await agent.arun("Hello")
        assert isinstance(result, AgentResult)
        assert result.output == "Hello! How can I help?"
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_run_with_system_prompt(self, mock_llm):
        agent = Agent(name="test", system_prompt="You are a pirate", llm=mock_llm)
        await agent.arun("Hello")
        # Verify system prompt was sent
        call = mock_llm._calls[0]
        messages = call["messages"]
        assert messages[0].role.value == "system"
        assert messages[0].content == "You are a pirate"

    @pytest.mark.asyncio
    async def test_run_with_tools(self, mock_llm_with_tools):
        def search(query: str) -> str:
            return f"Results for: {query}"

        tool = FunctionTool(name="search", fn=search)
        agent = Agent(
            name="test",
            llm=mock_llm_with_tools,
            tools=[tool],
        )
        result = await agent.arun("Search for something")

        assert result.output == "Based on the search results, here is the answer."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool_name"] == "search"
        assert result.tool_calls[0]["output"] == "Results for: test"

    @pytest.mark.asyncio
    async def test_run_with_multiple_tool_iterations(self):
        """Test agent handles multiple rounds of tool calls."""
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[ToolCall(id="c1", name="step1", arguments={"x": "a"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(
                    tool_calls=[ToolCall(id="c2", name="step2", arguments={"x": "b"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Done after 2 tool calls", finish_reason="stop"),
            ]
        )

        tools = [
            FunctionTool(name="step1", fn=lambda x: f"step1:{x}"),
            FunctionTool(name="step2", fn=lambda x: f"step2:{x}"),
        ]
        agent = Agent(name="test", llm=llm, tools=tools)
        result = await agent.arun("Do two steps")

        assert result.output == "Done after 2 tool calls"
        assert len(result.tool_calls) == 2

    @pytest.mark.asyncio
    async def test_max_iterations_raises(self):
        """Agent raises MaxIterationsError when loop doesn't terminate."""
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[ToolCall(id="c1", name="loop", arguments={})],
                    finish_reason="tool_calls",
                ),
            ]
        )
        tool = FunctionTool(name="loop", fn=lambda: "again")
        agent = Agent(
            name="test",
            llm=llm,
            tools=[tool],
            config=AgentConfig(max_iterations=3),
        )

        with pytest.raises(MaxIterationsError, match="3"):
            await agent.arun("Loop forever")

    @pytest.mark.asyncio
    async def test_unknown_tool_handled(self):
        """Agent handles LLM calling a tool that doesn't exist."""
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[ToolCall(id="c1", name="nonexistent", arguments={})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Ok, that tool failed", finish_reason="stop"),
            ]
        )
        agent = Agent(name="test", llm=llm, tools=[])
        result = await agent.arun("Use a tool")
        assert "nonexistent" in result.tool_calls[0].get("error", "")


# --- Guardrail integration tests ---


class TestAgentGuardrails:
    @pytest.mark.asyncio
    async def test_input_guardrail_blocks(self, mock_llm):
        guardrail = Guardrail(
            name="block_bad",
            position=GuardrailPosition.input,
            blocking=True,
            fn=lambda text: "bad" not in text,
        )
        agent = Agent(name="test", llm=mock_llm, guardrails=[guardrail])

        with pytest.raises(GuardrailBlockedError, match="block_bad"):
            await agent.arun("This is bad input")

    @pytest.mark.asyncio
    async def test_output_guardrail_blocks(self):
        llm = MockLLMClient(
            responses=[LLMResponse(content="PII: 123-45-6789", finish_reason="stop")]
        )
        from fastaiagent.guardrail import no_pii

        agent = Agent(name="test", llm=llm, guardrails=[no_pii()])

        with pytest.raises(GuardrailBlockedError, match="PII detected"):
            await agent.arun("Show me data")

    @pytest.mark.asyncio
    async def test_guardrail_passes(self, mock_llm):
        guardrail = Guardrail(
            name="always_pass",
            position=GuardrailPosition.output,
            fn=lambda text: True,
        )
        agent = Agent(name="test", llm=mock_llm, guardrails=[guardrail])
        result = await agent.arun("Hello")
        assert result.output == "Hello! How can I help?"


# --- Memory tests ---


class TestAgentMemory:
    def test_memory_add_and_get(self):
        from fastaiagent.llm.message import UserMessage

        mem = AgentMemory()
        mem.add(UserMessage("msg1"))
        mem.add(UserMessage("msg2"))
        assert len(mem) == 2
        assert len(mem.get_context()) == 2

    def test_memory_max_messages(self):
        from fastaiagent.llm.message import UserMessage

        mem = AgentMemory(max_messages=2)
        mem.add(UserMessage("msg1"))
        mem.add(UserMessage("msg2"))
        mem.add(UserMessage("msg3"))
        assert len(mem) == 2
        assert mem.get_context()[0].content == "msg2"

    def test_memory_save_and_load(self, temp_dir):
        from fastaiagent.llm.message import AssistantMessage, UserMessage

        mem = AgentMemory()
        mem.add(UserMessage("Hello"))
        mem.add(AssistantMessage("Hi there"))
        path = temp_dir / "memory.json"
        mem.save(path)

        mem2 = AgentMemory()
        mem2.load(path)
        assert len(mem2) == 2
        assert mem2.messages[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_agent_stores_in_memory(self, mock_llm):
        mem = AgentMemory()
        agent = Agent(name="test", llm=mock_llm, memory=mem)
        await agent.arun("Hello")
        assert len(mem) == 2  # user + assistant
        assert mem.messages[0].content == "Hello"
        assert mem.messages[1].content == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_memory_included_in_messages(self, mock_llm):
        from fastaiagent.llm.message import AssistantMessage, UserMessage

        mem = AgentMemory()
        mem.add(UserMessage("Previous question"))
        mem.add(AssistantMessage("Previous answer"))

        agent = Agent(name="test", llm=mock_llm, memory=mem)
        await agent.arun("Follow up")

        # Check the messages sent to LLM include memory
        call = mock_llm._calls[0]
        messages = call["messages"]
        contents = [m.content for m in messages]
        assert "Previous question" in contents
        assert "Previous answer" in contents


# --- Serialization tests ---


class TestAgentSerialization:
    def test_to_dict(self):
        agent = Agent(
            name="test-agent",
            system_prompt="Be helpful",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            tools=[FunctionTool(name="greet", description="Greet")],
            config=AgentConfig(max_iterations=5),
        )
        d = agent.to_dict()
        assert d["name"] == "test-agent"
        assert d["agent_type"] == "single"
        assert d["system_prompt"] == "Be helpful"
        assert d["llm_endpoint"]["provider"] == "openai"
        assert len(d["tools"]) == 1
        assert d["config"]["max_iterations"] == 5

    def test_from_dict(self):
        data = {
            "name": "restored",
            "system_prompt": "Test prompt",
            "llm_endpoint": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            "tools": [
                {
                    "name": "search",
                    "tool_type": "function",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "config": {"max_iterations": 8},
        }
        agent = Agent.from_dict(data)
        assert agent.name == "restored"
        assert agent.llm.provider == "anthropic"
        assert len(agent.tools) == 1
        assert agent.config.max_iterations == 8

    def test_roundtrip(self):
        original = Agent(
            name="roundtrip",
            system_prompt="Test",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            config=AgentConfig(max_iterations=7),
        )
        d = original.to_dict()
        restored = Agent.from_dict(d)
        assert restored.name == original.name
        assert restored.system_prompt == original.system_prompt
        assert restored.config.max_iterations == original.config.max_iterations
