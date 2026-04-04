"""Tests for tool-position guardrail wiring (tool_call and tool_result positions)."""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.agent import Agent
from fastaiagent.guardrail import Guardrail, GuardrailPosition
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import StreamDone, TextDelta, ToolCallEnd, ToolCallStart, Usage
from fastaiagent.tool import FunctionTool

# --- Mock LLM ---


class MockLLMClient(LLMClient):
    def __init__(self, responses: list[LLMResponse] | None = None):
        super().__init__(provider="mock", model="mock-model")
        self._responses = responses or [
            LLMResponse(content="Hello!", finish_reason="stop")
        ]
        self._call_count = 0

    async def acomplete(self, messages, tools=None, **kwargs):
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = self._responses[-1]
        self._call_count += 1
        return response

    async def astream(self, messages, tools=None, **kwargs):
        response = (
            self._responses[self._call_count]
            if self._call_count < len(self._responses)
            else self._responses[-1]
        )
        self._call_count += 1

        if response.tool_calls:
            for tc in response.tool_calls:
                yield ToolCallStart(call_id=tc.id, tool_name=tc.name)
                yield ToolCallEnd(call_id=tc.id, tool_name=tc.name, arguments=tc.arguments)
        elif response.content:
            yield TextDelta(text=response.content)
        yield Usage(prompt_tokens=5, completion_tokens=3)
        yield StreamDone()


# --- Test tools ---


def safe_tool(query: str) -> str:
    """A safe tool that returns normal output."""
    return f"Result for: {query}"


def secret_tool(query: str) -> str:
    """A tool that returns sensitive data."""
    return "API key: sk-abc123secret"


# --- Tests ---


class TestToolCallGuardrail:
    @pytest.mark.asyncio
    async def test_tool_call_guardrail_blocks_dangerous_input(self):
        """Blocking guardrail at tool_call position prevents tool execution."""
        pii_guard = Guardrail(
            name="no-ssn-in-tools",
            position=GuardrailPosition.tool_call,
            blocking=True,
            fn=lambda text: "ssn" not in text.lower(),
        )

        tool = FunctionTool(name="lookup", fn=safe_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="lookup", arguments={"query": "SSN 123-45-6789"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Done", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], guardrails=[pii_guard], llm=llm)

        with pytest.raises(GuardrailBlockedError) as exc_info:
            await agent.arun("Find SSN", trace=False)
        assert "no-ssn-in-tools" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_call_guardrail_passes(self):
        """Non-triggering tool_call guardrail allows execution."""
        safe_guard = Guardrail(
            name="no-ssn",
            position=GuardrailPosition.tool_call,
            blocking=True,
            fn=lambda text: "ssn" not in text.lower(),
        )

        tool = FunctionTool(name="search", fn=safe_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="search", arguments={"query": "weather"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="The weather is sunny.", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], guardrails=[safe_guard], llm=llm)
        result = await agent.arun("What's the weather?", trace=False)
        assert result.output == "The weather is sunny."

    @pytest.mark.asyncio
    async def test_non_blocking_guardrail_continues(self):
        """Non-blocking guardrail logs but doesn't stop execution."""
        audit_guard = Guardrail(
            name="audit-log",
            position=GuardrailPosition.tool_call,
            blocking=False,
            fn=lambda text: False,  # Always "fails" but non-blocking
        )

        tool = FunctionTool(name="search", fn=safe_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="search", arguments={"query": "test"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Done", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], guardrails=[audit_guard], llm=llm)
        result = await agent.arun("Search", trace=False)
        assert result.output == "Done"


class TestToolResultGuardrail:
    @pytest.mark.asyncio
    async def test_tool_result_guardrail_blocks_sensitive_output(self):
        """Blocking guardrail at tool_result position catches sensitive tool output."""
        secret_guard = Guardrail(
            name="no-secrets",
            position=GuardrailPosition.tool_result,
            blocking=True,
            fn=lambda text: "sk-" not in text,
        )

        tool = FunctionTool(name="get_key", fn=secret_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="get_key", arguments={"query": "api_key"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Here's the key", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], guardrails=[secret_guard], llm=llm)

        with pytest.raises(GuardrailBlockedError) as exc_info:
            await agent.arun("Get API key", trace=False)
        assert "no-secrets" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tool_result_only_on_success(self):
        """tool_result guardrail does NOT run when tool execution fails."""

        def failing_tool(query: str) -> str:
            raise ValueError("Tool crashed")

        result_guard = Guardrail(
            name="check-result",
            position=GuardrailPosition.tool_result,
            blocking=True,
            fn=lambda text: False,  # Would block if called
        )

        tool = FunctionTool(name="bad_tool", fn=failing_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="bad_tool", arguments={"query": "x"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Tool failed", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], guardrails=[result_guard], llm=llm)
        # Should NOT raise GuardrailBlockedError because tool failed
        # (tool_result guardrail only runs on success)
        result = await agent.arun("Do something", trace=False)
        assert "Tool failed" in result.output


class TestMixedPositions:
    @pytest.mark.asyncio
    async def test_all_four_positions(self):
        """Input, tool_call, tool_result, and output guardrails all execute."""
        call_log: list[str] = []

        def make_guard(pos_name: str, position: GuardrailPosition) -> Guardrail:
            def check_fn(text):
                call_log.append(pos_name)
                return True

            return Guardrail(
                name=pos_name,
                position=position,
                blocking=True,
                fn=check_fn,
            )

        guards = [
            make_guard("input", GuardrailPosition.input),
            make_guard("tool_call", GuardrailPosition.tool_call),
            make_guard("tool_result", GuardrailPosition.tool_result),
            make_guard("output", GuardrailPosition.output),
        ]

        tool = FunctionTool(name="search", fn=safe_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="search", arguments={"query": "test"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Final answer", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], guardrails=guards, llm=llm)
        result = await agent.arun("Search", trace=False)

        assert result.output == "Final answer"
        assert call_log == ["input", "tool_call", "tool_result", "output"]


class TestNoGuardrails:
    @pytest.mark.asyncio
    async def test_no_guardrails_unchanged_behavior(self):
        """Agent without guardrails works normally."""
        tool = FunctionTool(name="search", fn=safe_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="search", arguments={"query": "test"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Done", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], llm=llm)
        result = await agent.arun("Search", trace=False)
        assert result.output == "Done"


class TestStreamToolGuardrails:
    @pytest.mark.asyncio
    async def test_stream_tool_call_guardrail_blocks(self):
        """Tool-position guardrails also work in streaming mode."""
        block_guard = Guardrail(
            name="block-all-tools",
            position=GuardrailPosition.tool_call,
            blocking=True,
            fn=lambda text: False,  # Always blocks
        )

        tool = FunctionTool(name="search", fn=safe_tool)
        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="c1", name="search", arguments={"query": "test"})
                    ],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Done", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", tools=[tool], guardrails=[block_guard], llm=llm)

        with pytest.raises(GuardrailBlockedError):
            async for _event in agent.astream("Search", trace=False):
                pass
