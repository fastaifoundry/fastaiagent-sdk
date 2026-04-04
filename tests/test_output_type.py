"""Tests for Agent output_type — structured output with Pydantic models."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from fastaiagent.agent import Agent, AgentResult
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.llm.stream import StreamDone, TextDelta, Usage
from fastaiagent.tool import FunctionTool

# --- Test models ---


class OrderResult(BaseModel):
    order_id: str
    status: str
    total: float


class Address(BaseModel):
    street: str
    city: str


class Customer(BaseModel):
    name: str
    address: Address


# --- Mock LLM that returns JSON ---


class MockJSONLLMClient(LLMClient):
    """Returns a JSON string matching the requested model."""

    def __init__(self, json_text: str):
        super().__init__(provider="mock", model="mock-model")
        self._json_text = json_text
        self._calls: list[dict] = []

    async def acomplete(self, messages, tools=None, **kwargs):
        self._calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return LLMResponse(content=self._json_text, finish_reason="stop")

    async def astream(self, messages, tools=None, **kwargs):
        self._calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        # Yield each char as a TextDelta to simulate streaming
        for char in self._json_text:
            yield TextDelta(text=char)
        yield Usage(prompt_tokens=10, completion_tokens=5)
        yield StreamDone()


# --- Tests ---


class TestOutputType:
    @pytest.mark.asyncio
    async def test_parsed_model(self):
        """output_type parses response into Pydantic model."""
        data = json.dumps({"order_id": "ORD-123", "status": "shipped", "total": 49.99})
        llm = MockJSONLLMClient(data)
        agent = Agent(name="test", output_type=OrderResult, llm=llm)
        result = await agent.arun("Get order 123", trace=False)

        assert isinstance(result.parsed, OrderResult)
        assert result.parsed.order_id == "ORD-123"
        assert result.parsed.status == "shipped"
        assert result.parsed.total == 49.99
        assert isinstance(result.output, str)
        assert result.output == data

    @pytest.mark.asyncio
    async def test_parsed_none_by_default(self):
        """Without output_type, parsed is None."""
        llm = MockJSONLLMClient("Hello world")
        agent = Agent(name="test", llm=llm)
        result = await agent.arun("Hello", trace=False)

        assert result.parsed is None
        assert result.output == "Hello world"

    @pytest.mark.asyncio
    async def test_strips_code_fences(self):
        """Markdown-fenced JSON is stripped before parsing."""
        raw = '```json\n{"order_id":"1","status":"ok","total":10.0}\n```'
        llm = MockJSONLLMClient(raw)
        agent = Agent(name="test", output_type=OrderResult, llm=llm)
        result = await agent.arun("Get order", trace=False)

        assert isinstance(result.parsed, OrderResult)
        assert result.parsed.order_id == "1"

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self):
        """Malformed JSON results in parsed=None, output preserved."""
        llm = MockJSONLLMClient("This is not JSON at all")
        agent = Agent(name="test", output_type=OrderResult, llm=llm)
        result = await agent.arun("Get order 123", trace=False)

        assert result.parsed is None
        assert result.output == "This is not JSON at all"

    @pytest.mark.asyncio
    async def test_nested_models(self):
        """Nested Pydantic models parse correctly."""
        data = json.dumps(
            {"name": "Alice", "address": {"street": "123 Main St", "city": "Springfield"}}
        )
        llm = MockJSONLLMClient(data)
        agent = Agent(name="test", output_type=Customer, llm=llm)
        result = await agent.arun("Get customer", trace=False)

        assert isinstance(result.parsed, Customer)
        assert isinstance(result.parsed.address, Address)
        assert result.parsed.name == "Alice"
        assert result.parsed.address.city == "Springfield"

    @pytest.mark.asyncio
    async def test_response_format_passed_to_llm(self):
        """response_format kwargs reach the LLM client."""
        data = json.dumps({"order_id": "1", "status": "ok", "total": 5.0})
        llm = MockJSONLLMClient(data)
        agent = Agent(name="test", output_type=OrderResult, llm=llm)
        await agent.arun("Get order", trace=False)

        assert len(llm._calls) == 1
        kwargs = llm._calls[0]["kwargs"]
        assert "response_format" in kwargs
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "OrderResult"
        assert "properties" in rf["json_schema"]["schema"]

    def test_to_dict_includes_response_format(self):
        """to_dict() includes response_format when output_type is set."""
        agent = Agent(name="test", output_type=OrderResult, llm=MockJSONLLMClient("{}"))
        d = agent.to_dict()

        assert "response_format" in d["config"]
        assert d["config"]["response_format"]["type"] == "json_schema"
        assert d["config"]["response_format"]["json_schema"]["name"] == "OrderResult"
        assert "properties" in d["config"]["response_format"]["json_schema"]["schema"]

    def test_to_dict_no_response_format_without_output_type(self):
        """to_dict() does not include response_format when output_type is None."""
        agent = Agent(name="test", llm=MockJSONLLMClient("{}"))
        d = agent.to_dict()
        assert "response_format" not in d["config"]

    def test_from_dict_output_type_is_none(self):
        """from_dict() cannot restore output_type (Python class)."""
        agent = Agent(name="test", output_type=OrderResult, llm=MockJSONLLMClient("{}"))
        d = agent.to_dict()
        agent2 = Agent.from_dict(d)
        assert agent2.output_type is None

    def test_stream_sets_parsed(self):
        """Sync stream() returns AgentResult with parsed populated."""
        data = json.dumps({"order_id": "1", "status": "ok", "total": 5.0})
        llm = MockJSONLLMClient(data)
        agent = Agent(name="test", output_type=OrderResult, llm=llm)
        result = agent.stream("Get order", trace=False)

        assert isinstance(result, AgentResult)
        assert isinstance(result.parsed, OrderResult)
        assert result.parsed.order_id == "1"

    @pytest.mark.asyncio
    async def test_output_type_with_tools(self):
        """output_type works alongside tools — final response is structured."""

        def lookup(order_id: str) -> str:
            return f"Order {order_id} found: shipped, $49.99"

        tool = FunctionTool(name="lookup", fn=lookup)

        # First call: tool call. Second call: JSON response.
        class MockToolThenJSONLLM(LLMClient):
            def __init__(self):
                super().__init__(provider="mock", model="mock")
                self._call_count = 0

            async def acomplete(self, messages, tools=None, **kwargs):
                self._call_count += 1
                if self._call_count == 1:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall(id="c1", name="lookup", arguments={"order_id": "123"})
                        ],
                        finish_reason="tool_calls",
                    )
                return LLMResponse(
                    content='{"order_id":"123","status":"shipped","total":49.99}',
                    finish_reason="stop",
                )

        agent = Agent(name="test", output_type=OrderResult, tools=[tool], llm=MockToolThenJSONLLM())
        result = await agent.arun("Get order 123", trace=False)

        assert isinstance(result.parsed, OrderResult)
        assert result.parsed.order_id == "123"
        assert len(result.tool_calls) == 1
