"""Integration tests for RunContext with real LLM providers (OpenAI & Anthropic).

These tests make actual API calls to verify the full context injection pipeline
works end-to-end: Agent -> executor -> tool with RunContext.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from fastaiagent import Agent, AgentConfig, LLMClient, RunContext, tool

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class AppState:
    user_id: str
    db: dict  # fake in-memory DB


# --- Tools ---


@tool(name="get_user_info")
def get_user_info(ctx: RunContext[AppState]) -> str:
    """Get current user information."""
    return f"User ID: {ctx.state.user_id}"


@tool(name="lookup_record")
def lookup_record(ctx: RunContext[AppState], record_id: str) -> str:
    """Look up a record by ID from the database."""
    record = ctx.state.db.get(record_id, None)
    if record is None:
        return f"Record '{record_id}' not found"
    return f"Record {record_id}: {record}"


@tool(name="add_numbers")
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool(name="async_greet")
async def async_greet(ctx: RunContext[AppState], name: str) -> str:
    """Greet someone by name, including the current user."""
    return f"Hello {name}! (from user {ctx.state.user_id})"


# --- Helpers ---


def make_context() -> RunContext[AppState]:
    return RunContext(
        state=AppState(
            user_id="u-integration-test",
            db={
                "order-123": {"status": "shipped", "total": 49.99},
                "order-456": {"status": "pending", "total": 12.00},
            },
        )
    )


# ==================== OpenAI Tests ====================


@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
class TestRunContextOpenAI:
    """End-to-end tests with OpenAI."""

    def _make_agent(self, tools_list):
        return Agent(
            name="openai-ctx-test",
            system_prompt="You are a helpful assistant. Use tools when needed. Be concise.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini", api_key=OPENAI_API_KEY),
            tools=tools_list,
            config=AgentConfig(max_iterations=5),
        )

    @pytest.mark.asyncio
    async def test_tool_with_context_receives_it(self):
        """Tool with RunContext gets the state injected via OpenAI."""
        agent = self._make_agent([get_user_info])
        ctx = make_context()
        result = await agent.arun("What is my user info?", context=ctx)

        assert result.output  # LLM produced some response
        assert "u-integration-test" in result.output
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0]["tool_name"] == "get_user_info"
        print(f"\n[OpenAI] tool_with_context: {result.output}")

    @pytest.mark.asyncio
    async def test_tool_with_context_and_args(self):
        """Tool with RunContext + LLM-provided args works via OpenAI."""
        agent = self._make_agent([lookup_record])
        ctx = make_context()
        result = await agent.arun("Look up record order-123", context=ctx)

        assert result.output
        assert "shipped" in result.output or "49.99" in result.output
        print(f"\n[OpenAI] context_and_args: {result.output}")

    @pytest.mark.asyncio
    async def test_tool_without_context_backward_compat(self):
        """Tool WITHOUT RunContext works normally (no context passed)."""
        agent = self._make_agent([add_numbers])
        result = await agent.arun("What is 17 + 25?")

        assert result.output
        assert "42" in result.output
        print(f"\n[OpenAI] no_context: {result.output}")

    @pytest.mark.asyncio
    async def test_mixed_tools_with_context(self):
        """Mix of context-aware and plain tools in one agent."""
        agent = self._make_agent([get_user_info, add_numbers, lookup_record])
        ctx = make_context()
        result = await agent.arun(
            "First tell me my user info, then add 10 + 20.",
            context=ctx,
        )

        assert result.output
        assert len(result.tool_calls) >= 2
        tool_names = [tc["tool_name"] for tc in result.tool_calls]
        assert "get_user_info" in tool_names
        assert "add_numbers" in tool_names
        print(f"\n[OpenAI] mixed_tools: {result.output}")

    @pytest.mark.asyncio
    async def test_async_tool_with_context(self):
        """Async tool with RunContext works via OpenAI."""
        agent = self._make_agent([async_greet])
        ctx = make_context()
        result = await agent.arun("Greet Alice", context=ctx)

        assert result.output
        assert "Alice" in result.output
        print(f"\n[OpenAI] async_tool: {result.output}")

    @pytest.mark.asyncio
    async def test_streaming_with_context(self):
        """Streaming execution with RunContext via OpenAI."""
        from fastaiagent.llm.stream import TextDelta

        agent = self._make_agent([get_user_info])
        ctx = make_context()

        chunks = []
        async for event in agent.astream("What is my user info?", context=ctx):
            if isinstance(event, TextDelta):
                chunks.append(event.text)

        full_output = "".join(chunks)
        assert full_output
        assert "u-integration-test" in full_output
        print(f"\n[OpenAI] streaming: {full_output}")

    def test_to_dict_excludes_context(self):
        """Serialization never includes RunContext."""
        agent = self._make_agent([get_user_info, add_numbers])
        d = agent.to_dict()
        d_str = str(d)
        assert "RunContext" not in d_str
        assert "context" not in d  # top level
        # Verify tool schemas exclude context param
        for tool_dict in d["tools"]:
            props = tool_dict.get("parameters", {}).get("properties", {})
            assert "ctx" not in props
        print("\n[OpenAI] serialization: context excluded from to_dict()")

    def test_schema_sent_to_llm_excludes_context(self):
        """Verify the OpenAI tool schema sent to LLM has no context param."""
        schema = get_user_info.to_openai_format()
        params = schema["function"]["parameters"]
        assert "ctx" not in params.get("properties", {})
        # get_user_info only has ctx, so properties should be empty
        assert len(params.get("properties", {})) == 0
        print(f"\n[OpenAI] schema check: {schema}")


# ==================== Anthropic Tests ====================


@pytest.mark.skipif(not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set")
class TestRunContextAnthropic:
    """End-to-end tests with Anthropic Claude."""

    def _make_agent(self, tools_list):
        return Agent(
            name="anthropic-ctx-test",
            system_prompt="You are a helpful assistant. Use tools when needed. Be concise.",
            llm=LLMClient(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                api_key=ANTHROPIC_API_KEY,
            ),
            tools=tools_list,
            config=AgentConfig(max_iterations=5),
        )

    @pytest.mark.asyncio
    async def test_tool_with_context_receives_it(self):
        """Tool with RunContext gets the state injected via Anthropic."""
        agent = self._make_agent([get_user_info])
        ctx = make_context()
        result = await agent.arun("What is my user info?", context=ctx)

        assert result.output
        assert "u-integration-test" in result.output
        assert len(result.tool_calls) >= 1
        print(f"\n[Anthropic] tool_with_context: {result.output}")

    @pytest.mark.asyncio
    async def test_tool_with_context_and_args(self):
        """Tool with RunContext + LLM-provided args works via Anthropic."""
        agent = self._make_agent([lookup_record])
        ctx = make_context()
        result = await agent.arun("Look up record order-456", context=ctx)

        assert result.output
        assert "pending" in result.output or "12" in result.output
        print(f"\n[Anthropic] context_and_args: {result.output}")

    @pytest.mark.asyncio
    async def test_tool_without_context_backward_compat(self):
        """Tool WITHOUT RunContext works normally via Anthropic."""
        agent = self._make_agent([add_numbers])
        result = await agent.arun("What is 100 + 200?")

        assert result.output
        assert "300" in result.output
        print(f"\n[Anthropic] no_context: {result.output}")

    @pytest.mark.asyncio
    async def test_mixed_tools_with_context(self):
        """Mix of context-aware and plain tools via Anthropic."""
        agent = self._make_agent([get_user_info, add_numbers, lookup_record])
        ctx = make_context()
        result = await agent.arun(
            "Tell me my user info and also add 7 + 8.",
            context=ctx,
        )

        assert result.output
        assert len(result.tool_calls) >= 2
        tool_names = [tc["tool_name"] for tc in result.tool_calls]
        assert "get_user_info" in tool_names
        assert "add_numbers" in tool_names
        print(f"\n[Anthropic] mixed_tools: {result.output}")

    @pytest.mark.asyncio
    async def test_async_tool_with_context(self):
        """Async tool with RunContext works via Anthropic."""
        agent = self._make_agent([async_greet])
        ctx = make_context()
        result = await agent.arun("Greet Bob", context=ctx)

        assert result.output
        assert "Bob" in result.output
        print(f"\n[Anthropic] async_tool: {result.output}")

    @pytest.mark.asyncio
    async def test_record_not_found(self):
        """Tool returns 'not found' for missing records — context still injected."""
        agent = self._make_agent([lookup_record])
        ctx = make_context()
        result = await agent.arun("Look up record order-999", context=ctx)

        assert result.output
        assert "not found" in result.output.lower() or "999" in result.output
        print(f"\n[Anthropic] not_found: {result.output}")
