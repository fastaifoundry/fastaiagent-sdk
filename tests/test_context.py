"""Tests for RunContext — dependency injection for tools."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fastaiagent import RunContext
from fastaiagent._internal.errors import ToolExecutionError
from fastaiagent.agent import Agent, AgentConfig, AgentResult
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.tool import FunctionTool, RESTTool, tool


# --- Fixtures ---


@dataclass
class FakeState:
    user_id: str
    data: dict


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


# --- RunContext unit tests ---


class TestRunContext:
    def test_state_property(self):
        ctx = RunContext(state={"user_id": "u-1"})
        assert ctx.state == {"user_id": "u-1"}

    def test_state_with_dataclass(self):
        state = FakeState(user_id="u-1", data={"key": "val"})
        ctx = RunContext(state=state)
        assert ctx.state.user_id == "u-1"
        assert ctx.state.data == {"key": "val"}

    def test_repr(self):
        ctx = RunContext(state="hello")
        assert repr(ctx) == "RunContext(state='hello')"

    def test_generic_typing(self):
        ctx: RunContext[FakeState] = RunContext(state=FakeState(user_id="u-1", data={}))
        assert ctx.state.user_id == "u-1"


# --- FunctionTool context detection tests ---


class TestFunctionToolContextDetection:
    def test_detects_context_param(self):
        @tool(name="greet")
        def greet(ctx: RunContext[dict], name: str) -> str:
            return f"Hello {name}"

        assert greet._context_param_name == "ctx"

    def test_detects_unparameterized_context(self):
        @tool(name="greet")
        def greet(ctx: RunContext, name: str) -> str:
            return f"Hello {name}"

        assert greet._context_param_name == "ctx"

    def test_no_context_param(self):
        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        assert add._context_param_name is None

    def test_schema_excludes_context(self):
        @tool(name="search")
        def search(ctx: RunContext, query: str, limit: int = 10) -> str:
            return "results"

        assert "ctx" not in search.parameters["properties"]
        assert "query" in search.parameters["properties"]
        assert "limit" in search.parameters["properties"]
        assert search.parameters["required"] == ["query"]

    def test_openai_format_excludes_context(self):
        @tool(name="greet")
        def greet(ctx: RunContext[dict], name: str) -> str:
            return f"Hello {name}"

        openai_schema = greet.to_openai_format()
        params = openai_schema["function"]["parameters"]
        assert "ctx" not in params["properties"]
        assert "name" in params["properties"]


# --- FunctionTool context injection tests ---


class TestFunctionToolContextInjection:
    @pytest.mark.asyncio
    async def test_tool_receives_context(self):
        @tool(name="greet")
        def greet(ctx: RunContext[dict], name: str) -> str:
            return f"Hello {name}, user={ctx.state['user_id']}"

        result = await greet.aexecute(
            {"name": "Alice"},
            context=RunContext(state={"user_id": "u-1"}),
        )
        assert result.output == "Hello Alice, user=u-1"

    @pytest.mark.asyncio
    async def test_tool_without_context_still_works(self):
        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        result = await add.aexecute({"a": 1, "b": 2})
        assert result.output == 3

    @pytest.mark.asyncio
    async def test_tool_without_context_ignores_passed_context(self):
        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        result = await add.aexecute(
            {"a": 1, "b": 2},
            context=RunContext(state={"irrelevant": True}),
        )
        assert result.output == 3

    @pytest.mark.asyncio
    async def test_context_not_passed_raises_clear_error(self):
        @tool(name="needs_ctx")
        def needs_ctx(ctx: RunContext[dict], x: str) -> str:
            return ctx.state["key"]

        with pytest.raises(ToolExecutionError, match="needs_ctx"):
            await needs_ctx.aexecute({"x": "hello"})

    @pytest.mark.asyncio
    async def test_async_tool_with_context(self):
        @tool(name="async_search")
        async def async_search(ctx: RunContext[dict], query: str) -> str:
            return f"results for {query} by {ctx.state['user_id']}"

        result = await async_search.aexecute(
            {"query": "test"},
            context=RunContext(state={"user_id": "u-1"}),
        )
        assert result.output == "results for test by u-1"

    def test_sync_execute_with_context(self):
        @tool(name="greet")
        def greet(ctx: RunContext[dict], name: str) -> str:
            return f"Hello {name}, user={ctx.state['user_id']}"

        result = greet.execute(
            {"name": "Alice"},
            context=RunContext(state={"user_id": "u-1"}),
        )
        assert result.output == "Hello Alice, user=u-1"


# --- RESTTool / MCPTool signature compatibility ---


class TestToolSignatureCompatibility:
    @pytest.mark.asyncio
    async def test_rest_tool_accepts_context_param(self):
        """RESTTool should accept context kwarg without error (signature only)."""
        rest = RESTTool(name="api", url="https://example.com", method="GET")
        # We can't actually execute (no server), but verify the signature accepts context
        # by checking the method signature
        import inspect

        sig = inspect.signature(rest.aexecute)
        assert "context" in sig.parameters

    def test_function_tool_without_fn_accepts_context(self):
        """FunctionTool with no fn should handle context gracefully."""
        ft = FunctionTool(name="empty", description="No function")
        result = ft.execute({}, context=RunContext(state={}))
        assert not result.success
        assert result.error is not None and "No function" in result.error


# --- Agent integration tests ---


class TestAgentWithContext:
    @pytest.mark.asyncio
    async def test_agent_run_with_context(self):
        """Context flows from agent.arun() through to the tool."""
        received_ctx = {}

        @tool(name="get_user")
        def get_user(ctx: RunContext[FakeState], user_id: str) -> str:
            received_ctx["state"] = ctx.state
            return f"User: {ctx.state.user_id}"

        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="get_user", arguments={"user_id": "u-1"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Found the user.", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", llm=llm, tools=[get_user])
        ctx = RunContext(state=FakeState(user_id="u-456", data={"role": "admin"}))
        result = await agent.arun("Find user u-1", context=ctx)

        assert result.output == "Found the user."
        assert received_ctx["state"].user_id == "u-456"

    @pytest.mark.asyncio
    async def test_agent_run_without_context_backward_compat(self):
        """Agent.arun() without context works exactly as before."""
        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="The answer is 5.", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", llm=llm, tools=[add])
        result = await agent.arun("Add 2 and 3")
        assert result.output == "The answer is 5."

    @pytest.mark.asyncio
    async def test_agent_mixed_tools_with_context(self):
        """Context-aware and context-free tools work together."""
        @tool(name="lookup")
        def lookup(ctx: RunContext[dict], key: str) -> str:
            return ctx.state.get(key, "not found")

        @tool(name="add")
        def add(a: int, b: int) -> int:
            return a + b

        llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="lookup", arguments={"key": "name"})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="Done.", finish_reason="stop"),
            ]
        )

        agent = Agent(name="test", llm=llm, tools=[lookup, add])
        ctx = RunContext(state={"name": "Alice"})
        result = await agent.arun("Look up name", context=ctx)
        assert result.output == "Done."


# --- Serialization safety ---


class TestSerializationSafety:
    def test_to_dict_excludes_context(self):
        agent = Agent(
            name="test-agent",
            system_prompt="Be helpful",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            tools=[FunctionTool(name="greet", description="Greet")],
            config=AgentConfig(max_iterations=5),
        )
        d = agent.to_dict()
        assert "context" not in d
        assert "RunContext" not in str(d)

    def test_tool_schema_excludes_context_in_to_dict(self):
        @tool(name="search")
        def search(ctx: RunContext, query: str) -> str:
            return "results"

        d = search.to_dict()
        assert "ctx" not in str(d["parameters"])
        assert "query" in str(d["parameters"])
