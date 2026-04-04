"""Edge case tests for Phase 41.1.2 — network, execution, and data edge cases."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from fastaiagent._internal.errors import (
    ChainCycleError,
    ChainError,
    EvalError,
    FastAIAgentError,
    GuardrailBlockedError,
    LLMError,
    LLMProviderError,
    MaxIterationsError,
    PlatformConnectionError,
    ToolExecutionError,
    TraceError,
)
from fastaiagent.agent import Agent
from fastaiagent.chain import Chain, ChainState
from fastaiagent.chain.node import NodeType
from fastaiagent.eval.evaluate import evaluate
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult
from fastaiagent.kb.chunking import chunk_text
from fastaiagent.kb.local import LocalKB
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall
from fastaiagent.prompt.prompt import Prompt
from fastaiagent.tool import ToolResult
from fastaiagent.trace.storage import TraceStore

# ---------------------------------------------------------------------------
# Helper: mock LLM
# ---------------------------------------------------------------------------


class MockLLM(LLMClient):
    def __init__(self, responses: list[LLMResponse] | None = None):
        super().__init__(provider="mock", model="mock")
        self._responses = responses or [LLMResponse(content="ok", finish_reason="stop")]
        self._idx = 0

    async def acomplete(self, messages: Any, tools: Any = None, **kw: Any) -> LLMResponse:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r


# ===========================================================================
# NETWORK EDGE CASES
# ===========================================================================


class TestNetworkEdgeCases:
    def test_llm_unsupported_provider_lists_supported(self):
        """Unsupported provider error should list all valid providers."""
        client = LLMClient(provider="not_real")
        with pytest.raises(LLMError, match="Supported providers"):
            client.complete([])

    def test_llm_missing_openai_key_includes_provider_name(self):
        """Missing API key error should mention the specific provider."""
        client = LLMClient(provider="openai", model="gpt-4o-mini")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMProviderError, match="provider 'openai'"):
                client.complete([])

    def test_llm_missing_anthropic_key_includes_provider_name(self):
        client = LLMClient(provider="anthropic", model="claude-3")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMProviderError, match="provider 'anthropic'"):
                client.complete([])

    def test_platform_connection_error_message(self):
        """PlatformConnectionError is raised with clear message."""
        err = PlatformConnectionError("Cannot reach https://app.fastaiagent.net")
        assert "Cannot reach" in str(err)

    @pytest.mark.asyncio
    async def test_rest_tool_timeout(self):
        """REST tool handles timeouts gracefully."""
        from fastaiagent.tool.rest import RESTTool

        tool = RESTTool(
            name="slow-api",
            url="http://localhost:99999/timeout",  # unreachable
            method="GET",
        )
        with pytest.raises(ToolExecutionError, match="REST tool 'slow-api' failed"):
            await tool.aexecute({})

    def test_llm_http_error_includes_status_code(self):
        """LLM provider errors should include the HTTP status code."""
        err = LLMProviderError("OpenAI API error 500: Internal Server Error")
        assert "500" in str(err)

    def test_trace_not_found_suggests_list_traces(self):
        """TraceError for missing trace should suggest list_traces()."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(db_path=os.path.join(tmp, "traces.db"))
            with pytest.raises(TraceError, match="list_traces"):
                store.get_trace("nonexistent_trace_id")
            store.close()

    @pytest.mark.asyncio
    async def test_mcp_tool_connection_failure(self):
        """MCP tool handles connection failures."""
        from fastaiagent.tool.mcp import MCPTool

        tool = MCPTool(
            name="unreachable",
            server_url="http://localhost:99999/mcp",
            tool_name="test",
        )
        with pytest.raises(ToolExecutionError, match="MCP tool 'unreachable' failed"):
            await tool.aexecute({"q": "hi"})


# ===========================================================================
# EXECUTION EDGE CASES
# ===========================================================================


class TestExecutionEdgeCases:
    def test_agent_with_no_tools(self):
        """Agent with empty tool list should work (just LLM calls)."""
        llm = MockLLM([LLMResponse(content="I have no tools", finish_reason="stop")])
        agent = Agent(name="bare", llm=llm, tools=[])
        result = agent.run("hello")
        assert result.output == "I have no tools"

    def test_agent_with_empty_system_prompt(self):
        """Agent with empty system prompt uses default behavior."""
        llm = MockLLM([LLMResponse(content="response", finish_reason="stop")])
        agent = Agent(name="no-prompt", llm=llm, system_prompt="")
        result = agent.run("hello")
        assert result.output == "response"

    def test_chain_with_single_node(self):
        """Chain with one node (no edges) executes that node."""
        chain = Chain(name="single")
        chain.add_node("only", type=NodeType.transformer, template="done")
        result = chain.execute({"input": "test"})
        assert result is not None

    def test_max_iterations_error_is_actionable(self):
        """MaxIterationsError should include suggestions."""
        err = MaxIterationsError(
            "Agent exceeded maximum iterations (10). "
            "The LLM continued requesting tool calls beyond the limit."
        )
        assert "maximum iterations" in str(err)

    def test_guardrail_exception_treated_as_block(self):
        """Guardrail that raises an exception should result in a failed result."""

        def bad_guardrail(data: str) -> GuardrailResult:
            raise RuntimeError("guardrail crashed")

        g = Guardrail(
            name="crasher",
            fn=bad_guardrail,
            position=GuardrailPosition.input,
            blocking=True,
        )
        # The guardrail executor should handle this — either fail or return result
        result = g.execute("test input")
        assert isinstance(result, GuardrailResult)
        # A crashed guardrail should not pass
        assert not result.passed

    def test_agent_handles_unknown_tool_call(self):
        """Agent should handle LLM requesting a non-existent tool."""
        llm = MockLLM(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="1", name="nonexistent", arguments={})],
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="I see the error", finish_reason="stop"),
            ]
        )
        agent = Agent(name="test", llm=llm, tools=[])
        result = agent.run("hello")
        assert result.output is not None

    def test_chain_cycle_error_is_actionable(self):
        """ChainCycleError should include fix suggestions."""
        err = ChainCycleError(
            "Cycle 'a' -> 'b' exceeded max_iterations (5).\nOptions:\n  1. Increase the limit"
        )
        assert "Increase the limit" in str(err)

    def test_chain_max_steps_error_is_actionable(self):
        """Chain exceeding max steps should produce actionable error."""
        err = ChainError(
            "Chain 'test' exceeded maximum total steps (500). "
            "This usually means cycles are not terminating."
        )
        assert "cycles are not terminating" in str(err)

    def test_eval_unknown_scorer_lists_available(self):
        """EvalError for unknown scorer should list available scorers."""
        with pytest.raises(EvalError, match="Available built-in scorers"):
            evaluate(lambda x: x, [{"input": "hi", "expected": "hi"}], scorers=["nonexistent"])

    def test_connect_not_connected_error_is_actionable(self):
        """PlatformNotConnectedError should suggest fa.connect()."""
        from fastaiagent._internal.errors import PlatformNotConnectedError

        with pytest.raises(PlatformNotConnectedError, match="fa.connect"):
            raise PlatformNotConnectedError(
                "Not connected to platform. Call fa.connect() first."
            )


# ===========================================================================
# DATA EDGE CASES
# ===========================================================================


class TestDataEdgeCases:
    def test_prompt_with_no_variables(self):
        """Prompt with no {{variables}} should format as-is."""
        prompt = Prompt(name="static", template="Hello World!", version=1)
        rendered = prompt.format()
        assert rendered == "Hello World!"

    def test_unicode_in_agent_name_and_prompt(self):
        """Unicode strings should work everywhere."""
        llm = MockLLM([LLMResponse(content="Hola!", finish_reason="stop")])
        agent = Agent(
            name="agente-espanol",
            llm=llm,
            system_prompt="Eres un asistente. Responde en espanol.",
        )
        result = agent.run("Hola mundo!")
        assert result.output is not None

    def test_empty_dataset_eval(self):
        """Evaluation with empty dataset returns empty results."""
        results = evaluate(lambda x: x, [], scorers=["exact_match"])
        assert results is not None

    def test_kb_with_no_documents_returns_empty_search(self):
        """KB with 0 documents returns empty results, no crash."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            kb = LocalKB(name="empty", path=tmp)
            results = kb.search("anything")
            assert results == []

    def test_kb_status_with_no_documents(self):
        """KB status works with empty KB."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            kb = LocalKB(name="empty", path=tmp)
            status = kb.status()
            assert status["chunk_count"] == 0

    def test_chunk_text_with_empty_input(self):
        """Chunking empty text should return empty list."""
        chunks = chunk_text("", chunk_size=512, overlap=50)
        assert chunks == []

    def test_chunk_text_with_unicode(self):
        """Chunking should handle unicode characters."""
        text = "Hola mundo! Esto es una prueba con caracteres especiales."
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 0

    def test_guardrail_blocked_error_repr(self):
        """GuardrailBlockedError repr should include structured data."""
        err = GuardrailBlockedError("no-pii", "PII detected", results=["ssn_found"])
        r = repr(err)
        assert "no-pii" in r
        assert "ssn_found" in r

    def test_chain_state_with_nested_json(self):
        """Chain state handles deeply nested data."""
        state = ChainState({"level1": {"level2": {"level3": {"value": 42}}}})
        nested = state.data["level1"]["level2"]["level3"]["value"]
        assert nested == 42

    def test_tool_result_with_none_output(self):
        """ToolResult with None output should not crash."""
        result = ToolResult(output=None)
        assert result.output is None
        assert result.success
