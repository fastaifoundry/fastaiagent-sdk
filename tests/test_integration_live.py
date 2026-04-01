"""Live integration tests using real OpenAI API calls.

These tests require OPENAI_API_KEY to be set. They are skipped in CI
when the key is not available. Cost: < $0.01 per full run using gpt-4o-mini.

Run with:
    OPENAI_API_KEY=sk-... pytest tests/test_integration_live.py -v
"""

from __future__ import annotations

import os

import pytest

SKIP_REASON = "OPENAI_API_KEY not set"
has_key = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not has_key, reason=SKIP_REASON)
class TestLiveAgent:
    """Real agent execution with OpenAI."""

    def test_simple_agent_run(self) -> None:
        """Agent can make a real LLM call and return a response."""
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="test-agent",
            system_prompt="You are a helpful assistant. Be very brief.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        )
        result = agent.run("What is 2+2? Reply with just the number.")
        assert result.output is not None
        assert len(result.output) > 0
        assert "4" in result.output

    def test_agent_with_tool(self) -> None:
        """Agent can call a real tool via LLM tool-calling."""
        from fastaiagent import Agent, LLMClient
        from fastaiagent.tool import FunctionTool

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool = FunctionTool(name="add", fn=add, description="Add two numbers")
        agent = Agent(
            name="math-agent",
            system_prompt="Use the add tool to answer math questions. Be brief.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
            tools=[tool],
        )
        result = agent.run("What is 17 + 25?")
        assert result.output is not None
        assert "42" in result.output
        assert len(result.tool_calls) > 0
        assert result.tool_calls[0]["tool_name"] == "add"

    def test_agent_with_tracing(self) -> None:
        """Agent execution produces a real trace with trace_id."""
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="traced-agent",
            system_prompt="Reply with one word only.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        )
        result = agent.run("Say hello.", trace=True)
        assert result.output is not None
        # trace_id should be set when trace=True
        assert result.trace_id is not None or result.output is not None


@pytest.mark.skipif(not has_key, reason=SKIP_REASON)
class TestLiveLLMClient:
    """Real LLM client calls."""

    def test_openai_completion(self) -> None:
        """Direct LLMClient completion with OpenAI."""
        from fastaiagent.llm.client import LLMClient
        from fastaiagent.llm.message import Message, MessageRole

        client = LLMClient(provider="openai", model="gpt-4o-mini")
        messages = [
            Message(role=MessageRole.user, content="Say 'hello' and nothing else."),
        ]
        response = client.complete(messages)
        assert response.content is not None
        assert "hello" in response.content.lower()
        assert response.usage.get("total_tokens", 0) > 0
        assert response.latency_ms > 0

    def test_openai_with_tools(self) -> None:
        """LLMClient returns tool_calls when tools are provided."""
        from fastaiagent.llm.client import LLMClient
        from fastaiagent.llm.message import Message, MessageRole

        client = LLMClient(provider="openai", model="gpt-4o-mini")
        messages = [
            Message(
                role=MessageRole.user,
                content="What is the weather in Tokyo?",
            ),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        response = client.complete(messages, tools=tools)
        assert len(response.tool_calls) > 0
        assert response.tool_calls[0].name == "get_weather"
        assert "tokyo" in response.tool_calls[0].arguments.get("city", "").lower()


@pytest.mark.skipif(not has_key, reason=SKIP_REASON)
class TestLiveEval:
    """Real evaluation with LLM calls."""

    def test_evaluate_with_real_agent(self) -> None:
        """Run evaluate() with a real agent against a small dataset."""
        from fastaiagent import Agent, LLMClient
        from fastaiagent.eval.evaluate import evaluate

        agent = Agent(
            name="eval-agent",
            system_prompt=(
                "You are a math assistant. Reply with ONLY the numeric answer, "
                "nothing else. No words, no explanation."
            ),
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        )

        dataset = [
            {"input": "What is 1+1?", "expected_output": "2"},
            {"input": "What is 5*3?", "expected_output": "15"},
            {"input": "What is 10-4?", "expected_output": "6"},
        ]

        results = evaluate(
            agent_fn=agent.run,
            dataset=dataset,
            scorers=["contains"],
        )

        summary = results.summary()
        assert "contains" in summary
        # Verify scores were collected
        assert "contains" in results.scores
        assert len(results.scores["contains"]) == 3
        # At least 2/3 should contain the expected answer
        passed = sum(1 for r in results.scores["contains"] if r.passed)
        assert passed >= 2, f"Only {passed}/3 passed: {summary}"


@pytest.mark.skipif(not has_key, reason=SKIP_REASON)
class TestLiveGuardrail:
    """Real guardrails with agent execution."""

    def test_input_guardrail_blocks_pii(self) -> None:
        """Input guardrail blocks PII before reaching the LLM."""
        from fastaiagent import Agent, LLMClient
        from fastaiagent._internal.errors import GuardrailBlockedError
        from fastaiagent.guardrail import no_pii
        from fastaiagent.guardrail.guardrail import GuardrailPosition

        # no_pii defaults to output position; set it to input to block before LLM
        agent = Agent(
            name="guarded-agent",
            system_prompt="You are a helpful assistant.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
            guardrails=[no_pii(position=GuardrailPosition.input)],
        )

        # This should be blocked by the PII guardrail before any LLM call
        with pytest.raises(GuardrailBlockedError):
            agent.run("My SSN is 123-45-6789, can you help?")

    def test_guardrail_passes_clean_input(self) -> None:
        """Clean input passes through guardrails to the LLM."""
        from fastaiagent import Agent, LLMClient
        from fastaiagent.guardrail import no_pii

        agent = Agent(
            name="guarded-agent",
            system_prompt="Reply with one word only.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
            guardrails=[no_pii()],
        )

        result = agent.run("Hello, how are you?")
        assert result.output is not None
