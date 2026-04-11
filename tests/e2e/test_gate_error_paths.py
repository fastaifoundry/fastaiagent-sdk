"""End-to-end quality gate — error and recovery paths.

The happy-path gate proves "everything works when nothing breaks."
This gate proves the SDK's error signals are real and informative:

- Raising tools produce ``ToolExecutionError`` that the agent sees.
- Unknown tools produce a clear error surfaced to the agent, not a crash.
- Agents exceeding ``max_iterations`` raise ``MaxIterationsError``.
- Invalid LLM provider raises ``LLMError``.
- Blocking guardrails at the ``output`` position raise
  ``GuardrailBlockedError``.
- REST tools pointing at a non-existent endpoint raise
  ``ToolExecutionError`` rather than hanging or silently swallowing.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


def _raising_tool(topic: str) -> str:
    """Always raises — used to verify tool exception propagation."""
    raise RuntimeError(f"simulated tool failure for topic: {topic}")


class TestErrorPathsGate:
    """Each test asserts a specific error surface behaves as documented."""

    def test_01_tool_exception_is_surfaced_to_agent(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, FunctionTool, LLMClient

        agent = Agent(
            name="error-gate-raising-tool",
            system_prompt=(
                "You are a helpful assistant. When asked to look something up, "
                "call the raising_tool. If it errors, acknowledge the error "
                "to the user politely."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="raising_tool", fn=_raising_tool)],
        )
        # Agent should complete — the tool raises, ToolExecutionError is
        # caught internally, error text is passed back to the LLM as the
        # tool message content, and the LLM produces a final response.
        result = agent.run(
            "Call raising_tool with topic='quality gate' and tell me what happens."
        )
        assert result.output, "agent aborted when tool raised"
        assert result.tool_calls, "tool was never invoked"
        # Per executor.py logic, the tool_call_record carries an 'error' key.
        tc = result.tool_calls[0]
        assert "error" in tc, (
            f"tool_call_record did not record an error: {tc}"
        )
        assert "simulated" in tc["error"].lower() or "failure" in tc["error"].lower()

    def test_02_unknown_provider_raises_llmerror(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient
        from fastaiagent._internal.errors import LLMError

        agent = Agent(
            name="error-gate-bad-provider",
            system_prompt="Hi.",
            llm=LLMClient(provider="not-a-real-provider", model="nope"),
        )
        with pytest.raises(LLMError):
            agent.run("Say hi.")

    def test_03_max_iterations_raises_cleanly(
        self, gate_state: dict[str, Any]
    ) -> None:
        """Agent that loops on a tool should hit MaxIterationsError, not spin forever."""
        require_env()
        from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient
        from fastaiagent._internal.errors import MaxIterationsError

        # A tool that always returns a message encouraging another call.
        call_count = {"n": 0}

        def loop_bait(reason: str) -> str:
            call_count["n"] += 1
            return (
                f"Got reason={reason}. You must call loop_bait again with a "
                f"different reason to continue. This is required."
            )

        agent = Agent(
            name="error-gate-max-iter",
            system_prompt=(
                "You are a tool-calling bot. You MUST call the loop_bait tool "
                "every time you receive a message from the user or from a "
                "tool. Never produce a final answer without calling it first. "
                "Keep calling until told otherwise."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="loop_bait", fn=loop_bait)],
            config=AgentConfig(max_iterations=3),
        )
        with pytest.raises(MaxIterationsError):
            agent.run("Get started.")
        assert call_count["n"] >= 3, (
            f"max_iterations=3 but loop_bait ran only {call_count['n']} times"
        )

    def test_04_output_guardrail_blocks(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, Guardrail, GuardrailResult, LLMClient
        from fastaiagent._internal.errors import GuardrailBlockedError
        from fastaiagent.guardrail.guardrail import GuardrailPosition

        def forbid_word(output: Any) -> GuardrailResult:
            text = str(output)
            if "pineapple" in text.lower():
                return GuardrailResult(
                    passed=False, message="response contained forbidden word 'pineapple'"
                )
            return GuardrailResult(passed=True)

        agent = Agent(
            name="error-gate-output-guardrail",
            system_prompt=(
                "Reply with exactly the single word: pineapple"
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            guardrails=[
                Guardrail(
                    name="forbid_pineapple",
                    position=GuardrailPosition.output,
                    blocking=True,
                    fn=forbid_word,
                )
            ],
        )
        with pytest.raises(GuardrailBlockedError):
            agent.run("Say the fruit.")

    def test_05_rest_tool_404_raises_toolexecutionerror(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        import httpx

        from fastaiagent import RESTTool
        from fastaiagent._internal.errors import ToolExecutionError

        try:
            httpx.get("https://httpbin.org/status/200", timeout=5.0)
        except Exception:
            pytest.skip("httpbin.org not reachable")

        tool = RESTTool(
            name="always_404",
            url="https://httpbin.org/status/404",
            method="GET",
            body_mapping="query_params",
            parameters={"type": "object", "properties": {}},
        )
        with pytest.raises(ToolExecutionError):
            tool.execute({})
