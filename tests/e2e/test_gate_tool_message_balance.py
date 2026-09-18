"""End-to-end gate — the real provider must accept the history after a stop.

``tests/test_message_balance.py`` owns the invariant against the SDK's own
``MockLLMClient``, which is exactly the client that *cannot* tell you whether
OpenAI and Anthropic agree with our reading of their contract. They do, loudly:
before 1.67.0 this pairing reproduced

    400 An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'

on OpenAI and the equivalent ``tool_use ids were found without tool_result
blocks`` on Anthropic.

The pairing matters. ``ToolBudget`` ends the turn between the assistant message
and its results; ``output_type`` is what re-sends that history (a structured
re-ask), which is the second request — the one that 400s. A budget low enough to
bite mid-turn is the whole point, so the tools are cheap and deterministic.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.e2e


def _require(provider_key: str) -> None:
    if os.environ.get(provider_key):
        return
    message = f"{provider_key} not set — skipping tool-message-balance gate step"
    if os.environ.get("E2E_REQUIRED") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.mark.parametrize(
    ("provider", "provider_key", "model"),
    [
        ("openai", "OPENAI_API_KEY", "gpt-4o-mini"),
        ("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
    ],
    ids=["openai", "anthropic"],
)
def test_a_stopped_turn_is_still_a_history_the_provider_accepts(
    provider, provider_key, model
) -> None:
    _require(provider_key)

    from pydantic import BaseModel

    import fastaiagent as fa
    from fastaiagent import Agent, AgentConfig, LLMClient, ToolBudget

    calls: list[str] = []

    @fa.tool()
    def lookup_population(city: str) -> str:
        """Return the population of a city."""
        calls.append(city)
        return f"{city}: 1,000,000"

    class Report(BaseModel):
        summary: str

    agent = Agent(
        name="balance-gate",
        llm=LLMClient(provider=provider, model=model),
        system_prompt=(
            "Look up the population of EVERY city the user names. Call "
            "lookup_population once per city, all in the same turn."
        ),
        tools=[lookup_population],
        output_type=Report,
        # Bites in the middle of a multi-call turn.
        middleware=[ToolBudget(max_calls=1, message="Tool budget exhausted.")],
        config=AgentConfig(max_iterations=3),
    )

    # No pytest.raises: the assertion is that nothing 400s. A provider error
    # surfaces as LLMProviderError and fails the test with its own message,
    # which is more informative than anything we could assert here.
    result = agent.run("Population of Paris, Tokyo and Lagos?")

    assert result.output, "the run produced no output at all"
    assert len(calls) <= 1, f"ToolBudget(max_calls=1) let {len(calls)} calls through"
