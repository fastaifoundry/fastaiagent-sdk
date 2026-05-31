"""Live simulation test using a real OpenAI LLM (agent + simulated user + judge).

Skipped unless OPENAI_API_KEY is set. Run with:
    OPENAI_API_KEY=sk-... pytest tests/test_simulate_live.py -v

Cost: a few cents using gpt-4o-mini. Excluded from the fast suite.
"""

from __future__ import annotations

import os

import pytest

SKIP_REASON = "OPENAI_API_KEY not set"
has_key = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not has_key, reason=SKIP_REASON)
def test_live_scenario_end_to_end() -> None:
    from fastaiagent import Agent, LLMClient
    from fastaiagent.eval import LLMJudge, Scenario, SimulatedUser, simulate

    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    agent = Agent(
        name="support-bot",
        system_prompt=(
            "You are a friendly customer-support agent for an online store. "
            "Be concise, polite, and explain the 30-day refund policy when asked."
        ),
        llm=llm,
    )

    scenario = Scenario(
        name="refund-policy",
        user=SimulatedUser(
            persona=(
                "A customer who bought shoes 10 days ago and wants to know if "
                "they can get a refund. Ask one clear question, then say END."
            ),
            llm=llm,
        ),
        success_criteria=["The agent explains the refund policy clearly and politely."],
        failure_criteria=["The agent is rude or refuses to help."],
        max_turns=4,
    )

    results = simulate(
        scenario,
        agent,
        judge=LLMJudge(llm=llm),
        persist=False,
    )

    r = results.results[0]
    assert len(r.transcript) >= 2  # at least one user + one agent turn
    assert any(t.role == "assistant" for t in r.transcript)
    # Every assistant turn that ran traced should carry a trace_id.
    assert all(t.trace_id for t in r.transcript if t.role == "assistant")
    assert isinstance(r.passed, bool)
    assert r.verdicts  # judge produced at least one verdict
