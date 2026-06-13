"""AgentEval slice — real-LLM end-to-end tests (no mocking).

Exercises the LLM-backed paths against a live provider: scenario generation, the
new named metrics, and the hardening analysis. Gated on ``OPENAI_API_KEY`` so it
skips cleanly when absent; run with the key from your shell profile::

    zsh -lc 'pytest tests/e2e/test_eval_agenteval_e2e.py -q'
"""

from __future__ import annotations

import os

import pytest

from fastaiagent import (
    Agent,
    LLMClient,
    Scorecard,
    generate_scenarios,
    harden,
)
from fastaiagent.eval import ReflectionQuality, TaskCompletion
from fastaiagent.eval.simulate import (
    CriterionVerdict,
    SimulationResult,
    SimulationResults,
    TranscriptTurn,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]

MODEL = "gpt-4o-mini"


def _llm() -> LLMClient:
    return LLMClient(provider="openai", model=MODEL)


def _support_agent() -> Agent:
    return Agent(
        name="support",
        system_prompt=(
            "You are a customer-support agent for an online store. "
            "Help with orders, refunds, and shipping."
        ),
        llm=_llm(),
    )


# --------------------------------------------------------------------------- #
# Scenario generation
# --------------------------------------------------------------------------- #


def test_generate_scenarios_produces_usable_scenarios() -> None:
    scenarios = generate_scenarios(_support_agent(), n=3, llm=_llm())
    assert len(scenarios) >= 1
    s = scenarios[0]
    assert s.user.persona  # a simulated-user persona was generated
    assert s.success_criteria  # at least one success criterion
    assert s.name


# --------------------------------------------------------------------------- #
# Named metrics
# --------------------------------------------------------------------------- #


def test_task_completion_distinguishes_complete_vs_incomplete() -> None:
    tc = TaskCompletion(llm=_llm())
    complete = tc.score(input="What is 2 + 2?", output="2 + 2 equals 4.")
    incomplete = tc.score(
        input="Book a flight to Paris and give me the confirmation number.",
        output="Paris is lovely this time of year.",
    )
    assert complete.score > incomplete.score
    assert complete.passed is True


def test_reflection_quality_scores_a_consistent_answer() -> None:
    rq = ReflectionQuality(llm=_llm())
    good = rq.score(input="Is the Earth flat?", output="No — the Earth is an oblate spheroid.")
    assert good.score >= 0.5


# --------------------------------------------------------------------------- #
# Hardening loop
# --------------------------------------------------------------------------- #


def test_harden_recommends_fixes_for_failures() -> None:
    agent = Agent(
        name="order-bot",
        system_prompt="You answer customer questions.",
        llm=_llm(),
    )
    failing = SimulationResults(
        [
            SimulationResult(
                scenario_name="refund-policy",
                passed=False,
                transcript=[
                    TranscriptTurn(
                        turn_index=0, role="user", content="What is your refund policy?"
                    ),
                    TranscriptTurn(turn_index=1, role="assistant", content="I don't know."),
                ],
                verdicts=[
                    CriterionVerdict(
                        "states the 30-day refund policy",
                        "success",
                        False,
                        "the agent said it did not know",
                    )
                ],
            )
        ]
    )
    report = harden(agent, failing, llm=_llm())
    assert report.failure_count == 1
    assert len(report.recommendations) >= 1
    valid = {"instructions", "model", "tools", "guardrails", "memory"}
    assert all(r.target in valid for r in report.recommendations)
    # Scorecard rolls up the same run.
    assert Scorecard.from_simulation(failing).overall_pass_rate == 0.0
