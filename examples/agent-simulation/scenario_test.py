"""Agent simulation — multi-turn scenario tests, deterministic and offline.

Drives multi-turn conversations against an agent with ``simulate()`` and judges
the transcript against natural-language criteria. Uses ``TestModel`` /
``FunctionModel`` (real ``LLMClient`` subclasses) for the agent, the simulated
user, and the judge — so the whole run is deterministic, with no network and no
API key. Swap in a real ``LLMClient`` to run it against a live model.

Run via pytest:
    pytest examples/agent-simulation/scenario_test.py -v
"""

from __future__ import annotations

import json

from fastaiagent import Agent, Scenario, SimulatedUser, simulate
from fastaiagent.eval import LLMJudge
from fastaiagent.testing import FunctionModel, TestModel


def _criterion_aware_judge() -> LLMJudge:
    """A deterministic judge: success criteria pass (1.0), failure criteria are
    judged absent (0.0). The judge prompt phrases failure criteria as
    '...undesirable condition occurred...', so we branch on that marker."""

    def responder(messages):
        prompt = str(messages[-1].content)
        if "undesirable condition occurred" in prompt:
            return json.dumps({"score": 0.0, "reasoning": "did not occur"})
        return json.dumps({"score": 1.0, "reasoning": "criterion met"})

    return LLMJudge(llm=FunctionModel(responder))


# --- Scenario 1: scripted user --------------------------------------------- #


def test_scripted_refund_conversation() -> None:
    agent = Agent(
        name="support",
        system_prompt="You are a friendly support agent.",
        llm=TestModel(response="Our refund policy allows returns within 30 days."),
    )
    scenario = Scenario(
        name="refund-policy",
        user=SimulatedUser(script=["Can I get a refund?", "Great, thanks!"]),
        success_criteria=["The agent explains the refund policy."],
        failure_criteria=["The agent is rude."],
        max_turns=6,
    )

    results = simulate(scenario, agent, judge=_criterion_aware_judge(), persist=False)
    result = results.results[0]

    assert result.passed is True
    # user → assistant → user → assistant
    assert [t.role for t in result.transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


# --- Scenario 2: persona-driven user --------------------------------------- #


def test_persona_user() -> None:
    user_turns = iter(["I'm locked out of my account.", "END"])

    def user_responder(messages):
        return next(user_turns)

    agent = Agent(
        name="support",
        system_prompt="You help users with account issues.",
        llm=TestModel(response="I can help you reset your password."),
    )
    scenario = Scenario(
        name="account-lockout",
        user=SimulatedUser(
            persona="A user who is locked out.", llm=FunctionModel(user_responder)
        ),
        success_criteria=["The agent offers a concrete next step."],
        max_turns=8,
    )

    results = simulate(scenario, agent, judge=_criterion_aware_judge(), persist=False)
    assert results.results[0].passed is True
    print(results.summary())
