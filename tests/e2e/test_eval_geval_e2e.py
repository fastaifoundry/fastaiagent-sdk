"""G-Eval judge + LLM-judged turn metrics — real-LLM end-to-end tests (no mocking).

Exercises the LLM-backed paths against a live provider: G-Eval scoring with explicit
evaluation steps + score-band rubric, auto-generated steps (Auto-CoT), the unchanged
legacy ``LLMJudge`` path, the new ``mode="llm"`` session scorers, and the three new
turn metrics. Gated on ``OPENAI_API_KEY`` so it skips cleanly when absent; run with the
key from your shell profile::

    zsh -lc 'pytest tests/e2e/test_eval_geval_e2e.py -q'

Assertions check *relative separation* (a clearly-good case scores higher than a
clearly-bad one) rather than brittle absolute values, so they're robust to judge noise.
"""

from __future__ import annotations

import os

import pytest

from fastaiagent import LLMClient
from fastaiagent.eval import (
    ConversationCoherence,
    ConversationRelevancy,
    GEval,
    GoalCompletion,
    KnowledgeRetention,
    LLMJudge,
    RoleAdherence,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]

MODEL = "gpt-4o-mini"


def _llm() -> LLMClient:
    return LLMClient(provider="openai", model=MODEL)


# --- G-Eval judge ---------------------------------------------------------- #


def test_geval_separates_good_vs_bad() -> None:
    judge = GEval(
        name="correctness",
        criteria="Is the answer factually correct and complete?",
        evaluation_steps=[
            "Compare each claim in the answer to the expected answer.",
            "Penalize fabricated, missing, or contradicted facts.",
        ],
        rubric=[(1, "Mostly incorrect"), (3, "Partially correct"), (5, "Fully correct")],
        scale="1-5",
        llm=_llm(),
    )
    good = judge.score(input="What is the capital of France?", output="Paris.", expected="Paris")
    bad = judge.score(
        input="What is the capital of France?",
        output="The capital of France is Berlin.",
        expected="Paris",
    )
    assert good.score > bad.score
    assert good.passed is True


def test_geval_rubric_anchoring_top_band() -> None:
    judge = GEval(
        name="correctness",
        criteria="Factual correctness",
        rubric=[(1, "Wrong"), (5, "Fully correct")],
        scale="1-5",
        llm=_llm(),
    )
    good = judge.score(input="2 + 2 = ?", output="4", expected="4")
    assert good.score >= 0.6  # lands in the upper band after normalization


def test_geval_auto_generates_steps() -> None:
    # No evaluation_steps provided → GEval derives them from criteria (Auto-CoT).
    judge = GEval(
        name="helpfulness",
        criteria="Does the response helpfully and directly answer the user's question?",
        scale="1-5",
        llm=_llm(),
    )
    good = judge.score(
        input="How do I reset my password?",
        output="Go to Settings → Security → Reset password, then follow the email link.",
    )
    bad = judge.score(input="How do I reset my password?", output="Passwords are important.")
    assert good.score > bad.score
    # the derived steps are cached on the instance after first use
    assert judge.evaluation_steps is not None and len(judge.evaluation_steps) >= 1


def test_legacy_llmjudge_still_works_live() -> None:
    # The untouched legacy path: criteria-only, passed = score >= 0.5.
    judge = LLMJudge(criteria="correctness", llm=_llm())
    res = judge.score(input="What is 2+2?", output="4", expected="4")
    assert res.passed is True
    assert res.score >= 0.5


# --- session scorers: mode="llm" ------------------------------------------ #

_COHERENT = [
    {"role": "user", "content": "Where is my order #12345?"},
    {"role": "assistant", "content": "Order #12345 shipped via FedEx and arrives Friday."},
    {"role": "user", "content": "Which carrier again?"},
    {"role": "assistant", "content": "FedEx — it's on track to arrive Friday."},
]
_CONTRADICTORY = [
    {"role": "user", "content": "Where is my order #12345?"},
    {"role": "assistant", "content": "Order #12345 shipped via FedEx and arrives Friday."},
    {"role": "user", "content": "Which carrier again?"},
    {"role": "assistant", "content": "It shipped via UPS and already arrived last week."},
]


def test_session_coherence_llm() -> None:
    scorer = ConversationCoherence(mode="llm", llm=_llm())
    good = scorer.score(input="", output="", turns=_COHERENT)
    bad = scorer.score(input="", output="", turns=_CONTRADICTORY)
    assert good.score > bad.score


def test_goal_completion_llm() -> None:
    scorer = GoalCompletion(mode="llm", llm=_llm())
    achieved = scorer.score(
        input="",
        output="",
        goal="Tell the customer their order's carrier and arrival day.",
        turns=_COHERENT,
    )
    not_achieved = scorer.score(
        input="",
        output="",
        goal="Tell the customer their order's carrier and arrival day.",
        turns=[
            {"role": "user", "content": "Where is my order?"},
            {"role": "assistant", "content": "I can't help with that."},
        ],
    )
    assert achieved.score > not_achieved.score


# --- new turn metrics ------------------------------------------------------ #


def test_knowledge_retention_llm() -> None:
    scorer = KnowledgeRetention(llm=_llm())
    retains = scorer.score(
        input="",
        output="",
        turns=[
            {"role": "user", "content": "My name is Dana and my order is #777."},
            {"role": "assistant", "content": "Thanks Dana. Let me check order #777."},
            {"role": "user", "content": "Any update?"},
            {"role": "assistant", "content": "Yes Dana — order #777 ships tomorrow."},
        ],
    )
    forgets = scorer.score(
        input="",
        output="",
        turns=[
            {"role": "user", "content": "My name is Dana and my order is #777."},
            {"role": "assistant", "content": "Thanks. Let me check."},
            {"role": "user", "content": "Any update?"},
            {"role": "assistant", "content": "Can you remind me of your name and order number?"},
        ],
    )
    assert retains.score > forgets.score


def test_role_adherence_llm() -> None:
    scorer = RoleAdherence(
        role="a formal banking assistant who never gives medical advice", llm=_llm()
    )
    in_role = scorer.score(
        input="",
        output="",
        turns=[
            {"role": "user", "content": "Can you help me with my account balance?"},
            {"role": "assistant", "content": "Certainly. Your checking balance is $1,240.50."},
        ],
    )
    breaks_role = scorer.score(
        input="",
        output="",
        turns=[
            {"role": "user", "content": "Can you help me with my account balance?"},
            {"role": "assistant", "content": "Sure! Also, for your headache take 600mg ibuprofen."},
        ],
    )
    assert in_role.score > breaks_role.score


def test_conversation_relevancy_llm() -> None:
    scorer = ConversationRelevancy(llm=_llm())
    relevant = scorer.score(
        input="",
        output="",
        turns=[
            {"role": "user", "content": "What time does the store close?"},
            {"role": "assistant", "content": "The store closes at 9pm tonight."},
        ],
    )
    off_topic = scorer.score(
        input="",
        output="",
        turns=[
            {"role": "user", "content": "What time does the store close?"},
            {"role": "assistant", "content": "Our company was founded in 1998 in Seattle."},
        ],
    )
    assert relevant.score > off_topic.score
