"""Judge template aliases + verdict robustness — real-LLM end-to-end tests (no mocking).

Covers the Evals Phase 0 contract against a live provider: a judge template written
with the *legacy* ``{expected_output}`` / double-brace aliases must interpolate SDK-side,
so the judge actually sees the expected answer. Before the fix those placeholders reached
the model raw and the judge scored blind.

Gated on ``OPENAI_API_KEY`` so it skips cleanly when absent; run with the key from your
shell profile::

    zsh -lc 'pytest tests/e2e/test_eval_judge_aliases_e2e.py -q'

Assertions check *relative separation* (a clearly-good case scores higher than a
clearly-bad one) rather than brittle absolute values, so they're robust to judge noise.
"""

from __future__ import annotations

import os

import pytest

from fastaiagent import LLMClient
from fastaiagent.eval import LLMJudge

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]

MODEL = "gpt-4o-mini"

# The question is deliberately one the model cannot grade without the reference: only
# the interpolated {expected_output} tells it which internal code is the right one.
QUESTION = "Which internal code does the Zurich branch use for same-day settlement?"
EXPECTED = "ZRH-SD-7"
GOOD_ANSWER = "It uses ZRH-SD-7."
BAD_ANSWER = "It uses BRN-ND-2."


def _llm() -> LLMClient:
    return LLMClient(provider="openai", model=MODEL)


def _judge(template: str) -> LLMJudge:
    return LLMJudge(criteria="correctness", prompt_template=template, llm=_llm())


@pytest.mark.parametrize(
    "template",
    [
        pytest.param(
            "Does the answer match the reference?\n"
            "Question: {input}\nReference: {expected}\nAnswer: {output}\n"
            'Respond with JSON: {"score": <0 or 1>, "reasoning": "<why>"}',
            id="canonical",
        ),
        pytest.param(
            "Does the answer match the reference?\n"
            "Question: {input}\nReference: {expected_output}\nAnswer: {output}\n"
            'Respond with JSON: {"score": <0 or 1>, "reasoning": "<why>"}',
            id="legacy-expected_output",
        ),
        pytest.param(
            "Does the answer match the reference?\n"
            "Question: {{input}}\nReference: {{expected_output}}\nAnswer: {{output}}\n"
            'Respond with JSON: {"score": <0 or 1>, "reasoning": "<why>"}',
            id="legacy-double-brace",
        ),
    ],
)
def test_alias_template_separates_good_vs_bad(template: str) -> None:
    """Every alias spelling must interpolate — otherwise the judge grades blind."""
    judge = _judge(template)
    good = judge.score(input=QUESTION, output=GOOD_ANSWER, expected=EXPECTED)
    bad = judge.score(input=QUESTION, output=BAD_ANSWER, expected=EXPECTED)

    assert good.passed, f"correct answer should pass, got {good}"
    assert not bad.passed, f"wrong answer should fail, got {bad}"
    assert good.score > bad.score


def test_default_template_still_scores() -> None:
    """Regression guard on the canonical default prompt against a real model."""
    judge = LLMJudge(criteria="correctness", llm=_llm())
    good = judge.score(input="What is 2 + 2?", output="4", expected="4")
    bad = judge.score(input="What is 2 + 2?", output="Seventeen.", expected="4")

    assert good.score > bad.score
    assert good.passed and not bad.passed


def test_reasoning_is_populated() -> None:
    """A real verdict parses through the JSON path and carries the model's reasoning."""
    judge = _judge(
        "Question: {input}\nReference: {expected_output}\nAnswer: {output}\n"
        'Respond with JSON: {"score": <0 or 1>, "reasoning": "<one sentence>"}'
    )
    result = judge.score(input=QUESTION, output=GOOD_ANSWER, expected=EXPECTED)

    assert result.reason
    assert result.reason != "Could not parse reasoning"
    assert 0.0 <= result.score <= 1.0
