"""Smoke tests for calibrate_judge.py — no live LLM calls.

Run from examples/autollm (CI does): ``pytest tests/``. The data, the held-back
split, the agreement scorer, and the LLMJudge template the calibrated prompt
becomes.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import calibrate_judge as cj  # noqa: E402


def test_the_replies_are_labelled_and_balanced() -> None:
    rows = cj.rows()
    assert len(rows) == 80
    assert len({r["id"] for r in rows}) == 80
    assert sum(r["human"] for r in rows) == 40
    assert all(r["note"] and r["human"] in (0, 1) for r in rows)


def test_the_fresh_replies_are_held_back_and_balanced() -> None:
    calibration, fresh = cj.split(cj.rows())
    assert (len(calibration), len(fresh)) == (60, 20)
    assert not {r["id"] for r in calibration} & {r["id"] for r in fresh}
    assert sum(r["human"] for r in fresh) == 10


@pytest.mark.parametrize(
    ("output", "passed", "reason"),
    [
        ('{"score": 0, "reasoning": "promises a refund"}', True, None),
        (
            '{"score": 1, "reasoning": "friendly"}',
            False,
            "judge said pass; reviewers said fail: promises a refund",
        ),
        ("looks fine to me", False, "no verdict: 'looks fine to me'"),
    ],
)
def test_agreement_scorer_names_the_reviewers_note(
    output: str, passed: bool, reason: str | None
) -> None:
    expected = json.dumps({"score": 0, "note": "promises a refund"})
    result = cj.AgreesWithReviewers().score("case", output, expected)
    assert (result.passed, result.reason) == (passed, reason)


def test_the_judge_template_shows_the_judge_the_case() -> None:
    """The LLMJudge the calibrated prompt becomes renders question and reply."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a template with no placeholder warns
        judge = cj.as_llm_judge(cj.NAIVE_JUDGE, "naive_judge")
    assert "{input}" in judge.prompt_template and "{output}" in judge.prompt_template
