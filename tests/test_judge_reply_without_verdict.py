"""A judge reply with no verdict field is could-not-run, at the detector and the guardrail.

Before 1.89.0 the shared detectors (``_internal/safety_detectors.py``, mirrored
by the plane) read a missing field as its default: ``{}`` from the injection
classifier was "no injection", from the toxicity classifier toxicity 0.0, from
claim extraction "no claims", so the ``no_prompt_injection``, ``toxicity_check``
and ``grounded`` guardrails **passed** text nobody had judged. That is the
CLAUDE.md §2.4 shape. ``judge_field`` now raises, and the guardrail reports
``errored`` and applies ``on_error``.

The plane's ``topics.parse_topics`` already rejected ``{}`` this way.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fastaiagent._internal.safety_detectors import (
    detect_prompt_injection,
    detect_toxicity,
    judge_field,
    score_groundedness,
)
from fastaiagent.guardrail.builtins import grounded, no_prompt_injection, toxicity_check
from fastaiagent.guardrail.implementations import run_guardrail
from fastaiagent.testing import FunctionModel


def replying(text: str) -> FunctionModel:
    return FunctionModel(lambda messages: text)


# --------------------------------------------------------------------------- #
# judge_field
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "key", "kind", "want"),
    [
        ('{"injection": true}', "injection", "bool", True),
        ('{"injection": "False"}', "injection", "bool", False),
        ('```json\n{"score": 0.4}\n```', "score", "number", 0.4),
        ('{"score": "0.75"}', "score", "number", 0.75),
        ('{"claims": []}', "claims", "list", []),
        ({"score": 1}, "score", "number", 1.0),
    ],
)
def test_judge_field_reads_a_verdict(raw: Any, key: str, kind: str, want: Any) -> None:
    assert judge_field(raw, key, kind) == want


@pytest.mark.parametrize(
    ("raw", "key", "kind"),
    [
        ("{}", "injection", "bool"),
        ('{"reasoning": "looks fine"}', "score", "number"),
        ('{"injection": "maybe"}', "injection", "bool"),
        ('{"score": true}', "score", "number"),
        ('{"score": "high"}', "score", "number"),
        ('{"claims": "one claim"}', "claims", "list"),
        ("[]", "claims", "list"),
    ],
)
def test_judge_field_refuses_a_reply_without_one(raw: str, key: str, kind: str) -> None:
    with pytest.raises(ValueError):
        judge_field(raw, key, kind)


# --------------------------------------------------------------------------- #
# The detectors, both raise_on_error settings
# --------------------------------------------------------------------------- #


def test_injection_detector_does_not_read_empty_as_clean() -> None:
    with pytest.raises(ValueError):
        detect_prompt_injection("x", mode="llm", llm=replying("{}"), raise_on_error=True)
    res = detect_prompt_injection("x", mode="llm", llm=replying("{}"))
    assert res.detected is False and "error" in res.reason.lower()


def test_toxicity_detector_does_not_read_empty_as_zero() -> None:
    with pytest.raises(ValueError):
        detect_toxicity("x", mode="llm", llm=replying("{}"), raise_on_error=True)


def test_groundedness_does_not_read_empty_as_no_claims() -> None:
    with pytest.raises(ValueError):
        asyncio.run(score_groundedness("out", "ctx", llm=replying("{}"), raise_on_error=True))
    res = asyncio.run(score_groundedness("out", "ctx", llm=replying('{"claims": []}')))
    assert res.score == 1.0 and res.total == 0, "an explicit empty list is still an answer"


def test_unverifiable_claim_counts_as_errored() -> None:
    replies = iter(['{"claims": ["a", "b"]}', '{"supported": true}', "{}"])
    llm = FunctionModel(lambda messages: next(replies))
    res = asyncio.run(score_groundedness("out", "ctx", llm=llm))
    assert (res.supported, res.errored, res.total) == (1, 1, 2)


# --------------------------------------------------------------------------- #
# The guardrails: errored, never a pass
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "make",
    [
        lambda: no_prompt_injection(mode="llm", llm=replying("{}"), on_error="block"),
        lambda: toxicity_check(mode="llm", llm=replying("{}"), on_error="block"),
        lambda: grounded("Refunds within 30 days.", llm=replying("{}"), on_error="block"),
    ],
    ids=["no_prompt_injection", "toxicity_check", "grounded"],
)
def test_guardrail_on_an_empty_reply_could_not_run(make: Any) -> None:
    result = asyncio.run(run_guardrail(make(), "Refunds are accepted for 90 days."))
    assert result.errored is True, result
    assert result.passed is False, "on_error='block' fails closed"


def test_guardrail_with_a_real_verdict_still_passes() -> None:
    rail = no_prompt_injection(
        mode="llm", llm=replying('{"injection": false, "reasoning": "benign"}')
    )
    result = asyncio.run(run_guardrail(rail, "What is the refund window?"))
    assert result.errored is False and result.passed is True
