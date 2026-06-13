"""Responsible-AI "Trust Layer" — deterministic, mock-free unit tests.

These exercise the zero-dependency paths only (secrets, keyword toxicity,
keyword topics, bundle composition). The LLM-backed paths (groundedness,
reflection, LLM toxicity/topics) are covered by the real-LLM tests in
``tests/e2e/test_responsible_ai_e2e.py`` — no mocking anywhere.
"""

from __future__ import annotations

import fastaiagent as fa
from fastaiagent._internal.safety_detectors import (
    GroundednessResult,
    SecretMatch,
    ToxicityResult,
    detect_secrets,
    detect_toxicity,
)

# --------------------------------------------------------------------------- #
# Secrets detection
# --------------------------------------------------------------------------- #


def test_detect_secrets_finds_common_credentials() -> None:
    text = (
        "aws=AKIAIOSFODNN7EXAMPLE github=ghp_" + "a" * 36 + " "
        'config api_key = "supersecret_value_123456" '
        "jwt=eyJhbGciOi.JzdWIiOiIxMjM0.SflKxwRJSMeKKF"
    )
    kinds = {m.kind for m in detect_secrets(text)}
    assert "aws_access_key_id" in kinds
    assert "github_token" in kinds
    assert "generic_secret" in kinds
    assert "jwt" in kinds


def test_detect_secrets_masks_the_raw_value() -> None:
    raw = "AKIAIOSFODNN7EXAMPLE"
    matches = detect_secrets(f"key={raw}")
    assert matches
    m = matches[0]
    assert isinstance(m, SecretMatch)
    # The raw secret must never appear verbatim in the masked form.
    assert raw not in m.masked
    assert m.masked.startswith("AKIA")
    assert "kind" in m.to_dict() and "masked" in m.to_dict()


def test_detect_secrets_clean_text_is_empty() -> None:
    assert detect_secrets("the quick brown fox jumps over the lazy dog") == []


def test_no_secrets_guardrail_blocks_and_passes() -> None:
    g = fa.no_secrets()
    blocked = g.execute("token ghp_" + "b" * 36)
    assert blocked.passed is False
    assert "github_token" in blocked.message
    # The blocking message must not leak the raw secret.
    assert "ghp_" + "b" * 36 not in blocked.message
    assert g.execute("nothing sensitive here").passed is True


# --------------------------------------------------------------------------- #
# Toxicity (keyword default — behaviour preserved)
# --------------------------------------------------------------------------- #


def test_detect_toxicity_keyword_default() -> None:
    res = detect_toxicity("this is a racist slur")
    assert isinstance(res, ToxicityResult)
    assert res.toxic is True
    assert res.score == 1.0
    assert "racist" in res.matched
    assert detect_toxicity("have a wonderful day").toxic is False


def test_toxicity_check_default_is_unchanged_keyword_behaviour() -> None:
    # Default toxicity_check() must remain the zero-dep keyword guardrail.
    g = fa.toxicity_check()
    blocked = g.execute("I will attack you")
    assert blocked.passed is False
    # Back-compat metadata key preserved.
    assert "toxic_words" in blocked.metadata
    assert g.execute("the weather is lovely").passed is True


def test_detect_toxicity_unknown_mode_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        detect_toxicity("x", mode="bogus")


# --------------------------------------------------------------------------- #
# Topic controls (keyword mode — zero-dep)
# --------------------------------------------------------------------------- #


def test_banned_topics_keyword() -> None:
    g = fa.banned_topics(["politics"], mode="keyword")
    assert g.execute("let us discuss politics today").passed is False
    assert g.execute("the weather is nice").passed is True


def test_allowed_topics_keyword_whitelist() -> None:
    g = fa.allowed_topics(["finance", "hr"], mode="keyword")
    assert g.execute("quarterly finance report").passed is True
    assert g.execute("latest sports news").passed is False


# --------------------------------------------------------------------------- #
# Groundedness guardrail — no-LLM paths
# --------------------------------------------------------------------------- #


def test_grounded_empty_reference_is_skipped() -> None:
    # An empty/missing reference must not block (and must not call an LLM).
    g = fa.grounded(lambda: "")
    res = g.execute("any output at all")
    assert res.passed is True
    assert "skipped" in (res.message or "")


def test_no_hallucination_is_grounded_alias() -> None:
    assert fa.no_hallucination is fa.grounded


def test_reflect_skips_non_terminal_response_without_llm() -> None:
    # A response with no text content must be returned untouched and must NOT
    # construct an LLMClient or make a call (so this runs offline, no key).
    from fastaiagent import MiddlewareContext, Reflect
    from fastaiagent._internal.async_utils import run_sync
    from fastaiagent.llm.client import LLMResponse

    reflect = Reflect(facts=["never reached"])
    resp = LLMResponse(content="")
    out = run_sync(reflect.after_model(MiddlewareContext(), resp))
    assert out is resp


def test_groundedness_result_shape() -> None:
    r = GroundednessResult(score=0.5, supported=1, total=2, reason="x")
    assert r.to_dict() == {"score": 0.5, "supported": 1, "total": 2, "reason": "x"}


# --------------------------------------------------------------------------- #
# responsible_ai() bundle composition
# --------------------------------------------------------------------------- #


def test_responsible_ai_default_bundle_is_llm_free() -> None:
    rails = fa.responsible_ai()
    names = [r.name for r in rails]
    # Default = zero-dependency checks only (no groundedness / topics / LLM tox).
    assert names == ["no_prompt_injection", "no_pii", "no_secrets"]


def test_responsible_ai_options_add_rails() -> None:
    rails = fa.responsible_ai(
        grounded_to=lambda: "ref",
        banned=["politics"],
        allowed=["finance"],
        toxicity=True,
        moderation=False,
    )
    names = [r.name for r in rails]
    assert "grounded" in names
    assert "banned_topics" in names
    assert "allowed_topics" in names
    assert "toxicity_check" in names


def test_responsible_ai_can_disable_defaults() -> None:
    rails = fa.responsible_ai(pii=False, prompt_injection=False, secrets=False)
    assert rails == []


# --------------------------------------------------------------------------- #
# Faithfulness refactor — behaviour-preserving (no-LLM path)
# --------------------------------------------------------------------------- #


def test_faithfulness_no_context_path_unchanged() -> None:
    from fastaiagent.eval.rag import Faithfulness

    scorer = Faithfulness()
    assert scorer.name == "faithfulness"
    res = scorer.score(input="q", output="a")  # no context kwarg
    assert res.score == 0.0
    assert res.passed is False
    assert res.reason == "No context provided"
