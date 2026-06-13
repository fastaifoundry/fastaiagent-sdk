"""G-Eval judge + LLM-judged turn metrics — deterministic, mock-free unit tests.

Covers the no-LLM surfaces: scale normalization, rubric rendering, prompt building,
G-Eval mode activation, the verbatim legacy path, per-instance naming, the session
``mode="heuristic"`` defaults, and the early-return branches that never touch the
network. LLM-backed behaviour (good-vs-bad separation, auto-CoT, the session/new-metric
LLM scoring) is covered by the real-LLM tests in ``tests/e2e/test_eval_geval_e2e.py`` —
no mocking anywhere.
"""

from __future__ import annotations

import fastaiagent.eval as ev
from fastaiagent.eval import (
    ConversationCoherence,
    ConversationRelevancy,
    GEval,
    GoalCompletion,
    KnowledgeRetention,
    LLMJudge,
    RoleAdherence,
)
from fastaiagent.eval.llm_judge import (
    _build_geval_prompt,
    _normalize_to_unit,
    _parse_scale,
    _render_rubric,
)

# --- scale normalization (pure) ------------------------------------------- #


def test_parse_scale_bounds() -> None:
    assert _parse_scale("binary") == (0.0, 1.0)
    assert _parse_scale("0-1") == (0.0, 1.0)
    assert _parse_scale("1-5") == (1.0, 5.0)
    assert _parse_scale("1-10") == (1.0, 10.0)
    assert _parse_scale("nonsense") == (0.0, 1.0)  # graceful fallback


def test_normalize_to_unit() -> None:
    assert _normalize_to_unit(3, "1-5") == 0.5
    assert _normalize_to_unit(5, "1-5") == 1.0
    assert _normalize_to_unit(1, "1-5") == 0.0
    assert _normalize_to_unit(0.8, "0-1") == 0.8
    assert _normalize_to_unit(1, "binary") == 1.0
    # out-of-range clamps into [0, 1]
    assert _normalize_to_unit(7, "1-5") == 1.0
    assert _normalize_to_unit(-2, "1-5") == 0.0


# --- rubric + prompt building (pure) -------------------------------------- #


def test_render_rubric_orders_and_formats() -> None:
    out = _render_rubric([(1, "Mostly wrong"), (3, "Partial"), (5, "Fully correct")])
    assert out == "- Score 1: Mostly wrong\n- Score 3: Partial\n- Score 5: Fully correct"


def test_build_geval_prompt_includes_steps_rubric_and_io() -> None:
    p = _build_geval_prompt(
        criteria="correctness",
        steps=["Compare each claim to the expected answer."],
        rubric=[(1, "wrong"), (5, "right")],
        scale="1-5",
        input="What is 2+2?",
        output="4",
        expected="4",
        context=None,
    )
    assert "Compare each claim to the expected answer." in p
    assert "- Score 5: right" in p
    assert "What is 2+2?" in p and "4" in p
    assert "Reference / expected output" in p  # expected present
    assert "Context:" not in p  # context omitted when absent
    assert "from 1 to 5" in p


def test_build_geval_prompt_omits_absent_blocks() -> None:
    p = _build_geval_prompt(
        criteria="helpfulness",
        steps=None,
        rubric=None,
        scale="0-1",
        input="hi",
        output="hello",
        expected=None,
        context=None,
    )
    assert "Evaluation steps" not in p
    assert "Scoring rubric" not in p
    assert "Reference / expected output" not in p


# --- G-Eval mode activation ----------------------------------------------- #


def test_legacy_mode_off_by_default() -> None:
    j = LLMJudge(criteria="correctness")
    assert j._geval_mode is False
    # legacy prompt template is byte-identical to the historical default
    assert j.prompt_template == j._default_prompt()
    assert "{input}" in j.prompt_template
    assert "Score should be between 0 and 1." in j.prompt_template


def test_mode_activates_with_steps_or_rubric() -> None:
    assert LLMJudge(evaluation_steps=["a"])._geval_mode is True
    assert LLMJudge(rubric=[(1, "bad"), (5, "good")])._geval_mode is True


def test_geval_subclass_defaults() -> None:
    g = GEval(name="correctness", criteria="Is it correct?")
    assert g._geval_mode is True
    assert g._force_geval is True
    assert g.scale == "1-5"
    assert g.auto_steps is True
    assert g.name == "correctness"


# --- per-instance naming (avoids results-dict collisions) ----------------- #


def test_name_default_and_override() -> None:
    assert LLMJudge().name == "llm_judge"
    assert LLMJudge(name="custom_judge").name == "custom_judge"


# --- backward-compat: session heuristic defaults unchanged ---------------- #


def test_session_scorers_default_to_heuristic() -> None:
    assert ConversationCoherence().mode == "heuristic"
    assert GoalCompletion().mode == "heuristic"


def test_conversation_coherence_heuristic_penalizes_contradiction() -> None:
    coherent = ConversationCoherence().score(
        input="",
        output="",
        turns=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Who created Python language?"},
            {"role": "assistant", "content": "Python language was created by Guido van Rossum."},
        ],
    )
    contradictory = ConversationCoherence().score(
        input="",
        output="",
        turns=[
            {"role": "assistant", "content": "Your order ships Monday via FedEx."},
            {"role": "assistant", "content": "Actually, I was wrong — there is no such order."},
        ],
    )
    assert coherent.score > contradictory.score


def test_goal_completion_heuristic_unchanged() -> None:
    res = GoalCompletion().score(
        input="",
        output="The order shipped via FedEx and arrives Friday.",
        goal="order shipped FedEx arrives Friday",
    )
    assert res.score > 0.5
    # no-goal path
    none = GoalCompletion().score(input="", output="x")
    assert none.passed is False and none.reason == "No goal specified"


def test_unknown_mode_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        ConversationCoherence(mode="bogus")
    with pytest.raises(ValueError):
        GoalCompletion(mode="bogus")


# --- early-return branches hit no network (no API key needed) -------------- #


def test_llm_mode_empty_turns_short_circuits() -> None:
    # mode="llm" but empty turns → deterministic early return, no LLM call
    r = ConversationCoherence(mode="llm").score(input="", output="", turns=[])
    assert r.score == 1.0 and r.passed is True and r.reason == "No turns to evaluate"
    assert KnowledgeRetention().score(input="", output="", turns=[]).score == 1.0
    assert ConversationRelevancy().score(input="", output="", turns=[]).score == 1.0


def test_role_adherence_requires_role() -> None:
    r = RoleAdherence().score(input="", output="", turns=[{"role": "user", "content": "hi"}])
    assert r.score == 0.0 and r.passed is False and r.reason == "No role specified"


def test_goal_completion_llm_no_goal_short_circuits() -> None:
    r = GoalCompletion(mode="llm").score(input="", output="x")
    assert r.passed is False and r.reason == "No goal specified"


# --- exports -------------------------------------------------------------- #


def test_new_symbols_exported() -> None:
    for name in ("GEval", "KnowledgeRetention", "RoleAdherence", "ConversationRelevancy"):
        assert name in ev.__all__, name
        assert hasattr(ev, name), name
