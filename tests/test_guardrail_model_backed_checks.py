"""``content_safety`` and ``groundedness`` — the two model-backed check types.

No mocks and no live LLM. The judge call is one line in each runner; everything
that decides a verdict — which categories a rule scores, what bar each one gets,
how a response is parsed, where the context comes from, what happens when it is
missing — is a pure function, and that is what is tested here against the real
code. The judge round-trips against a real model live in
``tests/e2e/test_guardrail_actions_e2e.py``.

Test names mirror ``backend/tests/test_guardrail_model_backed_checks.py`` on the
plane wherever the property is the same. The two modules are deliberate mirrors
of each other, so a rule reaches the same verdict at the edge as it does at
``POST /guardrails/{id}/test``.
"""

from __future__ import annotations

import asyncio

import pytest

import fastaiagent as fa
from fastaiagent.guardrail import grounding
from fastaiagent.guardrail import hazard_taxonomy as tax
from fastaiagent.guardrail.from_policy import guardrail_from_policy_rule
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType


# --------------------------------------------------------------------------- #
# content_safety — the taxonomy and the bar per category
# --------------------------------------------------------------------------- #
def test_the_default_category_set_is_the_common_harms_not_all_fourteen() -> None:
    """A judge asked about every category at once is longer, costlier and less
    reliable, so an unconfigured rule scores the harms nearly everyone wants."""
    assert set(tax.DEFAULT_CATEGORIES) < set(tax.MLCOMMONS_HAZARDS)
    assert tax.resolve_categories({}) == list(tax.DEFAULT_CATEGORIES)


def test_the_taxonomy_is_the_published_fourteen() -> None:
    assert list(tax.MLCOMMONS_HAZARDS) == [f"S{i}" for i in range(1, 15)]


def test_categories_are_returned_in_taxonomy_order_whatever_the_input_order() -> None:
    assert tax.resolve_categories({"categories": ["S10", "S1"]}) == ["S1", "S10"]


def test_an_unknown_category_code_is_dropped_rather_than_scored() -> None:
    assert tax.resolve_categories({"categories": ["S1", "S99", "banana"]}) == ["S1"]


def test_a_dict_of_categories_doubles_as_the_category_list() -> None:
    assert tax.resolve_categories({"categories": {"S10": 0.3, "S1": 0.9}}) == ["S1", "S10"]


def test_a_per_category_bar_overrides_the_default() -> None:
    """ "Block hate at 0.3 but allow borderline specialised advice up to 0.8" is
    the policy real operators write, and PASS/FAIL cannot express it."""
    config = {"threshold": 0.9, "thresholds": {"S10": 0.3}}
    bars = tax.resolve_thresholds(config, ["S6", "S10"])
    assert bars == {"S6": 0.9, "S10": 0.3}


def test_a_dict_of_categories_carries_the_thresholds_directly() -> None:
    bars = tax.resolve_thresholds({"categories": {"S10": 0.3}}, ["S10"])
    assert bars["S10"] == 0.3


def test_an_unusable_threshold_falls_back_rather_than_crashing() -> None:
    assert tax.resolve_thresholds({"threshold": "high"}, ["S1"])["S1"] == tax.DEFAULT_THRESHOLD
    assert tax.resolve_thresholds({"threshold": 5.0}, ["S1"])["S1"] == 1.0
    assert tax.resolve_thresholds({"threshold": -2}, ["S1"])["S1"] == 0.0


def test_the_prompt_names_every_category_and_ships_no_payload() -> None:
    prompt = tax.build_prompt(["S1", "S10"])
    assert '"S1": Violent crimes' in prompt
    assert '"S10": Hate' in prompt
    # The content is shipped separately in its own <<DATA>> block by the runner.
    assert "<<DATA>>" in prompt and "untrusted" in prompt


def test_scores_are_read_out_of_the_judges_json() -> None:
    scores = tax.parse_scores('{"S1": 0.0, "S10": 0.82}', ["S1", "S10"])
    assert scores == {"S1": 0.0, "S10": 0.82}


def test_a_lowercase_key_is_still_read() -> None:
    assert tax.parse_scores('{"s10": 0.4}', ["S10"]) == {"S10": 0.4}


def test_scores_are_clamped_to_the_unit_interval() -> None:
    assert tax.parse_scores('{"S1": 7, "S10": -3}', ["S1", "S10"]) == {"S1": 1.0, "S10": 0.0}


def test_an_unreadable_judge_response_fails_closed() -> None:
    """Treating an unreadable answer as all-zeros would turn a model outage into
    a silent pass, so every unreadable path raises and ``on_error`` decides."""
    for raw in ("I'd rather not say", "{not json}", "[1, 2, 3]", '{"S9": 0.4}'):
        with pytest.raises(ValueError):
            tax.parse_scores(raw, ["S1", "S10"])


def test_a_category_the_judge_declined_to_score_is_recorded_not_assumed_absent() -> None:
    scores = tax.parse_scores('{"S1": 0.1}', ["S1", "S6"])
    assert "S6" not in scores  # the runner reports it under "unscored"


def test_a_rule_naming_no_known_category_cannot_run() -> None:
    g = Guardrail(
        name="nonsense",
        guardrail_type=GuardrailType.content_safety,
        config={"categories": ["S99"]},
    )
    res = g.execute("hello")
    assert res.errored is True
    assert "hazard categories" in (res.message or "")
    # An un-runnable safety control fails closed by default.
    assert res.passed is False
    assert res.action_taken == "blocked"


# --------------------------------------------------------------------------- #
# groundedness — the pair, and where the context comes from
# --------------------------------------------------------------------------- #
def test_the_threshold_is_the_whole_policy() -> None:
    assert grounding.resolve_threshold({}) == grounding.DEFAULT_THRESHOLD
    assert grounding.resolve_threshold({"threshold": 0.9}) == 0.9
    assert grounding.resolve_threshold({"threshold": "very good"}) == grounding.DEFAULT_THRESHOLD
    assert grounding.resolve_threshold({"threshold": 4}) == 1.0


def test_a_json_object_payload_carries_both_halves() -> None:
    """The plane's shape, so the same rule works unchanged at /guardrails/{id}/test."""
    context, answer = grounding.extract_pair(
        {}, '{"context": "Refunds take 5 days.", "answer": "Refunds take 5 days."}'
    )
    assert context == "Refunds take 5 days."
    assert answer == "Refunds take 5 days."


def test_payload_keys_are_configurable_because_they_belong_to_the_caller() -> None:
    context, answer = grounding.extract_pair(
        {"context_key": "docs", "answer_key": "reply"},
        {"docs": "the docs", "reply": "the reply"},
    )
    assert (context, answer) == ("the docs", "the reply")


def test_retrieved_chunks_may_arrive_as_a_list() -> None:
    context, _ = grounding.extract_pair({}, {"context": ["chunk one", "chunk two"], "answer": "a"})
    assert context == "chunk one\n\nchunk two"


def test_the_run_scoped_slot_supplies_the_context_when_the_payload_is_a_bare_answer() -> None:
    """An output guardrail only ever receives the answer. This is the convention
    that lets a plane-authored rule reach the runtime's retrieval step."""
    with fa.guardrail_context(context="Refunds take 5 days."):
        context, answer = grounding.extract_pair({}, "Refunds are instant.")
    assert context == "Refunds take 5 days."
    assert answer == "Refunds are instant."


def test_the_slot_honours_the_rules_own_context_key() -> None:
    with fa.guardrail_context(docs=["a", "b"]):
        context, _ = grounding.extract_pair({"context_key": "docs"}, "an answer")
    assert context == "a\n\nb"


def test_a_bare_string_payload_with_no_slot_fails_closed() -> None:
    """An answer scored against nothing blocks everything and scored against
    itself blocks nothing. Both are worse than saying the rule could not run."""
    with pytest.raises(ValueError) as exc:
        grounding.extract_pair({}, "Refunds are instant.")
    assert "context" in str(exc.value)
    assert "guardrail_context" in str(exc.value)


def test_a_missing_half_of_the_pair_is_named_in_the_error() -> None:
    with pytest.raises(ValueError) as exc:
        grounding.extract_pair({}, {"answer": "hello"})
    assert "context" in str(exc.value)


def test_an_empty_context_is_missing_context() -> None:
    with fa.guardrail_context(context=""):
        with pytest.raises(ValueError):
            grounding.extract_pair({}, "an answer")


def test_the_slot_does_not_leak_out_of_the_block() -> None:
    with fa.guardrail_context(context="inside"):
        assert fa.get_guardrail_context()["context"] == "inside"
    assert fa.get_guardrail_context() == {}


def test_the_slot_does_not_leak_across_tasks() -> None:
    """ContextVars are per-task, so concurrent runs never see each other's context."""

    async def _scenario() -> tuple[dict, dict]:
        async def _with_context() -> dict:
            with fa.guardrail_context(context="mine"):
                await asyncio.sleep(0)
                return fa.get_guardrail_context()

        async def _without() -> dict:
            await asyncio.sleep(0)
            return fa.get_guardrail_context()

        return await asyncio.gather(_with_context(), _without())  # type: ignore[return-value]

    mine, theirs = asyncio.run(_scenario())
    assert mine == {"context": "mine"}
    assert theirs == {}


def test_a_groundedness_rule_with_no_context_fails_closed_end_to_end() -> None:
    g = Guardrail(
        name="grounded",
        guardrail_type=GuardrailType.groundedness,
        position=GuardrailPosition.output,
    )
    res = g.execute("Refunds are instant.")
    assert res.errored is True
    assert res.passed is False
    assert res.action_taken == "blocked"
    assert "context" in (res.message or "")


def test_a_supported_verdict_is_read_with_its_claims() -> None:
    score, unsupported = grounding.parse_verdict('{"score": 0.95, "unsupported": []}')
    assert (score, unsupported) == (0.95, [])

    score, unsupported = grounding.parse_verdict(
        '{"score": 0.2, "unsupported": ["refunds are instant"]}'
    )
    assert unsupported == ["refunds are instant"]


def test_at_most_five_unsupported_claims_are_kept() -> None:
    raw = '{"score": 0.1, "unsupported": ["a", "b", "c", "d", "e", "f"]}'
    assert grounding.parse_verdict(raw)[1] == ["a", "b", "c", "d", "e"]


@pytest.mark.parametrize(
    "raw",
    [
        "the answer looks fine to me",
        "{not json}",
        '{"unsupported": []}',
        '{"score": "very good"}',
    ],
)
def test_a_verdict_without_a_usable_score_fails_closed(raw: str) -> None:
    with pytest.raises(ValueError):
        grounding.parse_verdict(raw)


# --------------------------------------------------------------------------- #
# Both types reconstruct from a policy rule instead of being skipped
# --------------------------------------------------------------------------- #
def test_both_new_types_reconstruct_from_a_policy_rule() -> None:
    for impl, config in (
        ("content_safety", {"categories": ["S10"], "threshold": 0.3}),
        ("groundedness", {"threshold": 0.7, "context_key": "context"}),
    ):
        g = guardrail_from_policy_rule(
            {
                "name": f"plane-{impl}",
                "implementation_type": impl,
                "guardrail_type": "output",
                "validation_mode": "blocking",
                "config": config,
                "action": "block",
                "severity": "high",
                "floor": False,
                "agent_ids": [],
            }
        )
        assert g is not None, f"{impl} should no longer be skipped"
        assert g.guardrail_type.value == impl
        assert g.config == config
        assert g.origin == "plane"
        assert g.severity == "high"
