"""The action spectrum: what a guardrail failure actually costs.

No mocks. Every property here is either a pure function (``coerce_action``,
``mask_payload``, ``halts``) or a real guardrail run over a regex/classifier/
callable — the same production path an agent takes. The one "errored check"
fixture is a Python callable that raises, which is a genuine failed check, not a
simulated one.

Test names mirror ``backend/tests/test_guardrail_actions.py`` on the plane
wherever the property is the same, so the two sides can be read side by side.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.actions import (
    ACTIONS,
    coerce_action,
    coerce_severity,
    halts,
    harness_halts,
    mask_payload,
)
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.trace.otel import get_tracer_provider
from fastaiagent.ui.events import _outcome

SSN = r"\b\d{3}-\d{2}-\d{4}\b"
DIRTY = "call me on 123-45-6789 tomorrow"


def _regex_rule(**kwargs) -> Guardrail:
    config = {"pattern": SSN, "should_match": False, **kwargs.pop("config", {})}
    return Guardrail(
        name=kwargs.pop("name", "no-ssn"),
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config=config,
        **kwargs,
    )


def _run(guardrails: list[Guardrail], data: str = DIRTY):
    return asyncio.run(execute_guardrails(guardrails, data, GuardrailPosition.output))


# --------------------------------------------------------------------------- #
# block — the default, and every rule that predates the action spectrum
# --------------------------------------------------------------------------- #
def test_block_is_the_default_and_stops_the_caller() -> None:
    g = _regex_rule()
    assert g.action == "block"
    with pytest.raises(GuardrailBlockedError):
        _run([g])

    res = g.execute(DIRTY)
    assert (res.passed, res.action_taken, res.modified_data) == (False, "blocked", None)
    assert halts(g, res) is True


def test_a_clean_payload_reports_no_action_at_all() -> None:
    """A pass is a pass whatever the rule was configured to cost."""
    g = _regex_rule(action="mask")
    res = g.execute("nothing sensitive here")
    assert res.passed is True
    assert res.action_taken == "none"
    assert halts(g, res) is False


def test_action_defaults_to_block_when_the_key_is_absent() -> None:
    """A rule from a plane that predates wire v1.9 sends no ``action`` at all."""
    assert coerce_action(None) == "block"
    assert Guardrail(name="x").action == "block"


def test_an_unrecognised_stored_action_fails_closed_to_block() -> None:
    """An action this build has never heard of must not become a silent pass."""
    assert coerce_action("teleport") == "block"
    g = _regex_rule(action="teleport")
    assert g.action == "block"
    res = g.execute(DIRTY)
    assert res.action_taken == "blocked"
    assert halts(g, res) is True


# --------------------------------------------------------------------------- #
# warn — evaluated inline, recorded, does not stop the run
# --------------------------------------------------------------------------- #
def test_warn_records_the_failure_and_lets_the_payload_through() -> None:
    g = _regex_rule(action="warn")
    outcome = _run([g])

    res = outcome.results[0]
    # The evidence stays honest: this is still a failure, it just doesn't halt.
    assert res.passed is False
    assert res.action_taken == "warned"
    assert halts(g, res) is False
    assert outcome.data == DIRTY
    assert outcome.modified is False


def test_warn_is_not_the_same_thing_as_observe_only() -> None:
    """``blocking=False`` means "run concurrently, don't wait"; ``warn`` still runs inline."""
    warn_rule = _regex_rule(action="warn")
    observe_rule = _regex_rule(blocking=False)
    assert warn_rule.blocking is True
    assert observe_rule.action == "block"
    assert halts(warn_rule, warn_rule.execute(DIRTY)) is False
    assert halts(observe_rule, observe_rule.execute(DIRTY)) is False


# --------------------------------------------------------------------------- #
# mask — redact the offending span and continue
# --------------------------------------------------------------------------- #
def test_mask_redacts_the_matched_span_and_continues() -> None:
    g = _regex_rule(action="mask")
    outcome = _run([g])

    res = outcome.results[0]
    assert res.action_taken == "masked"
    assert res.modified_data == "call me on [REDACTED] tomorrow"
    assert outcome.data == "call me on [REDACTED] tomorrow"
    assert outcome.modified is True
    assert halts(g, res) is False


def test_mask_token_is_configurable_and_inserted_literally() -> None:
    r"""A token containing ``\1`` must be inserted, not expanded as a backreference."""
    g = _regex_rule(action="mask", config={"mask_token": r"<\1>"})
    res = g.execute(DIRTY)
    assert res.modified_data == r"call me on <\1> tomorrow"


def test_a_mask_that_finds_no_span_degrades_to_a_block() -> None:
    """``should_match=True`` fails when the pattern is *absent* — nothing to redact."""
    g = _regex_rule(action="mask", config={"should_match": True})
    res = g.execute("nothing sensitive here")
    assert res.action_taken == "blocked"
    assert res.modified_data is None
    assert "blocked instead" in (res.message or "")
    with pytest.raises(GuardrailBlockedError):
        _run([g], "nothing sensitive here")


def test_classifier_masks_the_matched_keywords() -> None:
    g = Guardrail(
        name="no-darn",
        guardrail_type=GuardrailType.classifier,
        position=GuardrailPosition.output,
        config={"categories": {"profanity": ["darn"]}, "blocked": ["profanity"]},
        action="mask",
    )
    res = g.execute("well Darn it")
    assert res.action_taken == "masked"
    assert res.modified_data == "well [REDACTED] it"


def test_mask_on_a_judge_which_returns_only_a_verdict_degrades_to_a_block() -> None:
    """The SDK never receives one — the plane refuses to store it — but a local
    guardrail can still ask, and a verdict-only type cannot locate a span."""
    g = Guardrail(
        name="judge",
        guardrail_type=GuardrailType.llm_judge,
        position=GuardrailPosition.output,
        action="mask",
    )
    assert asyncio.run(mask_payload(g, DIRTY)) is None


# --------------------------------------------------------------------------- #
# override — substitute the operator's copy for the payload
# --------------------------------------------------------------------------- #
def test_override_substitutes_operator_copy_for_the_payload() -> None:
    g = _regex_rule(action="override", config={"override_message": "I can't share that."})
    outcome = _run([g])

    res = outcome.results[0]
    assert res.action_taken == "overridden"
    assert res.modified_data == "I can't share that."
    assert outcome.data == "I can't share that."
    assert halts(g, res) is False


def test_override_without_copy_falls_back_to_the_tripwire_message() -> None:
    g = _regex_rule(action="override", config={"tripwire_message": "SSN found"})
    assert g.execute(DIRTY).modified_data == "SSN found"


# --------------------------------------------------------------------------- #
# reask — the one action the plane cannot perform
# --------------------------------------------------------------------------- #
def test_reask_is_recorded_and_halts_wherever_there_is_no_loop_to_re_drive() -> None:
    """``halts`` blocks a reask; only the agent's output path opts out of that."""
    g = _regex_rule(action="reask")
    res = g.execute(DIRTY)
    assert res.action_taken == "reask"
    assert halts(g, res) is True
    with pytest.raises(GuardrailBlockedError):
        _run([g])


def test_the_executor_reports_a_reask_so_a_loop_owner_can_act_on_it() -> None:
    g = _regex_rule(action="reask", blocking=False)
    outcome = _run([g])
    assert outcome.reask is None, "an observe-only rule never asks for a re-ask to be honoured"

    blocking = _regex_rule(action="reask")
    with pytest.raises(GuardrailBlockedError):
        _run([blocking])


# --------------------------------------------------------------------------- #
# The safety rules that fall out of branching on action_taken
# --------------------------------------------------------------------------- #
def test_an_errored_check_blocks_whatever_the_action_says() -> None:
    """Nothing is known about the payload, so there is nothing to mask and
    nothing to warn about with confidence."""

    def _boom(_text: str) -> bool:
        raise RuntimeError("detector down")

    for action in ("warn", "mask", "override", "reask"):
        g = Guardrail(name="boom", fn=_boom, action=action, position=GuardrailPosition.output)
        res = g.execute(DIRTY)
        assert res.errored is True
        assert res.action_taken == "blocked", action
        assert res.modified_data is None
        assert halts(g, res) is True


def test_an_errored_check_that_fails_open_still_reports_no_action() -> None:
    """``on_error="allow"`` decides whether the error is a failure at all; the
    action spectrum only speaks about failures."""

    def _boom(_text: str) -> bool:
        raise RuntimeError("detector down")

    g = Guardrail(name="boom", fn=_boom, action="warn", on_error="allow")
    res = g.execute(DIRTY)
    assert res.passed is True
    assert res.action_taken == "none"


@pytest.mark.parametrize("action", ACTIONS)
def test_an_observe_only_rule_never_halts_whatever_its_action(action: str) -> None:
    g = _regex_rule(action=action, blocking=False)
    outcome = _run([g])
    res = outcome.results[0]
    # Evidence is still produced — it just doesn't stop anything.
    assert res.passed is False
    assert halts(g, res) is False
    assert outcome.modified is False, "an observe-only rule does not change the run"


# --------------------------------------------------------------------------- #
# Sequencing: what the next rule sees
# --------------------------------------------------------------------------- #
def test_a_masked_payload_is_what_the_next_rule_sees() -> None:
    masker = _regex_rule(name="mask-ssn", action="mask")
    checker = _regex_rule(name="no-ssn-at-all")
    # The second rule would block the original payload; it must see the masked one.
    outcome = _run([masker, checker])
    assert outcome.data == "call me on [REDACTED] tomorrow"
    assert [r.action_taken for r in outcome.results] == ["masked", "none"]


def test_a_non_blocking_rule_records_a_rewrite_but_never_applies_it() -> None:
    observer = _regex_rule(action="mask", blocking=False)
    outcome = _run([observer])
    assert outcome.results[0].action_taken == "masked"
    assert outcome.results[0].modified_data == "call me on [REDACTED] tomorrow"
    assert outcome.data == DIRTY
    assert outcome.modified is False


# --------------------------------------------------------------------------- #
# Serialization, severity, floor
# --------------------------------------------------------------------------- #
def test_the_three_new_fields_round_trip() -> None:
    g = _regex_rule(action="warn", severity="high", floor=True)
    data = g.to_dict()
    assert data["action"] == "warn"
    assert data["severity"] == "high"
    assert data["floor"] is True

    restored = Guardrail.from_dict(data)
    assert (restored.action, restored.severity, restored.floor) == ("warn", "high", True)


def test_an_unrecognised_severity_is_unset_rather_than_invented() -> None:
    assert coerce_severity("catastrophic") is None
    assert coerce_severity(None) is None
    assert coerce_severity("critical") == "critical"
    assert _regex_rule(severity="spicy").severity is None


def test_severity_and_floor_change_no_enforcement() -> None:
    plain = _regex_rule(action="warn")
    loud = _regex_rule(action="warn", severity="critical", floor=True)
    assert halts(plain, plain.execute(DIRTY)) == halts(loud, loud.execute(DIRTY))


# --------------------------------------------------------------------------- #
# What the span and the Local UI report
# --------------------------------------------------------------------------- #
class _Collector(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict, str]] = []

    def export(self, spans):  # type: ignore[override]
        for s in spans:
            self.spans.append((s.name, dict(s.attributes), s.status.status_code.name))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


@pytest.fixture()
def collector() -> _Collector:
    col = _Collector()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(col))
    return col


def test_the_span_carries_what_the_action_did(collector: _Collector) -> None:
    _run([_regex_rule(action="mask", severity="medium", floor=True)])
    attrs = next(a for n, a, _ in collector.spans if n == "guardrail.no-ssn")
    assert attrs["fastaiagent.guardrail.action"] == "mask"
    assert attrs["fastaiagent.guardrail.action_taken"] == "masked"
    assert attrs["fastaiagent.guardrail.severity"] == "medium"
    assert attrs["fastaiagent.guardrail.floor"] is True
    # The checks vocabulary the plane parses is deliberately unchanged.
    assert json.loads(attrs["fastaiagent.guardrail.checks"])[0]["result"] == "block"


def test_the_ui_outcome_distinguishes_a_rewrite_from_a_block() -> None:
    masker = _regex_rule(action="mask")
    assert _outcome(masker, masker.execute(DIRTY)) == "filtered"

    warner = _regex_rule(action="warn")
    assert _outcome(warner, warner.execute(DIRTY)) == "warned"

    blocker = _regex_rule()
    assert _outcome(blocker, blocker.execute(DIRTY)) == "blocked"

    observer = _regex_rule(blocking=False)
    assert _outcome(observer, observer.execute(DIRTY)) == "warned"


# --------------------------------------------------------------------------- #
# Foreign-framework proxies
# --------------------------------------------------------------------------- #
def test_a_harness_honours_warn_but_blocks_what_it_cannot_do() -> None:
    warner = _regex_rule(action="warn")
    assert harness_halts(warner, warner.execute(DIRTY)) is None

    for action, needle in (
        ("mask", "cannot do"),
        ("override", "cannot do"),
        ("reask", "agent loop"),
    ):
        g = _regex_rule(action=action, config={"override_message": "no"})
        reason = harness_halts(g, g.execute(DIRTY))
        assert reason is not None and needle in reason, action

    clean = _regex_rule(action="mask")
    assert harness_halts(clean, clean.execute("nothing here")) is None
