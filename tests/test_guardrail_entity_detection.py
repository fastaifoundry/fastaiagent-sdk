"""The ``pii`` and ``secrets`` check types — entity detection from a typed config.

No mocks and no LLM: these detectors are pure regex (Presidio is a local model),
so everything here runs against the real code.

The mirror direction is **reversed** for these two. ``topics.py``,
``hazard_taxonomy.py`` and ``grounding.py`` were authored on the plane and copied
here; ``_internal/safety_detectors.py`` is ours and the plane's
``app/agents/services/detectors.py`` is the copy, pinned by their
``backend/tests/test_detectors_mirror.py``. So a change to a pattern — or to
``mask_spans`` — breaks their build on purpose, and the metadata shapes asserted
below are the ones their ``summarize_pii`` / ``summarize_secrets`` persist.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from fastaiagent._internal.safety_detectors import detect_pii, detect_secrets, mask_spans
from fastaiagent.guardrail.actions import MASKABLE_TYPES, mask_payload
from fastaiagent.guardrail.executor import EXPORTABLE_DETAIL_KEYS, _exportable_detail
from fastaiagent.guardrail.from_policy import guardrail_from_policy_rule
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailType

SSN = "123-45-6789"
DIRTY = f"Contact dana@example.com or {SSN} about the invoice"
# The real overlap: this hits `openai_api_key` AND `generic_secret`, and the
# generic span covers the whole assignment including the quotes.
LEAKED = 'deploy with api_key = "sk-proj-abcdefghijklmnopqrstuvwxyz123456" now'


def _rule(impl: str, **kwargs: Any) -> Guardrail:
    return Guardrail(
        name=f"e-{impl}",
        guardrail_type=GuardrailType(impl),
        config=kwargs.pop("config", {}),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Rebuild from a plane rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("impl", "config"),
    [
        ("pii", {"entities": ["email", "ssn"], "backend": "regex"}),
        ("pii", {}),  # absent `entities` means the default four
        ("secrets", {}),  # `secrets` legitimately carries no config at all
        ("secrets", {"mask_token": "[GONE]"}),
    ],
)
def test_both_types_reconstruct_from_a_policy_rule(impl: str, config: dict) -> None:
    """Until now these arrived at the edge and were skipped at debug level, while
    the console listed them as edge-enforceable."""
    g = guardrail_from_policy_rule(
        {
            "name": f"plane-{impl}",
            "implementation_type": impl,
            "guardrail_type": "output",
            "validation_mode": "blocking",
            "config": config,
            "on_error": "block",
            "action": "block",
            "agent_ids": [],
        }
    )
    assert g is not None, f"{impl} should no longer be skipped"
    assert g.guardrail_type.value == impl
    assert g.config == config
    assert g.origin == "plane"


def test_a_pii_rule_with_no_entities_scans_the_default_four() -> None:
    res = _rule("pii").execute(DIRTY)
    assert res.errored is False
    assert res.metadata["entities"] == ["email", "phone", "ssn", "credit_card"]
    assert res.metadata["found"] == ["email", "ssn"]


def test_a_secrets_rule_needs_no_config() -> None:
    """A tenant narrowing a credential detector is a tenant weakening it, so
    there is no entity list to get wrong."""
    res = _rule("secrets").execute(LEAKED)
    assert res.errored is False and res.passed is False
    assert "openai_api_key" in res.metadata["found"]


# --------------------------------------------------------------------------- #
# Fail loud — a detection control whose absence reports success is worse than none
# --------------------------------------------------------------------------- #
def test_an_unknown_entity_errors_rather_than_scanning_for_less() -> None:
    res = _rule("pii", config={"entities": ["emial"]}).execute(DIRTY)
    assert res.errored is True
    assert res.passed is False  # fails closed by default
    assert "Unknown PII entity" in (res.message or "")


@pytest.mark.parametrize("backend", ["regex", "presidio"])
def test_an_unknown_entity_errors_under_either_backend(backend: str) -> None:
    """The regression this release fixes. ``detect_pii`` validated entity names
    only inside the regex branch, so ``presidio`` dropped unknown names silently
    — and with *every* name unknown the mapped list came out empty, which
    Presidio reads as "scan for everything". A typo widened the rule instead of
    failing it.
    """
    with pytest.raises(ValueError, match="Unknown PII entity"):
        detect_pii(DIRTY, entities=("emial",), backend=backend)


def test_an_unknown_backend_errors_rather_than_falling_back() -> None:
    res = _rule("pii", config={"backend": "presidoi"}).execute(DIRTY)
    assert res.errored is True
    assert "Unknown PII backend" in (res.message or "")


def test_presidio_without_the_extra_errors_rather_than_finding_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reported, not swallowed: a missing optional dependency must not read as a
    clean payload."""
    import builtins as _builtins

    real_import = _builtins.__import__

    def _blocked(name: str, *a: Any, **k: Any) -> Any:
        if name.startswith("presidio_analyzer"):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(_builtins, "__import__", _blocked)

    res = _rule("pii", config={"backend": "presidio"}).execute(DIRTY)
    assert res.errored is True
    assert "safety" in (res.message or "")


def test_a_non_list_entities_value_is_refused() -> None:
    res = _rule("pii", config={"entities": {"email": True}}).execute(DIRTY)
    assert res.errored is True
    assert "must be a list" in (res.message or "")


def test_a_single_entity_string_is_accepted() -> None:
    res = _rule("pii", config={"entities": "ssn"}).execute(DIRTY)
    assert res.errored is False
    assert res.metadata["found"] == ["ssn"]  # the email is not scanned for


# --------------------------------------------------------------------------- #
# Masking — the half only the edge can do
# --------------------------------------------------------------------------- #
def test_pii_masking_redacts_the_spans_and_keeps_the_rest() -> None:
    g = _rule("pii", action="mask")
    res = g.execute(DIRTY)
    assert res.action_taken == "masked"
    assert res.modified_data == "Contact [REDACTED] or [REDACTED] about the invoice"
    assert SSN not in (res.modified_data or "")


def test_overlapping_credential_patterns_merge_into_one_redaction() -> None:
    """One secret routinely matches two patterns. Replacing the ranges
    independently corrupts the output and leaves fragments of the very value
    being redacted."""
    kinds = {m.kind for m in detect_secrets(LEAKED)}
    assert {"openai_api_key", "generic_secret"} <= kinds, "the overlap fixture stopped overlapping"

    res = _rule("secrets", action="mask").execute(LEAKED)
    assert res.action_taken == "masked"
    # One token, not two, and no fragment of the key survives.
    assert res.modified_data == "deploy with [REDACTED] now"
    assert "sk-proj" not in (res.modified_data or "")


def test_the_mask_token_is_configurable() -> None:
    res = _rule("secrets", config={"mask_token": "<gone>"}, action="mask").execute(LEAKED)
    assert res.modified_data == "deploy with <gone> now"


def test_replacement_is_right_to_left_so_a_longer_token_does_not_shift_spans() -> None:
    """Every offset was computed against the original string."""
    text = "a@b.com then c@d.com"
    spans = [(m.start, m.end) for m in detect_pii(text, entities=("email",))]
    out = mask_spans(text, spans, "[A-MUCH-LONGER-TOKEN]")
    assert out == "[A-MUCH-LONGER-TOKEN] then [A-MUCH-LONGER-TOKEN]"


def test_a_clean_payload_masks_nothing_and_passes() -> None:
    res = _rule("pii", action="mask").execute("nothing sensitive in here")
    assert res.passed is True and res.action_taken == "none"
    assert res.modified_data is None


def test_mask_payload_refuses_a_type_outside_the_maskable_set() -> None:
    """``MASKABLE_TYPES`` was documentation-only until this release — it named a
    rule nothing enforced. It is now the gate."""
    assert set(MASKABLE_TYPES) == {
        GuardrailType.regex,
        GuardrailType.classifier,
        GuardrailType.pii,
        GuardrailType.secrets,
    }
    judge = Guardrail(name="j", guardrail_type=GuardrailType.llm_judge, config={"prompt": "x"})
    assert asyncio.run(mask_payload(judge, "anything")) is None


# --------------------------------------------------------------------------- #
# The §4 requirement: what may leave the machine
# --------------------------------------------------------------------------- #
def test_the_result_reports_counts_never_values() -> None:
    """``PIIMatch.value`` holds the matched text — correct in-process, where
    masking needs it, and unacceptable anywhere durable."""
    res = _rule("pii").execute(DIRTY)
    blob = json.dumps(res.metadata)
    assert SSN not in blob
    assert "dana@example.com" not in blob
    assert res.metadata == {
        "backend": "regex",
        "entities": ["email", "phone", "ssn", "credit_card"],
        "found": ["email", "ssn"],
        "counts": {"email": 1, "ssn": 1},
        "total": 2,
    }


def test_the_secrets_result_keeps_not_even_the_masked_form() -> None:
    res = _rule("secrets").execute(LEAKED)
    assert set(res.metadata) == {"found", "counts", "total"}
    assert "sk-proj" not in json.dumps(res.metadata)


def test_the_exported_detail_is_the_planes_row_shape() -> None:
    assert EXPORTABLE_DETAIL_KEYS[GuardrailType.pii] == frozenset(
        {"backend", "entities", "found", "counts", "total"}
    )
    assert EXPORTABLE_DETAIL_KEYS[GuardrailType.secrets] == frozenset({"found", "counts", "total"})


def test_no_raw_value_reaches_a_guardrail_span() -> None:
    """The test that matters most. Whatever lands on the span reaches a control
    plane's durable, tenant-visible execution row — so the control that *finds*
    personal data must not become a standing database of it."""
    g = _rule("pii")
    res = g.execute(DIRTY)
    detail = _exportable_detail(g, res)
    assert detail is not None
    blob = json.dumps(detail)
    assert SSN not in blob and "dana@example.com" not in blob
    assert detail["counts"] == {"email": 1, "ssn": 1}
