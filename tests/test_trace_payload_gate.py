"""``FASTAIAGENT_TRACE_PAYLOADS`` honours every documented false spelling.

``docs/configuration/environment-variables.md`` has always said booleans accept
``1/true/yes/on`` "and their negatives". The gate compared against the literal
``"0"``, so an operator who set ``false`` — the spelling the docs' own prose
invites — kept egressing prompts, completions, tool arguments and chain state
with no signal at all. This exercises the real filter, not the predicate.

CLAUDE.md §2.5: a span has three content channels and all three are gated, so
each one is driven here.
"""

from __future__ import annotations

import pytest

from fastaiagent._internal.env import FALSE_VALUES, TRUE_VALUES
from fastaiagent.trace.redaction import (
    SENSITIVE_ATTR_KEYS,
    apply_event_export_policy,
    apply_export_policy,
)

CANARY = "CANARY-payload-gate-9f8e7d"

_FALSE_SPELLINGS = sorted(FALSE_VALUES) + ["FALSE", "Off", "  no  ", "\tOFF\n"]
_TRUE_SPELLINGS = sorted(TRUE_VALUES) + ["TRUE", "On", "  yes  "]

_PAYLOAD_ATTRS = {
    "agent.input": f"in {CANARY}",
    "agent.output": f"out {CANARY}",
    "agent.system_prompt": f"sys {CANARY}",
    "gen_ai.request.messages": f"msgs {CANARY}",
    "gen_ai.response.content": f"resp {CANARY}",
    "tool.args": f"args {CANARY}",
    "tool.result": f"result {CANARY}",
    "chain.input": f"chain-in {CANARY}",
    # The five foreign-OTel normalization targets added in 1.67.0.
    "gen_ai.prompt": f"prompt {CANARY}",
    "gen_ai.completion": f"completion {CANARY}",
    "gen_ai.response.text": f"text {CANARY}",
    "fastaiagent.gen_ai.prompt": f"ns-prompt {CANARY}",
    "fastaiagent.gen_ai.response.text": f"ns-text {CANARY}",
}

_STRUCTURAL_ATTRS = {
    "agent.name": "support-bot",
    "gen_ai.request.model": "gpt-4o",
    "gen_ai.usage.input_tokens": 120,
    "fastaiagent.runner.type": "agent",
}


@pytest.mark.parametrize("value", _FALSE_SPELLINGS)
def test_false_spellings_drop_every_payload_attribute(value, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", value)
    out = apply_export_policy({**_PAYLOAD_ATTRS, **_STRUCTURAL_ATTRS})

    leaked = sorted(k for k in _PAYLOAD_ATTRS if k in out)
    assert not leaked, f"FASTAIAGENT_TRACE_PAYLOADS={value!r} still exported {leaked}"
    assert CANARY not in repr(out)
    # Structural metadata is deliberately unaffected — the gate is not a mute.
    for key, expected in _STRUCTURAL_ATTRS.items():
        assert out[key] == expected


@pytest.mark.parametrize("value", _TRUE_SPELLINGS)
def test_true_spellings_keep_payloads(value, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", value)
    out = apply_export_policy(dict(_PAYLOAD_ATTRS))
    assert out == _PAYLOAD_ATTRS


def test_unset_keeps_payloads(monkeypatch):
    monkeypatch.delenv("FASTAIAGENT_TRACE_PAYLOADS", raising=False)
    assert apply_export_policy(dict(_PAYLOAD_ATTRS)) == _PAYLOAD_ATTRS


def test_unparseable_value_fails_closed(monkeypatch):
    """A typo on an egress opt-out must not egress. Signed off for 1.67.0."""
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "flase")
    out = apply_export_policy(dict(_PAYLOAD_ATTRS))
    assert out == {}


@pytest.mark.parametrize("value", _FALSE_SPELLINGS)
def test_false_spellings_gate_the_event_channel_too(value, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", value)
    events = [
        {
            "name": "exception",
            "attributes": {
                "exception.message": f"blocked: {CANARY}",
                "exception.type": "GuardrailBlockedError",
            },
        }
    ]
    out = apply_event_export_policy(events)
    assert CANARY not in repr(out), f"the event channel leaked with value {value!r}"


@pytest.mark.parametrize("value", _FALSE_SPELLINGS)
def test_false_spellings_gate_the_status_channel_too(value, monkeypatch):
    from fastaiagent.trace.otel import _filtered_status

    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", value)

    class _Status:
        status_code = "ERROR"
        description = f"guardrail blocked: {CANARY}"

    class _Span:
        status = _Status()

    filtered = _filtered_status(_Span())
    assert filtered is not None, f"status was not filtered for {value!r}"
    assert CANARY not in repr(getattr(filtered, "description", ""))


def test_the_five_normalized_keys_are_registered():
    """Foreign-OTel spans carry prompts on their own keys (see trace.normalize).

    Behavioural cover lives in the parametrized cases above; this states the
    registry membership the §2.5 contract requires, so a future normalization
    target has an obvious place to be added.
    """
    for key in (
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.response.text",
        "fastaiagent.gen_ai.prompt",
        "fastaiagent.gen_ai.response.text",
    ):
        assert key in SENSITIVE_ATTR_KEYS


def test_normalized_foreign_span_is_gated_end_to_end(monkeypatch):
    """The real path: normalize a foreign span, then export it with payloads off."""
    from fastaiagent.trace.normalize import normalize_attributes

    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "off")
    foreign = {
        "openinference.span.kind": "LLM",
        "input.value": f"user asked {CANARY}",
        "output.value": f"model said {CANARY}",
        "llm.model_name": "gpt-4o",
    }
    normalized = normalize_attributes(foreign)
    assert CANARY in repr(normalized), "normalization did not carry the payload forward"

    exported = apply_export_policy(normalized)
    assert CANARY not in repr(exported), (
        "an integration-captured span still egressed its prompt/completion"
    )
    assert exported.get("gen_ai.request.model") == "gpt-4o"


def test_local_capture_is_unaffected_and_search_still_matches(tmp_path, monkeypatch):
    """§2.5: filtering happens on the way out; local.db stays full fidelity.

    The local UI's trace search indexes ``gen_ai.prompt``/``gen_ai.response.text``
    (``ui.db._FTS_INPUT_KEYS`` / ``_FTS_OUTPUT_KEYS``), so adding them to the
    export registry must not make them unsearchable.
    """
    import json

    from fastaiagent._internal.config import reset_config
    from fastaiagent.ui.db import init_local_db
    from fastaiagent.ui.routes.traces import _fts_query

    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "off")
    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()

    attrs = {
        "gen_ai.prompt": f"user asked {CANARY}",
        "gen_ai.response.text": f"model said {CANARY}",
    }
    db = init_local_db(str(db_path))
    try:
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)""",
            (
                "s1",
                "t1",
                "llm.chat",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
                "OK",
                json.dumps(attrs),
                "[]",
            ),
        )
        # Local capture kept the payload verbatim ...
        stored = db.fetchone("SELECT attributes FROM spans WHERE span_id = 's1'")
        assert CANARY in stored["attributes"]
        # ... and the UI's FTS search — which indexes exactly these two keys —
        # still matches it.
        hits = db.fetchall(
            "SELECT spans.span_id FROM spans "
            "JOIN span_fts ON span_fts.span_id = spans.span_id "
            "WHERE span_fts MATCH ?",
            (_fts_query(CANARY),),
        )
    finally:
        db.close()
        reset_config()

    assert hits, "local trace search stopped matching prompts after the registry change"
