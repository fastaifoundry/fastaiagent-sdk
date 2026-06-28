"""Tests for trace -> eval-dataset curation (``fastaiagent.eval.curate``).

No mocking: every test writes **real rows** into a real temp SQLite DB (via the
real ``SQLiteHelper``) and curates them back, and one test runs a **real Agent**
(offline ``TestModel``) so the ``agent.input``/``agent.output`` contract is
exercised end-to-end. The ``isolated_local_db`` fixture points
``FASTAIAGENT_LOCAL_DB`` at a temp file so curation reads the same DB.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from fastaiagent._internal.config import get_config
from fastaiagent.eval import Dataset, curate_from_traces, evaluate
from fastaiagent.ui.db import init_local_db


def _open_db() -> Any:
    return init_local_db(get_config().local_db_path)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _span(
    db: Any,
    trace_id: str,
    name: str,
    attrs: dict[str, Any],
    *,
    parent: str | None = None,
    status: str = "OK",
    ts: str | None = None,
) -> None:
    db.execute(
        "INSERT INTO spans (span_id, trace_id, parent_span_id, name, start_time, "
        "end_time, status, attributes, events) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            uuid.uuid4().hex,
            trace_id,
            parent,
            name,
            ts or _now(),
            ts or _now(),
            status,
            json.dumps(attrs),
            "[]",
        ),
    )


def _agent_attrs(name: str, inp: str, out: str, **extra: Any) -> dict[str, Any]:
    a = {"agent.name": name, "agent.input": inp, "agent.output": out}
    a.update(extra)
    return a


# --------------------------------------------------------------------------- #
# Selection + extraction
# --------------------------------------------------------------------------- #


def test_all_includes_root_and_nested_agent_spans(isolated_local_db: Any) -> None:
    db = _open_db()
    t_plain = uuid.uuid4().hex
    _span(db, t_plain, "agent.support", _agent_attrs("support", "refund window?", "30 days"))
    # A chain trace whose root is chain.* but which contains a nested agent span.
    t_chain = uuid.uuid4().hex
    _span(db, t_chain, "chain.flow", {"chain.input": "x", "chain.output": "y"})
    _span(db, t_chain, "agent.worker", _agent_attrs("worker", "sub task", "sub answer"), parent="r")
    db.close()

    items = curate_from_traces(filter="all")
    inputs = sorted(it["input"] for it in items)
    assert inputs == ["refund window?", "sub task"]  # nested agent span IS curated
    for it in items:
        assert it["trace_id"] and it["span_id"] and it["expected_output"]
        assert not it.get("needs_review")


def test_favorites_filter(isolated_local_db: Any) -> None:
    db = _open_db()
    fav = uuid.uuid4().hex
    other = uuid.uuid4().hex
    _span(db, fav, "agent.a", _agent_attrs("a", "keep me", "good"))
    _span(db, other, "agent.a", _agent_attrs("a", "skip me", "good"))
    db.execute("INSERT INTO trace_favorites (trace_id, created_at) VALUES (?, ?)", (fav, _now()))
    db.close()

    items = curate_from_traces(filter="favorites")
    assert [it["input"] for it in items] == ["keep me"]


def test_noted_filter_attaches_note(isolated_local_db: Any) -> None:
    db = _open_db()
    tid = uuid.uuid4().hex
    _span(db, tid, "agent.a", _agent_attrs("a", "q", "out"))
    db.execute(
        "INSERT INTO trace_notes (trace_id, note, updated_at) VALUES (?, ?, ?)",
        (tid, "look into this", _now()),
    )
    db.close()

    items = curate_from_traces(filter="noted")
    assert len(items) == 1
    assert items[0]["note"] == "look into this"


def test_guardrail_filter_marks_needs_review(isolated_local_db: Any) -> None:
    db = _open_db()
    tid = uuid.uuid4().hex
    _span(db, tid, "agent.support", _agent_attrs("support", "share pii", "email: a@b.com"))
    db.execute(
        "INSERT INTO guardrail_events (event_id, trace_id, guardrail_name, outcome, "
        "message, timestamp) VALUES (?,?,?,?,?,?)",
        (uuid.uuid4().hex, tid, "no_pii", "fail", "PII detected", _now()),
    )
    db.close()

    items = curate_from_traces(filter="guardrail")
    assert len(items) == 1
    it = items[0]
    assert it["needs_review"] is True
    assert it["expected_output"] == ""
    assert it["actual_output"] == "email: a@b.com"
    assert "no_pii" in it["reason"] and "PII detected" in it["reason"]


def test_failed_filter_uses_error_status(isolated_local_db: Any) -> None:
    db = _open_db()
    ok = uuid.uuid4().hex
    bad = uuid.uuid4().hex
    _span(db, ok, "agent.a", _agent_attrs("a", "fine", "ok"))
    _span(db, bad, "agent.a", _agent_attrs("a", "boom", "partial"))
    _span(db, bad, "tool.x", {"k": "v"}, parent="r", status="ERROR")
    db.close()

    items = curate_from_traces(filter="failed")
    assert [it["input"] for it in items] == ["boom"]
    assert items[0]["needs_review"] is True


def test_agent_filter(isolated_local_db: Any) -> None:
    db = _open_db()
    tid = uuid.uuid4().hex
    _span(db, tid, "agent.alpha", _agent_attrs("alpha", "a-in", "a-out"))
    _span(db, tid, "agent.beta", _agent_attrs("beta", "b-in", "b-out"), parent="r")
    db.close()

    items = curate_from_traces(filter="all", agent="beta")
    assert [it["input"] for it in items] == ["b-in"]


def test_since_and_limit(isolated_local_db: Any) -> None:
    db = _open_db()
    old = uuid.uuid4().hex
    recent1 = uuid.uuid4().hex
    recent2 = uuid.uuid4().hex
    long_ago = (datetime.now(tz=timezone.utc) - timedelta(hours=48)).isoformat()
    _span(db, old, "agent.a", _agent_attrs("a", "old", "o"), ts=long_ago)
    _span(db, recent1, "agent.a", _agent_attrs("a", "r1", "o"), ts=_now())
    _span(db, recent2, "agent.a", _agent_attrs("a", "r2", "o"), ts=_now())
    db.close()

    assert {it["input"] for it in curate_from_traces(filter="all", since_hours=1)} == {"r1", "r2"}
    assert len(curate_from_traces(filter="all", limit=1)) == 1  # capped


def test_dedup_by_input(isolated_local_db: Any) -> None:
    db = _open_db()
    for _ in range(3):
        _span(db, uuid.uuid4().hex, "agent.a", _agent_attrs("a", "same question", "ans"))
    db.close()

    assert len(curate_from_traces(filter="all")) == 3
    assert len(curate_from_traces(filter="all", dedup_by="input")) == 1


def test_skips_non_agent_and_empty_input(isolated_local_db: Any) -> None:
    db = _open_db()
    non_agent = uuid.uuid4().hex
    empty = uuid.uuid4().hex
    _span(db, non_agent, "chain.flow", {"chain.input": "x", "chain.output": "y"})
    _span(db, empty, "agent.a", _agent_attrs("a", "   ", "out"))  # blank input
    db.close()

    assert curate_from_traces(filter="all") == []


def test_multimodal_marks_needs_review(isolated_local_db: Any) -> None:
    db = _open_db()
    tid = uuid.uuid4().hex
    _span(
        db,
        tid,
        "agent.vision",
        _agent_attrs("vision", "What letters?", "CAT", **{"fastaiagent.input.media_count": 1}),
    )
    db.close()

    item = curate_from_traces(filter="all")[0]
    assert item["needs_review"] is True
    assert "multimodal" in item["reason"]


def test_mark_output_as_expected_override(isolated_local_db: Any) -> None:
    db = _open_db()
    tid = uuid.uuid4().hex
    _span(db, tid, "agent.support", _agent_attrs("support", "q", "captured answer"))
    db.execute(
        "INSERT INTO guardrail_events (event_id, trace_id, guardrail_name, outcome, "
        "message, timestamp) VALUES (?,?,?,?,?,?)",
        (uuid.uuid4().hex, tid, "g", "fail", "m", _now()),
    )
    db.close()

    # Override the guardrail default: treat the captured output as gold.
    item = curate_from_traces(filter="guardrail", mark_output_as_expected=True)[0]
    assert item.get("needs_review") is None
    assert item["expected_output"] == "captured answer"


def test_explicit_trace_ids(isolated_local_db: Any) -> None:
    db = _open_db()
    keep = uuid.uuid4().hex
    drop = uuid.uuid4().hex
    _span(db, keep, "agent.a", _agent_attrs("a", "keep", "o"))
    _span(db, drop, "agent.a", _agent_attrs("a", "drop", "o"))
    db.close()

    items = curate_from_traces(trace_ids=[keep])
    assert [it["input"] for it in items] == ["keep"]


def test_invalid_args_raise(isolated_local_db: Any) -> None:
    with pytest.raises(ValueError):
        curate_from_traces(filter="bogus")
    with pytest.raises(ValueError):
        curate_from_traces(dedup_by="bogus")
    with pytest.raises(ValueError):
        curate_from_traces(exclude_infra_errors="bogus")


# --------------------------------------------------------------------------- #
# Infrastructure-error exclusion (don't curate gold from a crashed run)
# --------------------------------------------------------------------------- #


def test_all_excludes_infra_errored_run(isolated_local_db: Any) -> None:
    """A run that produced no usable output is dropped from the gold set; a clean
    run is kept, and the drop count is surfaced on the returned dataset."""
    db = _open_db()
    clean = uuid.uuid4().hex
    errored = uuid.uuid4().hex
    _span(db, clean, "agent.a", _agent_attrs("a", "good q", "good answer"))
    # Infra-errored: agent captured input but produced NO output, and a span
    # carries an error status (corroborating the infra failure).
    _span(db, errored, "agent.a", {"agent.name": "a", "agent.input": "boom"})  # no agent.output
    _span(db, errored, "llm.call", {"k": "v"}, parent="r", status="ERROR")
    db.close()

    items = curate_from_traces(filter="all")
    assert [it["input"] for it in items] == ["good q"]  # errored run dropped
    assert items.infra_excluded == 1
    assert items.emitted == 1
    assert "1 dropped as infra-errored" in items.coverage_summary()


def test_tool_errored_but_agent_recovered_is_kept_under_agent_mode(isolated_local_db: Any) -> None:
    """The 'recovered = good signal' guarantee: a trace where a tool errored but
    the agent still produced a clean answer is KEPT under the default ("agent")
    mode, and only dropped under the strict ("trace") mode. This test exists to
    stop a future refactor from quietly turning A into B."""
    db = _open_db()
    tid = uuid.uuid4().hex
    _span(db, tid, "agent.a", _agent_attrs("a", "q", "clean recovered answer"))
    _span(db, tid, "tool.x", {"k": "v"}, parent="r", status="ERROR")  # tool failed; agent recovered
    db.close()

    # Default A — keyed on agent-output presence → the clean answer is kept.
    kept = curate_from_traces(filter="all")
    assert [it["input"] for it in kept] == ["q"]
    assert kept.infra_excluded == 0

    # Strict B — any error-status span drops the run.
    strict = curate_from_traces(filter="all", exclude_infra_errors="trace")
    assert list(strict) == []
    assert strict.infra_excluded == 1


def test_dataset_from_traces_surfaces_coverage(isolated_local_db: Any) -> None:
    """The Dataset wrapper exposes curation coverage via ``ds.curation`` so the
    drop count is visible on the headline API, not just curate_from_traces."""
    db = _open_db()
    clean = uuid.uuid4().hex
    errored = uuid.uuid4().hex
    _span(db, clean, "agent.a", _agent_attrs("a", "good q", "good answer"))
    _span(db, errored, "agent.a", {"agent.name": "a", "agent.input": "boom"})  # no output
    db.close()

    ds = Dataset.from_traces(filter="all")
    assert len(ds) == 1
    assert ds.curation is not None
    assert ds.curation.infra_excluded == 1
    assert ds.curation.emitted == 1
    # A Dataset from a plain list has no curation metadata.
    assert Dataset.from_list([{"input": "x", "expected_output": "y"}]).curation is None


# --------------------------------------------------------------------------- #
# Round-trip + real-Agent contract
# --------------------------------------------------------------------------- #


def test_to_jsonl_roundtrip_and_evaluate(isolated_local_db: Any, tmp_path: Any) -> None:
    db = _open_db()
    _span(db, uuid.uuid4().hex, "agent.bot", _agent_attrs("bot", "ping", "pong"))
    db.close()

    out = tmp_path / "cases.jsonl"
    Dataset.from_traces(filter="all").to_jsonl(out)
    reloaded = Dataset.from_jsonl(out)
    assert [it["input"] for it in reloaded] == ["ping"]

    # agent_fn is the function under evaluation (a deterministic replay fn), not a
    # mock of a dependency — evaluate() runs it for real and scores the output.
    results = evaluate(
        agent_fn=lambda _x: "pong", dataset=str(out), scorers=["exact_match"], persist=False
    )
    assert results.scores["exact_match"][0].passed

    # append adds, not overwrites
    Dataset.from_list([{"input": "extra", "expected_output": "z"}]).to_jsonl(out, append=True)
    assert len(Dataset.from_jsonl(out)) == 2


def test_real_agent_contract_with_testmodel(isolated_local_db: Any) -> None:
    """A real Agent (offline TestModel) writes agent.input/agent.output; curate reads them."""
    from fastaiagent import Agent
    from fastaiagent.testing import TestModel
    from fastaiagent.trace import otel

    otel.reset()  # rebuild the tracer against the temp DB from isolated_local_db
    try:
        agent = Agent(
            name="contractbot",
            system_prompt="be brief",
            llm=TestModel(response="the answer is 4"),
        )
        agent.run("what is 2+2?")

        items = curate_from_traces(filter="all", agent="contractbot")
        assert len(items) == 1
        it = items[0]
        assert it["input"] == "what is 2+2?"
        assert it["expected_output"] == "the answer is 4"
        assert not it.get("needs_review")
        assert it["trace_id"]
    finally:
        otel.reset()
