"""The provenance foundation: stamping, local-only export, the exclusion filter (1.89.0).

Every experiment (a replay rerun, a Playground call, a learn pass, an eval case)
used to look exactly like production. ``provenance_scope`` stamps the origin on
every span started inside it; local-only sources are written ``synced=1`` so no
exporter sends them; ``non_production_filter`` lets every reader exclude them.

Real tracing, real SQLite, real OTel exporters; no mocks.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from fastaiagent.trace.provenance import (
    LOCAL_ONLY_ATTR,
    SOURCE_ATTR,
    non_production_filter,
    provenance_scope,
)


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET", "FASTAIAGENT_EXPORT_EXPERIMENTS"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _rows(db: Any) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT trace_id, span_id, parent_span_id, name, attributes, synced FROM spans"
        ).fetchall()
    return [{**dict(r), "attributes": json.loads(r["attributes"])} for r in rows]


def _span_tree(name: str = "root", children: tuple[str, ...] = ("child",)) -> None:
    from fastaiagent.trace.otel import get_tracer

    tracer = get_tracer("test-provenance")
    with tracer.start_as_current_span(name):
        for child in children:
            with tracer.start_as_current_span(child):
                pass


def test_every_span_in_a_scope_carries_the_source(traced_db: Any) -> None:
    with provenance_scope("replay", attributes={"fastaiagent.replay.source_trace_id": "abc"}):
        _span_tree("agent.x", ("llm.openai.m", "tool.charge"))
    rows = {r["name"]: r for r in _rows(traced_db)}
    assert {r["attributes"].get(SOURCE_ATTR) for r in rows.values()} == {"replay"}
    # Root-only attributes land on the scope's root, not its children.
    assert rows["agent.x"]["attributes"].get("fastaiagent.replay.source_trace_id") == "abc"
    assert "fastaiagent.replay.source_trace_id" not in rows["llm.openai.m"]["attributes"]


def test_the_innermost_scope_wins(traced_db: Any) -> None:
    from fastaiagent.trace.otel import get_tracer

    tracer = get_tracer("test-provenance")
    with provenance_scope("eval"), tracer.start_as_current_span("eval.case"):
        with provenance_scope("replay"), tracer.start_as_current_span("agent.inner"):
            pass
    rows = {r["name"]: r["attributes"].get(SOURCE_ATTR) for r in _rows(traced_db)}
    assert rows == {"eval.case": "eval", "agent.inner": "replay"}


def test_no_scope_means_no_stamp(traced_db: Any, trace_export_on: None) -> None:
    _span_tree()
    assert all(SOURCE_ATTR not in r["attributes"] for r in _rows(traced_db))
    assert all(r["synced"] == 0 for r in _rows(traced_db))


@pytest.mark.parametrize("source", ["replay", "playground", "kb_search"])
def test_a_local_only_source_is_written_as_already_sent(
    traced_db: Any, trace_export_on: None, source: str
) -> None:
    with provenance_scope(source):
        _span_tree()
    rows = _rows(traced_db)
    assert rows and all(r["synced"] == 1 for r in rows), "a local-only span would be exported"
    assert all(r["attributes"].get(LOCAL_ONLY_ATTR) is True for r in rows)


@pytest.mark.parametrize("source", ["learn", "eval", "simulation", "optimize", "batch-job"])
def test_other_sources_are_still_exported(
    traced_db: Any, trace_export_on: None, source: str
) -> None:
    with provenance_scope(source):
        _span_tree()
    assert all(r["synced"] == 0 for r in _rows(traced_db))


@pytest.mark.parametrize("how", ["scope", "env"])
def test_exporting_experiments_is_an_opt_in(
    traced_db: Any, trace_export_on: None, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    if how == "env":
        monkeypatch.setenv("FASTAIAGENT_EXPORT_EXPERIMENTS", "1")
    with provenance_scope("replay", export=True if how == "scope" else None):
        _span_tree()
    rows = _rows(traced_db)
    assert all(r["synced"] == 0 for r in rows)
    assert all(LOCAL_ONLY_ATTR not in r["attributes"] for r in rows)
    assert all(r["attributes"].get(SOURCE_ATTR) == "replay" for r in rows)


def test_a_third_party_exporter_never_receives_local_only_spans(traced_db: Any) -> None:
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fastaiagent.trace import otel

    sink = InMemorySpanExporter()
    otel.add_span_processor(SimpleSpanProcessor(otel._EgressFilteredExporter(sink)))
    with provenance_scope("replay"):
        _span_tree("agent.rerun")
    _span_tree("agent.real")
    names = {s.name for s in sink.get_finished_spans()}
    assert "agent.real" in names
    assert "agent.rerun" not in names and "child" in names  # the real run's child only


def test_the_filter_excludes_a_trace_by_any_of_its_spans(traced_db: Any) -> None:
    """Trace-level: a root-only stamp (Playground before 1.89.0) excludes the
    whole trace, and so does a rerun wrapped in a user's own span."""
    from fastaiagent.trace.otel import get_tracer

    tracer = get_tracer("test-provenance")
    _span_tree("agent.prod")
    with tracer.start_as_current_span("user.wrapper"):
        with provenance_scope("replay"), tracer.start_as_current_span("agent.rerun"):
            pass
    with sqlite3.connect(traced_db) as conn:
        conn.execute(
            "INSERT INTO spans (span_id, trace_id, name, attributes, events) VALUES "
            "('old', 'old-trace', 'playground.run', ?, '[]')",
            (json.dumps({SOURCE_ATTR: "playground"}),),
        )
        clause, params = non_production_filter()
        kept = {
            r[0]
            for r in conn.execute(f"SELECT DISTINCT name FROM spans WHERE 1=1 {clause}", params)
        }
        only_clause, only_params = non_production_filter(mode="only")
        experiments = {
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT name FROM spans WHERE 1=1 {only_clause}", only_params
            )
        }
    assert kept == {"agent.prod", "child"}
    assert experiments == {"user.wrapper", "agent.rerun", "playground.run"}


def test_the_filter_honours_include() -> None:
    assert non_production_filter(include=True) == ("", ())
    _, params = non_production_filter(include={"eval"})
    assert "eval" not in params and "replay" in params


def test_registration_is_suppressed_only_where_asked() -> None:
    from fastaiagent.trace.provenance import registration_suppressed

    assert registration_suppressed() is False
    with provenance_scope("replay", suppress_registration=True):
        assert registration_suppressed() is True
        with provenance_scope("eval"):
            assert registration_suppressed() is False  # innermost wins
    assert registration_suppressed() is False


def test_a_replay_scope_does_not_track_a_rebuilt_agent() -> None:
    import fastaiagent as fa
    from fastaiagent._platform import push
    from fastaiagent.testing import TestModel

    with provenance_scope("replay", suppress_registration=True):
        rebuilt = fa.Agent(name="rebuilt-in-replay", system_prompt="s", llm=TestModel())
    real = fa.Agent(name="real-agent-tracked", system_prompt="s", llm=TestModel())
    tracked = {a.name for a in list(push._agent_registry)}
    assert real.name in tracked
    assert rebuilt.name not in tracked
