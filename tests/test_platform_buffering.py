"""Durable offline buffering + retry for platform trace export.

All tests exercise the real code paths against **real SQLite and a real
localhost HTTP server** (``CaptureServer`` from conftest) — no mocking of
``httpx`` or the store. The exporter's own backoff tunable is lowered via
``monkeypatch`` in retry tests so they don't sleep; that configures the
system-under-test, it doesn't fake a dependency.
"""

from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from fastaiagent.client import _connection
from fastaiagent.trace.platform_export import PlatformSpanExporter
from fastaiagent.trace.storage import TraceStore


@pytest.fixture(autouse=True)
def _reset_connection():
    """Reset platform connection state around every test."""
    yield
    _connection.api_key = None
    _connection.target = "https://app.fastaiagent.net"
    _connection.project = None
    _connection.project_id = None
    _connection._platform_processor = None


def _connect(target: str) -> None:
    """Put ``_connection`` in a connected state pointed at ``target``."""
    _connection.api_key = "fa_k_test"
    _connection.target = target
    _connection.project = "test-project"
    _connection.project_id = "platform-uuid-1"


def _dead_url() -> str:
    """A URL whose port has nothing listening (connection refused)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def _recent(seconds_ago: float = 3600.0) -> str:
    """An ISO timestamp ``seconds_ago`` in the past — recent enough to stay inside
    the exporter's age-based buffer bound (``_MAX_AGE_DAYS``) on any run date, so
    these tests don't rot as the calendar advances past a hard-coded seed date."""
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def _seed_span(
    store: TraceStore,
    span_id: str,
    *,
    trace_id: str = "t1",
    start: str | None = None,
    project_id: str = "test-proj",
    synced: int = 0,
    attrs: dict | None = None,
) -> None:
    """Insert one span row directly (still real SQLite — just deterministic).

    ``start`` defaults to a recent timestamp so the row survives the exporter's
    age bound regardless of when the test runs; pass an explicit old value to
    exercise age-out.
    """
    if start is None:
        start = _recent()
    store._db.execute(
        "INSERT INTO spans (span_id, trace_id, parent_span_id, name, start_time, "
        "end_time, status, attributes, events, project_id, synced) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            span_id,
            trace_id,
            None,
            f"span-{span_id}",
            start,
            start,
            "OK",
            json.dumps(attrs if attrs is not None else {"k": "v"}),
            "[]",
            project_id,
            synced,
        ),
    )


# --- Connection gating -------------------------------------------------------


def test_export_returns_success_when_not_connected() -> None:
    assert PlatformSpanExporter().export([]) == SpanExportResult.SUCCESS


# --- Happy path --------------------------------------------------------------


def test_happy_path_drains_pushes_and_marks_synced(isolated_local_db, capture_server) -> None:
    store = TraceStore()
    _seed_span(store, "s1")
    _seed_span(store, "s2", start="2026-06-07T00:00:01+00:00")
    _connect(capture_server.url)

    result = PlatformSpanExporter().export([])

    assert result == SpanExportResult.SUCCESS
    assert len(capture_server.ingest_requests) == 1
    body = capture_server.ingest_requests[0]["body"]
    assert body["project"] == "platform-uuid-1"
    assert {s["span_id"] for s in body["spans"]} == {"s1", "s2"}
    assert store.count_unsynced("test-proj") == 0  # both marked synced=1
    store.close()


def test_wire_shape_is_frozen(isolated_local_db, capture_server) -> None:
    store = TraceStore()
    _seed_span(store, "s1")
    _connect(capture_server.url)

    PlatformSpanExporter().export([])

    req = capture_server.ingest_requests[0]
    assert req["path"] == "/public/v1/traces/ingest"
    # X-API-Key header travels (case-insensitive lookup).
    assert any(k.lower() == "x-api-key" for k in req["headers"])
    body = req["body"]
    assert set(body.keys()) == {"project", "spans"}
    assert set(body["spans"][0].keys()) == {
        "span_id",
        "trace_id",
        "parent_span_id",
        "name",
        "start_time",
        "end_time",
        "status",
        "attributes",
        "events",
    }
    store.close()


# --- Outage / buffering ------------------------------------------------------


def test_outage_buffers_spans_and_agent_unaffected(
    isolated_local_db, capture_server, monkeypatch
) -> None:
    monkeypatch.setattr(PlatformSpanExporter, "_BACKOFF_BASE", 0.0)
    store = TraceStore()
    _seed_span(store, "s1")
    _connect(_dead_url())  # nothing listening

    # export() must not raise and must report SUCCESS (never blocks/breaks the agent).
    result = PlatformSpanExporter().export([])

    assert result == SpanExportResult.SUCCESS
    assert store.count_unsynced("test-proj") == 1  # still buffered
    assert capture_server.ingest_requests == []
    store.close()


# --- Retry behaviour ---------------------------------------------------------


def test_retry_on_5xx_then_success(isolated_local_db, capture_server, monkeypatch) -> None:
    monkeypatch.setattr(PlatformSpanExporter, "_BACKOFF_BASE", 0.0)
    capture_server.set_status_sequence([500, 500, 200])
    store = TraceStore()
    _seed_span(store, "s1")
    _connect(capture_server.url)

    PlatformSpanExporter().export([])

    assert len(capture_server.ingest_requests) == 3  # retried twice, succeeded on 3rd
    assert store.count_unsynced("test-proj") == 0
    store.close()


def test_retry_exhausted_keeps_buffered(isolated_local_db, capture_server, monkeypatch) -> None:
    monkeypatch.setattr(PlatformSpanExporter, "_BACKOFF_BASE", 0.0)
    capture_server.set_status_sequence([500, 500, 500])
    store = TraceStore()
    _seed_span(store, "s1")
    _connect(capture_server.url)

    PlatformSpanExporter().export([])

    assert len(capture_server.ingest_requests) == 3  # 3 attempts, all 5xx
    assert store.count_unsynced("test-proj") == 1  # left buffered
    store.close()


def test_no_retry_on_4xx(isolated_local_db, capture_server, monkeypatch) -> None:
    monkeypatch.setattr(PlatformSpanExporter, "_BACKOFF_BASE", 0.0)
    capture_server.set_status_sequence([401])
    store = TraceStore()
    _seed_span(store, "s1")
    _connect(capture_server.url)

    PlatformSpanExporter().export([])

    assert len(capture_server.ingest_requests) == 1  # 4xx → no retry
    assert store.count_unsynced("test-proj") == 1  # left buffered (bound ages it out)
    store.close()


# --- Outage → recovery → re-drain (re-send is safe; server dedups by span_id) ---


def test_recovery_redrains_after_outage(isolated_local_db, capture_server, monkeypatch) -> None:
    monkeypatch.setattr(PlatformSpanExporter, "_BACKOFF_BASE", 0.0)
    store = TraceStore()
    _seed_span(store, "s1")
    exporter = PlatformSpanExporter()

    # Outage.
    _connect(_dead_url())
    assert exporter.export([]) == SpanExportResult.SUCCESS
    assert store.count_unsynced("test-proj") == 1
    assert capture_server.ingest_requests == []

    # Recovery — same buffered span re-drains to the now-reachable platform.
    _connect(capture_server.url)
    assert exporter.export([]) == SpanExportResult.SUCCESS
    assert len(capture_server.ingest_requests) == 1
    assert capture_server.ingest_requests[0]["body"]["spans"][0]["span_id"] == "s1"
    assert store.count_unsynced("test-proj") == 0

    exporter.shutdown()
    store.close()


# --- Buffer bound (abandon-but-keep-local) -----------------------------------


def test_buffer_bound_count_abandons_oldest_keeps_local(
    isolated_local_db, capture_server, monkeypatch
) -> None:
    # Cap the queue at 2; with the dead platform the spans can't push, so the
    # bound abandons the oldest excess (marks synced=1) but keeps every row.
    monkeypatch.setattr(PlatformSpanExporter, "_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(PlatformSpanExporter, "_MAX_UNSYNCED", 2)
    store = TraceStore()
    for i in range(5):
        _seed_span(store, f"s{i}", start=_recent(3600 - i))  # s0 oldest … s4 newest, all recent
    _connect(_dead_url())

    PlatformSpanExporter().export([])

    assert store.count_unsynced("test-proj") == 2  # bounded to the 2 newest
    total = store._db.fetchone("SELECT COUNT(*) AS n FROM spans")["n"]
    assert total == 5  # abandon != delete — all rows remain locally
    store.close()


def test_buffer_bound_age_abandons_old_keeps_local(
    isolated_local_db, capture_server, monkeypatch
) -> None:
    monkeypatch.setattr(PlatformSpanExporter, "_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(PlatformSpanExporter, "_MAX_AGE_DAYS", 7)
    store = TraceStore()
    _seed_span(store, "recent", start=_recent())  # ~1h ago → inside the 7-day window
    _seed_span(store, "ancient", start="2020-01-01T00:00:00+00:00")  # far outside → abandoned
    _connect(_dead_url())

    PlatformSpanExporter().export([])

    # 'ancient' is older than the cutoff → abandoned; 'recent' stays buffered.
    rows = {
        r["span_id"]: r["synced"] for r in store._db.fetchall("SELECT span_id, synced FROM spans")
    }
    assert rows["ancient"] == 1
    assert rows["recent"] == 0
    assert store._db.fetchone("SELECT COUNT(*) AS n FROM spans")["n"] == 2  # kept
    store.close()


# --- Storage helper units ----------------------------------------------------


def test_fetch_unsynced_order_limit_and_project_scope(isolated_local_db) -> None:
    store = TraceStore()
    _seed_span(store, "b", start="2026-06-07T00:00:02+00:00")
    _seed_span(store, "a", start="2026-06-07T00:00:01+00:00")  # earlier → first
    _seed_span(store, "other", project_id="other-proj")
    _seed_span(store, "acked", synced=1)

    got = store.fetch_unsynced(limit=10, project_id="test-proj")
    assert [s.span_id for s in got] == ["a", "b"]  # oldest first, scoped, synced=0 only

    assert [s.span_id for s in store.fetch_unsynced(limit=1, project_id="test-proj")] == ["a"]
    store.close()


def test_mark_synced_chunks_over_999(isolated_local_db) -> None:
    store = TraceStore()
    ids = [f"s{i}" for i in range(1200)]  # > SQLite's 999 variable limit
    for i, sid in enumerate(ids):
        _seed_span(store, sid, start=f"2026-06-07T00:00:{i:02d}+00:00")
    assert store.count_unsynced("test-proj") == 1200

    store.mark_synced(ids)  # must not raise on the variable limit

    assert store.count_unsynced("test-proj") == 0
    store.close()


# --- Migration v11 -----------------------------------------------------------


def test_migration_v11_adds_synced_backfills_and_is_idempotent(tmp_path: Path) -> None:
    from fastaiagent._internal.storage import SQLiteHelper
    from fastaiagent.ui import db as ui_db

    p = tmp_path / "v10.db"
    h = SQLiteHelper(str(p))
    h.execute(
        "CREATE TABLE spans (span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, "
        "parent_span_id TEXT, name TEXT, start_time TEXT, end_time TEXT, "
        "status TEXT DEFAULT 'OK', attributes TEXT DEFAULT '{}', events TEXT DEFAULT '[]', "
        "project_id TEXT NOT NULL DEFAULT '')"
    )
    h.execute(
        "INSERT INTO spans (span_id, trace_id, name, start_time, project_id) "
        "VALUES ('old', 't1', 'old', '2026-06-01T00:00:00+00:00', 'p')"
    )
    h.execute("PRAGMA user_version = 10")
    h.close()

    db = ui_db.init_local_db(str(p))
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(spans)")}
    assert "synced" in cols
    # Existing rows backfilled to synced=1 → upgrade does NOT back-push history.
    assert db.fetchone("SELECT synced FROM spans WHERE span_id='old'")["synced"] == 1
    # New rows still default to synced=0 → they become push candidates.
    db.execute(
        "INSERT INTO spans (span_id, trace_id, name, start_time, project_id) "
        "VALUES ('new', 't1', 'new', '2026-06-07T00:00:00+00:00', 'p')"
    )
    assert db.fetchone("SELECT synced FROM spans WHERE span_id='new'")["synced"] == 0
    # init_local_db always migrates to the current head (now 12: + hitl_events).
    assert ui_db._get_user_version(db) == 12

    ui_db._run_migrations(db)  # idempotent re-run
    assert ui_db._get_user_version(db) == 12
    db.close()


# --- Migration v12 -----------------------------------------------------------


def test_migration_v12_adds_hitl_events_table_and_is_idempotent(tmp_path: Path) -> None:
    from fastaiagent._internal.storage import SQLiteHelper
    from fastaiagent.ui import db as ui_db

    # Start from a v11 DB (spans + synced, no hitl_events yet).
    p = tmp_path / "v11.db"
    h = SQLiteHelper(str(p))
    h.execute(
        "CREATE TABLE spans (span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, "
        "synced INTEGER NOT NULL DEFAULT 0, project_id TEXT NOT NULL DEFAULT '')"
    )
    h.execute("PRAGMA user_version = 11")
    h.close()

    db = ui_db.init_local_db(str(p))
    # New append-only table exists with the wire-mirroring columns + outbox flag.
    tbls = {
        r["name"]
        for r in db.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hitl_events'"
        )
    }
    assert "hitl_events" in tbls
    cols = {r["name"] for r in db.fetchall("PRAGMA table_info(hitl_events)")}
    assert {
        "event_id",
        "run_id",
        "event_type",
        "status",
        "resolver",
        "project_id",
        "synced",
        "created_at",
    } <= cols
    assert ui_db._get_user_version(db) == 12

    ui_db._run_migrations(db)  # idempotent re-run
    assert ui_db._get_user_version(db) == 12
    db.close()


# --- Full local-first → push, through the real LocalStorageProcessor ----------


def test_local_first_then_push_end_to_end(isolated_local_db, capture_server) -> None:
    """A real OTel span written by LocalStorageProcessor (synced=0) is drained
    and pushed by the exporter, then marked synced=1 — usage rides inside it."""
    from opentelemetry.sdk.trace import TracerProvider

    from fastaiagent.trace.storage import LocalStorageProcessor

    prov = TracerProvider()
    lsp = LocalStorageProcessor(db_path=str(isolated_local_db))
    prov.add_span_processor(lsp)
    tracer = prov.get_tracer("test")
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("gen_ai.usage.input_tokens", 11)
        span.set_attribute("gen_ai.usage.output_tokens", 7)
    lsp.shutdown()

    store = TraceStore()
    assert store.count_unsynced("test-proj") == 1  # local-first write, un-acked

    _connect(capture_server.url)
    exporter = PlatformSpanExporter()
    assert exporter.export([]) == SpanExportResult.SUCCESS
    exporter.shutdown()

    assert len(capture_server.ingest_requests) == 1
    pushed = capture_server.ingest_requests[0]["body"]["spans"][0]
    assert pushed["name"] == "agent.run"
    assert pushed["attributes"]["gen_ai.usage.input_tokens"] == 11  # usage embedded
    assert store.count_unsynced("test-proj") == 0  # acked → synced=1
    store.close()
