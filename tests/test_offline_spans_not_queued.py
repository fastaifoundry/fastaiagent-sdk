"""A span recorded offline never reaches a plane on a later connect() (1.95.0, S1).

Before 1.95.0 every span was written ``synced=0``, so the first ``connect()``
uploaded up to 7 days of offline history, which the docs said it does not do.
Real tracer, real SQLite, a real ``connect()`` against a localhost HTTP server.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

import fastaiagent
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.trace import otel, trace_context
from fastaiagent.trace.platform_export import PlatformSpanExporter
from fastaiagent.ui.db import _run_migrations


@pytest.fixture(autouse=True)
def _fresh_provider() -> Any:
    otel.reset()
    yield
    fastaiagent.disconnect()
    otel.reset()


def _record(name: str) -> str:
    with trace_context(name) as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
    otel.get_tracer_provider().force_flush()
    return trace_id


def _synced(db_path: Path, trace_id: str) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT synced FROM spans WHERE trace_id = ?", (trace_id,))
        return [r[0] for r in rows]


def _sent_trace_ids(capture_server: Any) -> set[str]:
    return {
        span["trace_id"]
        for req in capture_server.ingest_requests
        for span in (req["body"] or {}).get("spans", [])
    }


def test_offline_span_is_not_sent_by_a_later_connect(
    isolated_local_db: Path, capture_server: Any
) -> None:
    offline = _record("offline-run")
    assert _synced(isolated_local_db, offline) == [2]  # kept: never sent

    fastaiagent.connect(api_key="fa_k_test", target=capture_server.url, auto_register=False)
    connected = _record("connected-run")  # the flush drains it to the server
    PlatformSpanExporter().export([])

    sent = _sent_trace_ids(capture_server)
    assert connected in sent
    assert _synced(isolated_local_db, connected) == [1]  # the plane confirmed it
    assert offline not in sent
    assert _synced(isolated_local_db, offline) == [2]  # kept: never sent


def test_export_traces_false_queues_nothing(isolated_local_db: Path, capture_server: Any) -> None:
    fastaiagent.connect(
        api_key="fa_k_test", target=capture_server.url, auto_register=False, export_traces=False
    )
    trace_id = _record("no-export-run")
    assert _synced(isolated_local_db, trace_id) == [2]  # kept: never sent


def test_after_disconnect_spans_are_not_queued(
    isolated_local_db: Path, capture_server: Any
) -> None:
    fastaiagent.connect(api_key="fa_k_test", target=capture_server.url, auto_register=False)
    fastaiagent.disconnect()
    trace_id = _record("after-disconnect")
    assert _synced(isolated_local_db, trace_id) == [2]  # kept: never sent


def test_v29_marks_the_existing_queue_handled(tmp_path: Path) -> None:
    db = SQLiteHelper(str(tmp_path / "local.db"))
    _run_migrations(db)
    db.execute("PRAGMA user_version = 28")
    db.execute(
        "INSERT INTO spans (span_id, trace_id, name, start_time, end_time, status, "
        "attributes, events, project_id, synced) "
        "VALUES ('s1', 't1', 'old', '2026-10-30T00:00:00+00:00', '2026-10-30T00:00:01+00:00', "
        "'OK', '{}', '[]', 'p', 0)"
    )
    _run_migrations(db)
    row = db.fetchone("SELECT synced FROM spans WHERE span_id = 's1'")
    # Handled by v29, then kept by v30 (1.97.0): never sent, so prune keeps it.
    assert row is not None and row["synced"] == 2
    db.close()
