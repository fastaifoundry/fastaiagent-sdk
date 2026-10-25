"""Trace export pauses at the plane's monthly trace limit (1.94.0, S7 / D161).

Wire v1.11: over the limit ``POST /public/v1/traces/ingest`` still answers 201,
with ``limit_reached``, ``resets_at``, ``rejected_traces`` and ``trace_limit``.
Before 1.94.0 the SDK ignored those keys and kept sending for the rest of the
month (checked live on a local plane: three more requests after the first
``limit_reached``, each refused). Real SQLite and a real localhost HTTP server
answering with the plane's body; no mocks.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from fastaiagent.client import _connection
from fastaiagent.trace.platform_export import PlatformSpanExporter, _pause_scope
from fastaiagent.trace.storage import TraceStore
from tests.test_platform_buffering import _seed_span

RESETS_AT = "2099-11-01T00:00:00Z"
LIMIT_ANSWER = {
    "ingested": 1,
    "limit_reached": True,
    "resets_at": RESETS_AT,
    "rejected_traces": 1,
    "trace_limit": 3,
}


@pytest.fixture(autouse=True)
def _connected(capture_server):
    from fastaiagent.trace import platform_export

    _connection.api_key = "fa_k_test"
    _connection.target = capture_server.url
    _connection.project = "test-project"
    _connection.project_id = "platform-uuid-1"
    platform_export._announced_pauses.clear()
    yield
    _connection.api_key = None
    _connection.target = "https://app.fastaiagent.net"
    _connection.project = None
    _connection.project_id = None


def _ago(**delta: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


def _limit(capture_server) -> None:
    capture_server.set_response_sequence([(201, json.dumps(LIMIT_ANSWER), "application/json")])


def test_a_limit_answer_pauses_export(isolated_local_db, capture_server) -> None:
    store = TraceStore()
    _seed_span(store, "s1")
    _limit(capture_server)
    exporter = PlatformSpanExporter()
    exporter.export([])
    assert len(capture_server.ingest_requests) == 1
    assert store.count_unsynced("test-proj") == 0, "the answered batch is done"

    _seed_span(store, "s2", trace_id="t2")  # recorded while paused
    exporter.export([])
    exporter.export([])
    assert len(capture_server.ingest_requests) == 1, "nothing is sent while paused"
    assert store.export_pause(_pause_scope(_connection)) == "2099-11-01T00:00:00+00:00"
    assert store.list_spans(trace_id="t2"), "still recorded locally"
    store.close()


def test_the_pause_survives_a_restart(isolated_local_db, capture_server) -> None:
    store = TraceStore()
    _seed_span(store, "s1")
    _limit(capture_server)
    PlatformSpanExporter().export([])
    store.close()

    restarted = TraceStore()  # a new process: a new store and a new exporter
    _seed_span(restarted, "s2", trace_id="t2")
    PlatformSpanExporter().export([])
    assert len(capture_server.ingest_requests) == 1
    restarted.close()


def test_after_the_reset_only_new_spans_are_sent(isolated_local_db, capture_server) -> None:
    store = TraceStore()
    resets_at = _ago(hours=1)
    store.set_export_pause(_pause_scope(_connection), resets_at, _ago(days=3))
    _seed_span(store, "during-pause", trace_id="old", start=_ago(hours=2))
    _seed_span(store, "after-reset", trace_id="new", start=_ago(minutes=30))

    PlatformSpanExporter().export([])

    (request,) = capture_server.ingest_requests
    assert [s["span_id"] for s in request["body"]["spans"]] == ["after-reset"]
    assert store.export_pause(_pause_scope(_connection)) is None, "the pause is over"
    assert store.count_unsynced("test-proj") == 0, "the backlog is marked done, not sent"
    store.close()


def test_one_warning_per_pause(isolated_local_db, capture_server, caplog) -> None:
    store = TraceStore()
    _seed_span(store, "s1")
    _limit(capture_server)
    exporter = PlatformSpanExporter()
    with caplog.at_level(logging.WARNING, logger="fastaiagent.trace.platform_export"):
        for n in range(4):
            _seed_span(store, f"later-{n}", trace_id=f"t{n}")
            exporter.export([])
    warnings = [r.getMessage() for r in caplog.records if "Trace limit reached" in r.getMessage()]
    assert warnings == [
        "Trace limit reached; export resumes on 2099-11-01 (the workspace's monthly limit "
        "is 3 traces). Traces are still recorded locally; ones from before the reset are "
        "not sent."
    ]
    store.close()


def test_the_pause_is_per_project(isolated_local_db, capture_server) -> None:
    store = TraceStore()
    store.set_export_pause(f"{capture_server.url}|another-project", RESETS_AT, _ago(minutes=1))
    _seed_span(store, "s1")
    PlatformSpanExporter().export([])
    assert len(capture_server.ingest_requests) == 1, "another project's pause is not ours"
    store.close()


def test_an_answer_without_resets_at_does_not_pause(
    isolated_local_db, capture_server, caplog
) -> None:
    store = TraceStore()
    _seed_span(store, "s1")
    body = {**LIMIT_ANSWER, "resets_at": None}
    capture_server.set_response_sequence([(201, json.dumps(body), "application/json")])
    with caplog.at_level(logging.WARNING, logger="fastaiagent.trace.platform_export"):
        PlatformSpanExporter().export([])
    assert store.export_pause(_pause_scope(_connection)) is None
    assert any("without a valid resets_at" in r.getMessage() for r in caplog.records)
    store.close()


def test_an_ordinary_answer_never_pauses(isolated_local_db, capture_server) -> None:
    store = TraceStore()
    _seed_span(store, "s1")
    capture_server.set_response_sequence([(201, json.dumps({"ingested": 1}), "application/json")])
    exporter = PlatformSpanExporter()
    exporter.export([])
    _seed_span(store, "s2", trace_id="t2")
    exporter.export([])
    assert len(capture_server.ingest_requests) == 2
    store.close()
