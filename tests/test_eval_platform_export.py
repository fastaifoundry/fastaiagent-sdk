"""Eval-run outbox → plane (Part D, 1.49.0).

Policy per ``tests/test_platform_buffering.py``: real SQLite + a real localhost
HTTP server; no mocking of httpx or the store. ``monkeypatch`` only lowers the
system-under-test's own backoff so retry tests don't sleep — that configures the
subject, it doesn't fake a dependency.
"""

from __future__ import annotations

import socket
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from fastaiagent.client import _connection
from fastaiagent.eval import evaluate
from fastaiagent.eval.platform_export import (
    EvalRunExporter,
    EvalRunStore,
    build_payloads,
    eval_export_enabled,
)
from fastaiagent.eval.results import record_gate_result

DATASET = [
    {"input": "greet", "expected_output": "hi"},
    {"input": "farewell", "expected_output": "bye"},
]

_BANNED = {"input", "expected_output", "actual_output"}


def _agent(x: str) -> str:
    return {"greet": "hi", "farewell": "bye"}[x]


@pytest.fixture(autouse=True)
def _reset_connection():
    """Every test owns the process-global connection state."""
    saved = (
        _connection.api_key,
        _connection.target,
        _connection.project,
        _connection.project_id,
        _connection.export_evals,
    )
    yield
    (
        _connection.api_key,
        _connection.target,
        _connection.project,
        _connection.project_id,
        _connection.export_evals,
    ) = saved


def _connect(target: str) -> None:
    """Set connection state directly — connect() would need a live auth check."""
    _connection.api_key = "test-key"
    _connection.target = target
    _connection.project = "test-proj"
    _connection.project_id = "test-proj"
    _connection.export_evals = None  # unset → default on when connected


def _dead_url() -> str:
    """A port guaranteed to refuse connections."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def _seed_run(db_path, *, gate_outcome: str = "passed", run_name: str = "main") -> str:
    """Persist a real gated run, then make it pending deterministically.

    Seeding runs with the posture off so ``record_gate_result``'s background drain
    kick (real, and covered separately below) can't race these assertions; the row
    is then flipped to ``synced=0`` by hand. Still real SQLite and the real persist
    path — just deterministic, the same approach ``test_platform_buffering.py``
    takes with ``_seed_span``.
    """
    saved = _connection.export_evals
    _connection.export_evals = False
    try:
        results = evaluate(_agent, DATASET, scorers=["exact_match"], persist=False, concurrency=1)
        run_id = results.persist_local(db_path=db_path, run_name=run_name)
        record_gate_result(
            run_id,
            gate_outcome=gate_outcome,
            thresholds={"overall.pass_rate": 0.9},
            db_path=db_path,
        )
    finally:
        _connection.export_evals = saved
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE eval_runs SET synced = 0 WHERE run_id = ?", (run_id,))
    return run_id


def _synced(db_path, run_id: str) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT synced FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone()[
            0
        ]


def _eval_requests(server) -> list:
    return [r for r in server.requests if r["path"].endswith("/eval/runs/ingest")]


# ── posture ────────────────────────────────────────────────────────────────


def test_disconnected_never_exports(isolated_local_db, capture_server):
    assert eval_export_enabled() is False
    exporter = EvalRunExporter()
    assert exporter.export([]) is not None
    assert _eval_requests(capture_server) == []


def test_export_disabled_writes_synced_and_sends_nothing(isolated_local_db, capture_server):
    """A disabled install must not grow an outbox it can never drain."""
    _connect(capture_server.url)
    _connection.export_evals = False
    results = evaluate(_agent, DATASET, scorers=["exact_match"], persist=False, concurrency=1)
    run_id = results.persist_local(db_path=isolated_local_db, run_name="off")
    record_gate_result(run_id, gate_outcome="passed", thresholds={}, db_path=isolated_local_db)
    assert _synced(isolated_local_db, run_id) == 1  # never queued
    EvalRunExporter().export([])
    assert _eval_requests(capture_server) == []


def test_env_var_controls_posture(isolated_local_db, monkeypatch, capture_server):
    _connect(capture_server.url)
    monkeypatch.setenv("FASTAIAGENT_EXPORT_EVALS", "0")
    assert eval_export_enabled() is False
    monkeypatch.setenv("FASTAIAGENT_EXPORT_EVALS", "1")
    assert eval_export_enabled() is True
    # kwarg wins over env
    _connection.export_evals = False
    assert eval_export_enabled() is False


# ── happy path + wire shape ────────────────────────────────────────────────


def test_happy_path_drains_and_marks_synced(isolated_local_db, capture_server):
    _connect(capture_server.url)
    run_id = _seed_run(isolated_local_db)
    assert _synced(isolated_local_db, run_id) == 0

    EvalRunExporter().export([])

    assert len(_eval_requests(capture_server)) == 1
    assert _synced(isolated_local_db, run_id) == 1


def test_wire_shape_is_frozen_and_carries_no_content(isolated_local_db, capture_server):
    """The privacy contract: content fields must never reach the wire."""
    _connect(capture_server.url)
    _seed_run(isolated_local_db)
    EvalRunExporter().export([])

    req = _eval_requests(capture_server)[0]
    assert req["path"].endswith("/public/v1/eval/runs/ingest")
    assert req["headers"].get("X-API-Key") == "test-key"

    body = req["body"]  # CaptureServer already JSON-decodes
    assert set(body.keys()) == {"runs"}
    run = body["runs"][0]
    assert set(run.keys()) == {
        "run_id",
        "run_name",
        "dataset_name",
        "agent_name",
        "started_at",
        "finished_at",
        "pass_count",
        "fail_count",
        "pass_rate",
        "errored_count",
        "error_rate",
        "gate_outcome",
        "thresholds",
        "scorers",
        "git_sha",
        "git_branch",
        "baseline",
        "sdk_version",
        "instance_id",
        "cases",
    }
    assert not (_BANNED & set(run)), "run payload leaks case content"
    case = run["cases"][0]
    assert set(case.keys()) == {"case_id", "ordinal", "per_scorer", "trace_id", "error"}
    assert not (_BANNED & set(case)), "case payload leaks content"
    # The plane requires a verdict.
    assert run["gate_outcome"] in ("passed", "failed", "invalid")


def test_build_payloads_never_includes_content(isolated_local_db):
    _seed_run(isolated_local_db)
    for run in build_payloads(limit=5, db_path=isolated_local_db):
        assert not (_BANNED & set(run))
        for c in run["cases"]:
            assert not (_BANNED & set(c))


def test_gate_verdict_and_baseline_reach_the_payload(isolated_local_db):
    results = evaluate(_agent, DATASET, scorers=["exact_match"], persist=False, concurrency=1)
    run_id = results.persist_local(db_path=isolated_local_db, run_name="pr")
    record_gate_result(
        run_id,
        gate_outcome="failed",
        thresholds={"overall.pass_rate": 0.9},
        baseline={"run_id": "base-1", "pass_rate_delta": -0.5, "regressed_count": 1},
        db_path=isolated_local_db,
    )
    payload = build_payloads(run_id=run_id, db_path=isolated_local_db)[0]
    assert payload["gate_outcome"] == "failed"
    assert payload["thresholds"] == {"overall.pass_rate": 0.9}
    assert payload["baseline"]["regressed_count"] == 1


# ── failure handling ───────────────────────────────────────────────────────


def test_outage_buffers_and_does_not_raise(isolated_local_db, monkeypatch):
    _connect(_dead_url())
    run_id = _seed_run(isolated_local_db)
    exporter = EvalRunExporter()
    monkeypatch.setattr(exporter, "_BACKOFF_BASE", 0.0)
    exporter.export([])
    assert _synced(isolated_local_db, run_id) == 0  # still a re-send candidate


def test_retry_on_5xx_then_success(isolated_local_db, capture_server, monkeypatch):
    _connect(capture_server.url)
    run_id = _seed_run(isolated_local_db)
    capture_server.set_status_sequence([500, 500, 200])
    exporter = EvalRunExporter()
    monkeypatch.setattr(exporter, "_BACKOFF_BASE", 0.0)
    exporter.export([])
    assert len(_eval_requests(capture_server)) == 3
    assert _synced(isolated_local_db, run_id) == 1


def test_retry_exhausted_keeps_buffered(isolated_local_db, capture_server, monkeypatch):
    _connect(capture_server.url)
    run_id = _seed_run(isolated_local_db)
    capture_server.set_status_sequence([500, 500, 500])
    exporter = EvalRunExporter()
    monkeypatch.setattr(exporter, "_BACKOFF_BASE", 0.0)
    exporter.export([])
    assert len(_eval_requests(capture_server)) == 3
    assert _synced(isolated_local_db, run_id) == 0


def test_no_retry_on_4xx(isolated_local_db, capture_server, monkeypatch):
    """403 (missing eval:execute scope / unentitled domain) is terminal."""
    _connect(capture_server.url)
    run_id = _seed_run(isolated_local_db)
    capture_server.set_status_sequence([403])
    exporter = EvalRunExporter()
    monkeypatch.setattr(exporter, "_BACKOFF_BASE", 0.0)
    exporter.export([])
    assert len(_eval_requests(capture_server)) == 1  # exactly one — no retry
    assert _synced(isolated_local_db, run_id) == 0  # left for the bound to age out


def test_recovery_redrains_after_outage(isolated_local_db, capture_server, monkeypatch):
    _connect(_dead_url())
    run_id = _seed_run(isolated_local_db)
    exporter = EvalRunExporter()
    monkeypatch.setattr(exporter, "_BACKOFF_BASE", 0.0)
    exporter.export([])
    assert _synced(isolated_local_db, run_id) == 0

    _connection.target = capture_server.url  # plane comes back
    exporter.export([])
    assert _synced(isolated_local_db, run_id) == 1


# ── store mechanics ────────────────────────────────────────────────────────


def test_buffer_bound_by_age_abandons_but_keeps_rows(isolated_local_db):
    run_id = _seed_run(isolated_local_db)
    old = (datetime.now(tz=timezone.utc) - timedelta(days=90)).isoformat()
    with sqlite3.connect(isolated_local_db) as conn:
        conn.execute("UPDATE eval_runs SET started_at = ? WHERE run_id = ?", (old, run_id))

    store = EvalRunStore(db_path=isolated_local_db)
    try:
        dropped = store.enforce_buffer_bound(50_000, 30, project_id="test-proj")
    finally:
        store.close()
    assert dropped == 1
    assert _synced(isolated_local_db, run_id) == 1  # abandoned == marked, not deleted
    with sqlite3.connect(isolated_local_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 1


def test_buffer_bound_by_count_abandons_oldest(isolated_local_db):
    ids = [_seed_run(isolated_local_db, run_name=f"r{i}") for i in range(3)]
    store = EvalRunStore(db_path=isolated_local_db)
    try:
        dropped = store.enforce_buffer_bound(1, 3650, project_id="test-proj")
    finally:
        store.close()
    assert dropped == 2
    assert _synced(isolated_local_db, ids[-1]) == 0  # newest survives


def test_mark_synced_chunks_over_999(isolated_local_db):
    store = EvalRunStore(db_path=isolated_local_db)
    try:
        store.mark_synced([f"id-{i}" for i in range(1200)])  # must not raise
    finally:
        store.close()


def test_gate_kick_drains_in_the_background(isolated_local_db, capture_server):
    """The real trigger: recording a gate verdict pushes without an explicit drain."""
    import time

    _connect(capture_server.url)
    results = evaluate(_agent, DATASET, scorers=["exact_match"], persist=False, concurrency=1)
    run_id = results.persist_local(db_path=isolated_local_db, run_name="kicked")
    record_gate_result(run_id, gate_outcome="passed", thresholds={}, db_path=isolated_local_db)
    deadline = time.time() + 10
    while time.time() < deadline and _synced(isolated_local_db, run_id) == 0:
        time.sleep(0.05)
    assert _synced(isolated_local_db, run_id) == 1
    assert _eval_requests(capture_server)


def test_count_unsynced_is_project_scoped(isolated_local_db):
    _seed_run(isolated_local_db)
    store = EvalRunStore(db_path=isolated_local_db)
    try:
        assert store.count_unsynced() == 1
        assert store.count_unsynced(project_id="other-project") == 0
    finally:
        store.close()


# ── agent attribution ──────────────────────────────────────────────────────
#
# Load-bearing for connected mode: the plane resolves ``agent_name`` to a real
# agent, so an unattributed run marks nothing and evidences nothing.


def test_infer_agent_name_from_bound_method():
    from fastaiagent.agent import Agent
    from fastaiagent.eval.evaluate import infer_agent_name
    from fastaiagent.testing import TestModel

    agent = Agent(name="support-bot", llm=TestModel(response="hi"))
    assert infer_agent_name(agent.run) == "support-bot"


def test_infer_agent_name_handles_partial_and_plain_callables():
    import functools

    from fastaiagent.agent import Agent
    from fastaiagent.eval.evaluate import infer_agent_name
    from fastaiagent.testing import TestModel

    agent = Agent(name="wrapped-bot", llm=TestModel(response="hi"))
    assert infer_agent_name(functools.partial(agent.run)) == "wrapped-bot"
    # A bare function has no agent — None beats guessing wrong.
    assert infer_agent_name(lambda x: x) is None


def test_evaluate_records_agent_name(isolated_local_db):
    import sqlite3

    from fastaiagent.agent import Agent
    from fastaiagent.testing import TestModel

    agent = Agent(name="attributed-bot", llm=TestModel(response="hi"))
    # persist=True exercises the real inference path inside aevaluate.
    evaluate(
        agent.run,
        [{"input": "x", "expected_output": "hi"}],
        scorers=["contains"],
        run_name="attr",
    )
    with sqlite3.connect(isolated_local_db) as conn:
        name = conn.execute("SELECT agent_name FROM eval_runs").fetchone()[0]
    assert name == "attributed-bot"


def test_session_agent_name_is_none_when_agents_differ():
    """A multi-agent suite has no single owner — don't credit the first one."""
    from fastaiagent.eval.pytest_plugin import (
        _CollectedCase,
        _session_agent_name,
        _SessionCollector,
    )
    from fastaiagent.eval.results import EvalCaseRecord

    c = _SessionCollector()
    c.cases = [
        _CollectedCase(record=EvalCaseRecord(input="a"), agent_name="one"),
        _CollectedCase(record=EvalCaseRecord(input="b"), agent_name="one"),
    ]
    assert _session_agent_name(c) == "one"
    c.cases.append(_CollectedCase(record=EvalCaseRecord(input="c"), agent_name="two"))
    assert _session_agent_name(c) is None


def test_kick_drains_the_db_the_run_was_written_to(isolated_local_db, capture_server, tmp_path):
    """A custom --db must not be silently skipped.

    The exporter singleton opens the *configured* DB. When a run is persisted
    elsewhere (``fastaiagent eval run --db ...``) the kick has to drain that file,
    or those runs never export and nothing says so.
    """
    import time

    other_db = tmp_path / "elsewhere.db"
    _connect(capture_server.url)

    results = evaluate(_agent, DATASET, scorers=["exact_match"], persist=False, concurrency=1)
    run_id = results.persist_local(db_path=other_db, run_name="custom-db")
    record_gate_result(run_id, gate_outcome="passed", thresholds={}, db_path=other_db)

    deadline = time.time() + 10
    while time.time() < deadline and _synced(other_db, run_id) == 0:
        time.sleep(0.05)
    assert _synced(other_db, run_id) == 1, "run in a custom DB was never drained"
    assert _eval_requests(capture_server)
