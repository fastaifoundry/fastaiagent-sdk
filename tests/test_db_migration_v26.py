"""Schema v26 (1.89.0): case-level eval columns, run kind, eval_case_media.

Three paths against real SQLite (no mocks):

* a fresh database initializes to v26 with every column, index and table;
* a database at v25 with legacy eval rows upgrades, and the backfill fills the
  case columns from ``eval_cases`` and infers ``kind`` (ui-echo, optimize,
  pytest), leaving ``pass_*`` and ``summary_version`` untouched;
* a row INSERTed in the 1.88.0 shape *after* the upgrade (an older SDK writing
  to a shared ``local.db``) leaves the case columns NULL, and the route falls
  back to computing them from the cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db

V26_RUN_COLUMNS = (
    "kind",
    "summary_version",
    "case_count",
    "case_pass_count",
    "case_fail_count",
    "case_error_count",
    "case_pass_rate",
    "check_error_count",
    "check_skip_count",
)


def _user_version(db: SQLiteHelper) -> int:
    row = db.fetchone("PRAGMA user_version")
    return int(next(iter(row.values())))


def _columns(db: SQLiteHelper, table: str) -> set[str]:
    return {r["name"] for r in db.fetchall(f"PRAGMA table_info({table})")}


def test_fresh_db_is_v26(tmp_path: Path) -> None:
    db = init_local_db(tmp_path / "fresh.db")
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION == 26
        assert set(V26_RUN_COLUMNS) <= _columns(db, "eval_runs")
        assert "scoring_trace_id" in _columns(db, "eval_cases")
        assert {"case_id", "idx", "media_type", "size_bytes", "sha256", "thumbnail"} <= _columns(
            db, "eval_case_media"
        )
        run_idx = {r["name"] for r in db.fetchall("PRAGMA index_list(eval_runs)")}
        case_idx = {r["name"] for r in db.fetchall("PRAGMA index_list(eval_cases)")}
        assert "idx_eval_runs_kind_started" in run_idx
        assert "idx_eval_cases_scoring_trace" in case_idx
    finally:
        db.close()


def _seed_v25(db_file: Path) -> None:
    """A real v25 database: today's schema with the v26 additions removed."""
    db = init_local_db(db_file)
    try:
        db.execute("DROP INDEX IF EXISTS idx_eval_runs_kind_started")
        db.execute("DROP INDEX IF EXISTS idx_eval_cases_scoring_trace")
        for column in V26_RUN_COLUMNS:
            db.execute(f"ALTER TABLE eval_runs DROP COLUMN {column}")
        db.execute("ALTER TABLE eval_cases DROP COLUMN scoring_trace_id")
        db.execute("DROP TABLE eval_case_media")
        db.execute("PRAGMA user_version = 25")
    finally:
        db.close()


def _legacy_run(
    db: SQLiteHelper, run_id: str, *, run_name: str, agent_name: str, cases: list[dict]
) -> None:
    db.execute(
        """INSERT INTO eval_runs (run_id, run_name, agent_name, scorers, started_at,
               finished_at, pass_count, fail_count, pass_rate, metadata)
           VALUES (?, ?, ?, '[]', '2026-09-01T00:00:00', '2026-09-01T00:00:00', 20, 4,
                   0.8333, '{}')""",
        (run_id, run_name, agent_name),
    )
    for i, c in enumerate(cases):
        db.execute(
            """INSERT INTO eval_cases (case_id, run_id, ordinal, input, per_scorer, error)
               VALUES (?, ?, ?, '"q"', ?, ?)""",
            (f"{run_id}-{i}", run_id, i, json.dumps(c.get("per_scorer", {})), c.get("error")),
        )


CANDIDATE_C_CASES = [
    {"per_scorer": {"a": {"passed": True}, "b": {"passed": True}}},
    {"per_scorer": {"a": {"passed": False}, "b": {"passed": True}}},
    {"per_scorer": {"a": {"passed": False}, "b": {"passed": True}}},
    {"error": "RuntimeError: BOOM"},
    {"per_scorer": {"a": {"passed": False}, "b": {"passed": False}}},
]


def test_upgrade_from_v25_backfills_case_columns_and_kind(tmp_path: Path) -> None:
    db_file = tmp_path / "v25.db"
    _seed_v25(db_file)
    with SQLiteHelper(db_file) as seed:
        assert "case_count" not in _columns(seed, "eval_runs")
        _legacy_run(seed, "r-eval", run_name="nightly", agent_name="bot", cases=CANDIDATE_C_CASES)
        _legacy_run(seed, "r-echo", run_name="editor-x", agent_name="editor-echo", cases=[])
        _legacy_run(seed, "r-pytest", run_name="pytest::session", agent_name="bot", cases=[])
        _legacy_run(seed, "r-opt", run_name="cand", agent_name="bot", cases=[])
        if "optimize_iterations" in {
            r["name"] for r in seed.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            cols = _columns(seed, "optimize_iterations")
            assert "eval_run_id" in cols
            seed.execute(
                "INSERT INTO optimize_iterations (run_id, iteration, eval_run_id) "
                "VALUES ('opt-1', 0, 'r-opt')"
            )

    db = init_local_db(db_file)
    try:
        assert _user_version(db) == 26
        rows = {r["run_id"]: r for r in db.fetchall("SELECT * FROM eval_runs")}
        c = rows["r-eval"]
        assert (c["case_count"], c["case_pass_count"], c["case_fail_count"]) == (5, 1, 3)
        assert (c["case_error_count"], c["case_pass_rate"]) == (1, 0.25)
        # Legacy pass_* is not rewritten, and the row stays marked legacy.
        assert (c["pass_count"], c["fail_count"], c["pass_rate"]) == (20, 4, 0.8333)
        assert c["summary_version"] is None and c["kind"] is None
        assert rows["r-echo"]["kind"] == "ui-echo"
        assert rows["r-pytest"]["kind"] == "pytest"
        assert rows["r-opt"]["kind"] == "optimize"
        assert rows["r-echo"]["case_count"] == 0 and rows["r-echo"]["case_pass_rate"] is None
    finally:
        db.close()

    # Idempotent: a second init changes nothing.
    db = init_local_db(db_file)
    try:
        again = {r["run_id"]: r for r in db.fetchall("SELECT * FROM eval_runs")}
        assert again == rows
    finally:
        db.close()


def test_older_sdk_insert_falls_back_to_cases_in_the_route(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    db_file = tmp_path / "local.db"
    with init_local_db(db_file) as db:
        # The 1.88.0 INSERT names its columns, so the v26 columns stay NULL.
        _legacy_run(db, "old-sdk", run_name="old", agent_name="bot", cases=CANDIDATE_C_CASES)
        row = db.fetchone("SELECT case_count FROM eval_runs WHERE run_id = 'old-sdk'")
        assert row["case_count"] is None

    app = build_app(db_path=str(db_file), auth_path=tmp_path / "auth.json", no_auth=True)
    with TestClient(app) as client:
        listed = next(
            r for r in client.get("/api/evals").json()["rows"] if r["run_id"] == "old-sdk"
        )
        detail = client.get("/api/evals/old-sdk").json()["run"]
    for run in (listed, detail):
        assert run["case_count"] == 5
        assert (run["case_pass_count"], run["case_fail_count"], run["case_error_count"]) == (
            1,
            3,
            1,
        )
        assert run["case_pass_rate"] == 0.25
