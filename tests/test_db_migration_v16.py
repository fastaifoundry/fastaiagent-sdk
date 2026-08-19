"""Migration test for schema v16 (``eval_cases.error``, Agent CI).

Two paths, both against a real SQLite file (no mocks):

  - a fresh DB initializes straight to v16 with the ``error`` column present;
  - a DB seeded at v15 (with an existing eval run + case) upgrades cleanly,
    keeping its data, gaining the column with NULL for pre-existing rows.
"""

from __future__ import annotations

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db


def _user_version(db: SQLiteHelper) -> int:
    row = db.fetchone("PRAGMA user_version")
    return int(next(iter(row.values())))


def test_fresh_db_is_v16_with_error_column(tmp_path) -> None:
    db = init_local_db(tmp_path / "fresh.db")
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION >= 16
        cols = {r["name"] for r in db.fetchall("PRAGMA table_info(eval_cases)")}
        assert "error" in cols
    finally:
        db.close()


def test_upgrade_from_v15_preserves_rows(tmp_path) -> None:
    db_file = tmp_path / "v15.db"

    # Seed a minimal v15-shape DB: eval tables WITHOUT the error column.
    seed = SQLiteHelper(db_file)
    seed.execute(
        """CREATE TABLE eval_runs (
            run_id TEXT PRIMARY KEY, run_name TEXT, dataset_name TEXT,
            agent_name TEXT, agent_version TEXT, scorers TEXT,
            started_at TEXT, finished_at TEXT, pass_count INTEGER,
            fail_count INTEGER, pass_rate REAL, metadata TEXT,
            project_id TEXT NOT NULL DEFAULT ''
        )"""
    )
    seed.execute(
        """CREATE TABLE eval_cases (
            case_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, ordinal INTEGER,
            input TEXT, expected_output TEXT, actual_output TEXT,
            trace_id TEXT, per_scorer TEXT,
            project_id TEXT NOT NULL DEFAULT ''
        )"""
    )
    seed.execute(
        "INSERT INTO eval_runs (run_id, run_name, pass_count, fail_count, pass_rate, metadata)"
        " VALUES ('r1', 'legacy', 1, 0, 1.0, '{}')"
    )
    seed.execute(
        "INSERT INTO eval_cases (case_id, run_id, ordinal, input, per_scorer)"
        " VALUES ('c1', 'r1', 0, '\"q\"', '{}')"
    )
    seed.execute("PRAGMA user_version = 15")
    seed.close()

    db = init_local_db(db_file)
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION
        cols = {r["name"] for r in db.fetchall("PRAGMA table_info(eval_cases)")}
        assert "error" in cols
        row = db.fetchone("SELECT run_id, error FROM eval_cases WHERE case_id = 'c1'")
        assert row["run_id"] == "r1"
        assert row["error"] is None  # pre-existing rows: no error recorded
        run = db.fetchone("SELECT run_name FROM eval_runs WHERE run_id = 'r1'")
        assert run["run_name"] == "legacy"
    finally:
        db.close()
