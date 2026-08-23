"""Migration test for schema v17 (``eval_runs.synced``, Part D).

Two paths against a real SQLite file (no mocks): a fresh DB initializes straight
to v17 with the column, and a DB seeded at v16 upgrades cleanly with pre-existing
rows backfilled to ``synced=1`` — so connecting an existing project never
back-pushes local eval history.
"""

from __future__ import annotations

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db


def _user_version(db: SQLiteHelper) -> int:
    row = db.fetchone("PRAGMA user_version")
    return int(next(iter(row.values())))


def test_fresh_db_is_v17_with_synced_column(tmp_path) -> None:
    db = init_local_db(tmp_path / "fresh.db")
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION >= 17
        cols = {r["name"] for r in db.fetchall("PRAGMA table_info(eval_runs)")}
        assert "synced" in cols
        idx = {r["name"] for r in db.fetchall("PRAGMA index_list(eval_runs)")}
        assert "idx_eval_runs_synced" in idx
    finally:
        db.close()


def test_upgrade_from_v16_backfills_existing_rows(tmp_path) -> None:
    db_file = tmp_path / "v16.db"

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
            trace_id TEXT, per_scorer TEXT, error TEXT,
            project_id TEXT NOT NULL DEFAULT ''
        )"""
    )
    seed.execute(
        "INSERT INTO eval_runs (run_id, run_name, pass_count, fail_count, pass_rate, metadata)"
        " VALUES ('legacy-1', 'pre-existing', 2, 0, 1.0, '{}')"
    )
    seed.execute("PRAGMA user_version = 16")
    seed.close()

    db = init_local_db(db_file)
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION
        row = db.fetchone("SELECT run_name, synced FROM eval_runs WHERE run_id = 'legacy-1'")
        assert row["run_name"] == "pre-existing"
        # Load-bearing: history must not become push candidates on upgrade.
        assert row["synced"] == 1
    finally:
        db.close()


def test_upgrade_survives_eval_runs_without_started_at(tmp_path) -> None:
    """A partial/hand-rolled eval_runs table must not break the upgrade.

    The drain orders by ``started_at``, but older and hand-rolled tables in the
    wild don't have it — indexing it unconditionally made init_local_db raise
    ``no such column: started_at`` (caught by tests/test_optimize_persist.py).
    """
    db_file = tmp_path / "partial.db"
    seed = SQLiteHelper(db_file)
    seed.execute(
        """CREATE TABLE eval_runs (
            run_id TEXT PRIMARY KEY, run_name TEXT, dataset_name TEXT,
            agent_name TEXT, pass_rate REAL, project_id TEXT NOT NULL DEFAULT ''
        )"""
    )
    seed.execute("INSERT INTO eval_runs (run_id, run_name) VALUES ('keep', 'legacy')")
    seed.execute("PRAGMA user_version = 14")
    seed.close()

    db = init_local_db(db_file)  # must not raise
    try:
        cols = {r["name"] for r in db.fetchall("PRAGMA table_info(eval_runs)")}
        assert "synced" in cols
        row = db.fetchone("SELECT run_name, synced FROM eval_runs WHERE run_id = 'keep'")
        assert row["run_name"] == "legacy" and row["synced"] == 1
    finally:
        db.close()


def test_migration_is_idempotent(tmp_path) -> None:
    path = tmp_path / "twice.db"
    db = init_local_db(path)
    db.close()
    db = init_local_db(path)  # re-running the ladder must not raise
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION
    finally:
        db.close()
