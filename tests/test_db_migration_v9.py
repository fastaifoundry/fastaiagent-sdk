"""Migration test for schema v9 (agent-simulation tables).

Two paths, both against a real SQLite file (no mocks):

  - a fresh DB initializes straight to v9 with the ``sim_runs`` / ``sim_cases``
    tables present;
  - a DB seeded at v8 (with a learned_memory row) upgrades cleanly to v9,
    keeping its existing data and gaining the new tables.
"""

from __future__ import annotations

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db


def _user_version(db: SQLiteHelper) -> int:
    row = db.fetchone("PRAGMA user_version")
    return int(next(iter(row.values())))


def _tables(db: SQLiteHelper) -> set[str]:
    rows = db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    return {r["name"] for r in rows}


def test_fresh_db_has_v9_tables(tmp_path) -> None:
    db = init_local_db(tmp_path / "fresh.db")
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION >= 9
        tables = _tables(db)
        assert "sim_runs" in tables
        assert "sim_cases" in tables

        # Columns match the spec.
        run_cols = {r["name"] for r in db.fetchall("PRAGMA table_info(sim_runs)")}
        assert {"run_id", "agent_name", "scenario_count", "pass_rate", "project_id"} <= run_cols
        case_cols = {r["name"] for r in db.fetchall("PRAGMA table_info(sim_cases)")}
        assert {"case_id", "run_id", "transcript", "per_criterion", "criteria"} <= case_cols
    finally:
        db.close()


def test_upgrade_from_v8(tmp_path) -> None:
    db_file = tmp_path / "v8.db"

    # Seed a minimal v8-shape DB: the learned_memory table + user_version 8.
    seed = SQLiteHelper(db_file)
    seed.execute(
        """CREATE TABLE learned_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            scope_id TEXT NOT NULL DEFAULT '',
            fact TEXT NOT NULL,
            source_trace_id TEXT,
            confidence REAL DEFAULT 1.0,
            created_at REAL NOT NULL,
            superseded_by INTEGER,
            project_id TEXT NOT NULL DEFAULT '',
            UNIQUE(scope, scope_id, fact, project_id)
        )"""
    )
    seed.execute(
        "INSERT INTO learned_memory (scope, scope_id, fact, created_at) VALUES (?, ?, ?, ?)",
        ("user", "u1", "likes brevity", 1000.0),
    )
    seed.execute("PRAGMA user_version = 8")
    seed.close()

    # Re-open through init_local_db → runs the v9 migration.
    db = init_local_db(db_file)
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION
        tables = _tables(db)
        assert "sim_runs" in tables and "sim_cases" in tables
        # Pre-existing data survived.
        row = db.fetchone("SELECT fact FROM learned_memory WHERE scope_id = ?", ("u1",))
        assert row["fact"] == "likes brevity"
    finally:
        db.close()


def test_migration_is_idempotent(tmp_path) -> None:
    db_file = tmp_path / "idem.db"
    init_local_db(db_file).close()
    # Second open must be a no-op (already at CURRENT_SCHEMA_VERSION).
    db = init_local_db(db_file)
    try:
        assert _user_version(db) == CURRENT_SCHEMA_VERSION
        assert {"sim_runs", "sim_cases"} <= _tables(db)
    finally:
        db.close()
