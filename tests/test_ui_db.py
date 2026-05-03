"""Tests for fastaiagent.ui.db — unified local.db schema + migrations."""

from __future__ import annotations

from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db

EXPECTED_TABLES = {
    "spans",
    "checkpoints",
    "prompts",
    "prompt_versions",
    "prompt_aliases",
    "prompt_fragments",
    "eval_runs",
    "eval_cases",
    "guardrail_events",
    "trace_notes",
    "trace_favorites",
    "saved_filters",
}


def _list_tables(db) -> set[str]:
    rows = db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {row["name"] for row in rows}


class TestInitLocalDB:
    def test_creates_all_expected_tables(self, temp_dir):
        db = init_local_db(temp_dir / "local.db")
        try:
            assert EXPECTED_TABLES.issubset(_list_tables(db))
        finally:
            db.close()

    def test_sets_user_version(self, temp_dir):
        db = init_local_db(temp_dir / "local.db")
        try:
            row = db.fetchone("PRAGMA user_version")
            assert row is not None
            assert next(iter(row.values())) == CURRENT_SCHEMA_VERSION
        finally:
            db.close()

    def test_is_idempotent(self, temp_dir):
        path = temp_dir / "local.db"
        db = init_local_db(path)
        db.close()
        # Second call on an already-migrated DB must not error or duplicate work.
        db2 = init_local_db(path)
        try:
            row = db2.fetchone("PRAGMA user_version")
            assert next(iter(row.values())) == CURRENT_SCHEMA_VERSION
            assert EXPECTED_TABLES.issubset(_list_tables(db2))
        finally:
            db2.close()

    def test_respects_explicit_path(self, temp_dir):
        custom = temp_dir / "nested" / "custom.db"
        db = init_local_db(custom)
        try:
            assert custom.exists()
            assert EXPECTED_TABLES.issubset(_list_tables(db))
        finally:
            db.close()

    def test_resolves_from_config_when_path_omitted(self, temp_dir, monkeypatch):
        from fastaiagent._internal.config import reset_config

        default_path = temp_dir / "via-config.db"
        monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(default_path))
        reset_config()
        try:
            db = init_local_db()
            try:
                assert default_path.exists()
                assert EXPECTED_TABLES.issubset(_list_tables(db))
            finally:
                db.close()
        finally:
            reset_config()

    def test_schema_tables_accept_inserts(self, temp_dir):
        """Smoke-test every table is writable with a representative insert."""
        db = init_local_db(temp_dir / "local.db")
        try:
            db.execute(
                "INSERT INTO spans (span_id, trace_id, name) VALUES (?, ?, ?)",
                ("s1", "t1", "example"),
            )
            db.execute(
                """INSERT INTO checkpoints
                   (id, chain_name, execution_id, node_id)
                   VALUES (?, ?, ?, ?)""",
                ("cp1", "chain", "exec1", "node1"),
            )
            db.execute(
                "INSERT INTO prompts (slug, latest_version) VALUES (?, ?)",
                ("my-prompt", "v1"),
            )
            db.execute(
                """INSERT INTO prompt_versions (slug, version, template)
                   VALUES (?, ?, ?)""",
                ("my-prompt", "v1", "hello {{name}}"),
            )
            db.execute(
                """INSERT INTO eval_runs
                   (run_id, run_name, pass_count, fail_count, pass_rate)
                   VALUES (?, ?, ?, ?, ?)""",
                ("run1", "smoke", 5, 1, 0.833),
            )
            db.execute(
                """INSERT INTO eval_cases (case_id, run_id, ordinal)
                   VALUES (?, ?, ?)""",
                ("case1", "run1", 0),
            )
            db.execute(
                """INSERT INTO guardrail_events
                   (event_id, guardrail_name, outcome)
                   VALUES (?, ?, ?)""",
                ("evt1", "no_pii", "passed"),
            )
            db.execute(
                "INSERT INTO trace_notes (trace_id, note) VALUES (?, ?)",
                ("t1", "interesting"),
            )
            db.execute(
                "INSERT INTO trace_favorites (trace_id) VALUES (?)",
                ("t1",),
            )
            db.execute(
                "INSERT INTO saved_filters (id, name, filters) VALUES (?, ?, ?)",
                ("f1", "failing traces", "{\"status\":\"error\"}"),
            )
        finally:
            db.close()


class TestV6Migration:
    """Sprint 3 — span_fts virtual table + sync triggers + saved_filters
    project_id. Each test exercises the migration on a fresh DB so the
    assertions reflect what the route layer can rely on."""

    def test_span_fts_table_and_triggers_exist(self, temp_dir):
        db = init_local_db(temp_dir / "local.db")
        try:
            tables = {
                r["name"]
                for r in db.fetchall(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert "span_fts" in tables
            triggers = {
                r["name"]
                for r in db.fetchall(
                    "SELECT name FROM sqlite_master WHERE type='trigger'"
                )
            }
            assert {"spans_fts_ai", "spans_fts_au", "spans_fts_ad"}.issubset(triggers)
        finally:
            db.close()

    def test_saved_filters_has_project_id(self, temp_dir):
        db = init_local_db(temp_dir / "local.db")
        try:
            cols = {r["name"] for r in db.fetchall("PRAGMA table_info(saved_filters)")}
            assert "project_id" in cols
        finally:
            db.close()

    def test_v6_is_idempotent_no_double_index(self, temp_dir):
        path = temp_dir / "local.db"
        # First run lands the v6 schema.
        db = init_local_db(path)
        db.close()
        # Second run must be a no-op (no errors, no duplicate FTS rows
        # for any spans we wrote between).
        db = init_local_db(path)
        try:
            import json

            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, name, attributes, events)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    "s-once",
                    "t-once",
                    "llm.gpt",
                    json.dumps({"gen_ai.prompt": "uniqueterm"}),
                    "[]",
                ),
            )
        finally:
            db.close()
        db = init_local_db(path)
        try:
            row = db.fetchone(
                "SELECT COUNT(*) AS n FROM span_fts WHERE span_id = ?",
                ("s-once",),
            )
            assert row is not None
            assert row["n"] == 1, "second init_local_db call duplicated FTS rows"
        finally:
            db.close()

    def test_backfill_picks_up_pre_v6_spans(self, temp_dir):
        """Spans inserted before v6 still need to land in span_fts via
        the bulk backfill the migration runs."""
        import sqlite3
        import json

        path = temp_dir / "legacy.db"
        # Seed a v5-shaped DB by hand so the v6 backfill has work to do.
        # Use raw sqlite3 so we don't trigger init_local_db's migration
        # path before we want it.
        with sqlite3.connect(path) as conn:
            conn.execute(
                """CREATE TABLE spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    status TEXT DEFAULT 'OK',
                    attributes TEXT DEFAULT '{}',
                    events TEXT DEFAULT '[]',
                    project_id TEXT NOT NULL DEFAULT ''
                )"""
            )
            for i in range(20):
                conn.execute(
                    "INSERT INTO spans (span_id, trace_id, name, attributes) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        f"legacy-{i}",
                        f"trace-{i}",
                        "llm.gpt",
                        json.dumps({"gen_ai.prompt": f"phrase {i}"}),
                    ),
                )
            conn.execute("PRAGMA user_version = 5")
            conn.commit()

        # Now run the migration — v6 should backfill all 20.
        db = init_local_db(path)
        try:
            row = db.fetchone("SELECT COUNT(*) AS n FROM span_fts")
            assert row is not None
            assert row["n"] == 20
        finally:
            db.close()
