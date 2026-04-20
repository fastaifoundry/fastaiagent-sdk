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
