"""SQLite fact reads stop after ``limit`` rows (1.82.0).

``PersistentFactBlock`` reads the newest ``max_facts`` active facts on every
turn. The only matching index covered the filter, so SQLite sorted every
matching row (``USE TEMP B-TREE FOR ORDER BY``). Postgres got partial indexes
matching the read in 1.80.0; ``local.db`` schema v24 adds the same two.

No mocking: a real ``local.db`` and SQLite's own query plan.
"""

from __future__ import annotations

import sqlite3

from fastaiagent.learn import Fact, MemoryStore
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION

_ACTIVE = "scope = ? AND project_id = ? AND superseded_by IS NULL"


def _plan(db_path: str, sql: str, params: tuple) -> str:
    with sqlite3.connect(db_path) as conn:
        return " | ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params))


def test_a_subjects_newest_facts_are_read_without_a_sort(tmp_path):
    db = str(tmp_path / "local.db")
    MemoryStore(db_path=db).add(Fact(scope="user", scope_id="u1", fact="x"))
    plan = _plan(
        db,
        f"SELECT * FROM learned_memory WHERE {_ACTIVE} AND scope_id = ? "
        "ORDER BY created_at DESC LIMIT 50",
        ("user", "", "u1"),
    )
    assert "TEMP B-TREE" not in plan, plan


def test_a_scopes_newest_facts_are_read_without_a_sort(tmp_path):
    db = str(tmp_path / "local.db")
    MemoryStore(db_path=db).add(Fact(scope="agent", scope_id="a", fact="x"))
    plan = _plan(
        db,
        f"SELECT * FROM learned_memory WHERE {_ACTIVE} ORDER BY created_at DESC LIMIT 50",
        ("agent", ""),
    )
    assert "TEMP B-TREE" not in plan, plan


def test_an_older_local_db_gains_the_source_column(tmp_path):
    """A v23 ``local.db`` keeps its facts, and they read back as not learned."""
    db = str(tmp_path / "local.db")
    MemoryStore(db_path=db).add(Fact(scope="user", scope_id="u1", fact="old fact"))
    with sqlite3.connect(db) as conn:  # rewind to v23: drop what v24 adds
        conn.execute("DROP INDEX IF EXISTS idx_learned_memory_subject_active")
        conn.execute("DROP INDEX IF EXISTS idx_learned_memory_scope_active")
        conn.execute("ALTER TABLE learned_memory DROP COLUMN source")
        conn.execute("PRAGMA user_version = 23")
    facts = MemoryStore(db_path=db).list_active(scope="user", scope_id="u1")
    assert [(f.fact, f.source) for f in facts] == [("old fact", "")]
    with sqlite3.connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_SCHEMA_VERSION >= 24
