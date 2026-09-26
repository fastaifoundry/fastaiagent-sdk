"""Postgres fact reads stop after ``limit`` rows (1.80.0).

``PersistentFactBlock`` reads the newest ``max_facts`` active facts on every
turn. The only index used to be the ``UNIQUE`` one, so each read sorted every
matching row. Opening a store now creates partial indexes that match the read
(``scope[, scope_id], project_id, created_at DESC WHERE superseded_by IS
NULL``) — on existing tables too, since the DDL runs at every open.

No mocking: a real Postgres, gated on ``PG_TEST_DSN`` (see
``tests/integration/conftest.py``).
"""

from __future__ import annotations

import os

import pytest

PG_DSN = os.environ.get("PG_TEST_DSN")

pytestmark = pytest.mark.skipif(not PG_DSN, reason="PG_TEST_DSN not set")


def test_opening_a_store_indexes_the_per_turn_read():
    import psycopg

    from fastaiagent.learn import Fact, PostgresFactStore

    store = PostgresFactStore(PG_DSN)
    store.add(Fact(scope="agent", scope_id="index-probe", fact="probe"))
    with psycopg.connect(PG_DSN) as conn:
        rows = conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'learned_memory'"
        ).fetchall()
    defs = {name: definition for name, definition in rows}
    assert "idx_learned_memory_subject_active" in defs
    assert "idx_learned_memory_scope_active" in defs
    for name in ("idx_learned_memory_subject_active", "idx_learned_memory_scope_active"):
        assert "created_at DESC" in defs[name]
        assert "superseded_by IS NULL" in defs[name]
