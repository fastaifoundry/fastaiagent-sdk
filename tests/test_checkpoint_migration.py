"""Migration test: pre-existing v0.9.4 local.db files survive Phase 1.

We construct a fixture local.db at schema v1 (the 0.9.4 shape), insert a
checkpoint row using only the legacy columns, then open it through
``SQLiteCheckpointer`` (which runs the v2 migration) and verify:

  - new columns exist on the ``checkpoints`` table
  - new tables (``idempotency_cache``, ``pending_interrupts``) exist
  - the pre-existing row's ``checkpoint_id`` was backfilled from ``id``
  - the row is readable through ``SQLiteCheckpointer.get_last()``
"""

from __future__ import annotations

import json

from fastaiagent import SQLiteCheckpointer
from fastaiagent._internal.storage import SQLiteHelper

_V1_SCHEMA = """
CREATE TABLE checkpoints (
    id                 TEXT PRIMARY KEY,
    chain_name         TEXT NOT NULL,
    execution_id       TEXT NOT NULL,
    node_id            TEXT NOT NULL,
    node_index         INTEGER,
    status             TEXT DEFAULT 'completed',
    state_snapshot     TEXT DEFAULT '{}',
    node_input         TEXT DEFAULT '{}',
    node_output        TEXT DEFAULT '{}',
    iteration          INTEGER DEFAULT 0,
    iteration_counters TEXT DEFAULT '{}',
    created_at         TEXT
);
CREATE INDEX idx_cp_exec ON checkpoints (execution_id);
"""


def _seed_v1_local_db(path) -> str:
    """Create a v1-shape local.db with one legacy checkpoint row and return its id."""
    db = SQLiteHelper(path)
    for stmt in _V1_SCHEMA.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    db.execute("PRAGMA user_version = 1")

    legacy_id = "legacy-cp-uuid-1"
    db.execute(
        """INSERT INTO checkpoints
           (id, chain_name, execution_id, node_id, node_index,
            status, state_snapshot, node_input, node_output,
            iteration, iteration_counters, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            legacy_id,
            "legacy-chain",
            "legacy-exec-1",
            "legacy-node",
            0,
            "completed",
            json.dumps({"step": "legacy"}),
            "{}",
            json.dumps({"output": "ok"}),
            0,
            "{}",
            "2026-04-01T00:00:00+00:00",
        ),
    )
    db.close()
    return legacy_id


def _columns(db: SQLiteHelper, table: str) -> set[str]:
    return {r["name"] for r in db.fetchall(f"PRAGMA table_info({table})")}


def _tables(db: SQLiteHelper) -> set[str]:
    rows = db.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return {r["name"] for r in rows}


class TestCheckpointMigration:
    def test_existing_db_gains_new_columns(self, temp_dir):
        path = temp_dir / "legacy.db"
        _seed_v1_local_db(path)

        store = SQLiteCheckpointer(db_path=str(path))
        store.setup()

        with SQLiteHelper(path) as db:
            cols = _columns(db, "checkpoints")
            assert {
                "checkpoint_id",
                "parent_checkpoint_id",
                "interrupt_reason",
                "interrupt_context",
                "agent_path",
            }.issubset(cols), f"missing columns after migrate: {cols}"
        store.close()

    def test_existing_db_gains_new_tables(self, temp_dir):
        path = temp_dir / "legacy.db"
        _seed_v1_local_db(path)

        SQLiteCheckpointer(db_path=str(path)).setup()

        with SQLiteHelper(path) as db:
            tables = _tables(db)
            assert "idempotency_cache" in tables
            assert "pending_interrupts" in tables

    def test_existing_row_checkpoint_id_backfilled(self, temp_dir):
        path = temp_dir / "legacy.db"
        legacy_id = _seed_v1_local_db(path)

        SQLiteCheckpointer(db_path=str(path)).setup()

        with SQLiteHelper(path) as db:
            row = db.fetchone(
                "SELECT id, checkpoint_id FROM checkpoints WHERE id = ?",
                (legacy_id,),
            )
        assert row is not None
        assert row["checkpoint_id"] == legacy_id, (
            "v2 migration must backfill checkpoint_id = id for old rows"
        )

    def test_existing_row_readable_through_checkpointer(self, temp_dir):
        path = temp_dir / "legacy.db"
        _seed_v1_local_db(path)

        store = SQLiteCheckpointer(db_path=str(path))
        store.setup()
        latest = store.get_last("legacy-exec-1")
        assert latest is not None
        assert latest.node_id == "legacy-node"
        assert latest.state_snapshot == {"step": "legacy"}
        assert latest.checkpoint_id  # backfilled
        # Defaults for new fields should be applied cleanly.
        assert latest.parent_checkpoint_id is None
        assert latest.interrupt_reason is None
        assert latest.interrupt_context == {}
        assert latest.agent_path is None
        store.close()

    def test_user_version_is_2_after_migrate(self, temp_dir):
        path = temp_dir / "legacy.db"
        _seed_v1_local_db(path)

        SQLiteCheckpointer(db_path=str(path)).setup()

        with SQLiteHelper(path) as db:
            row = db.fetchone("PRAGMA user_version")
        assert next(iter(row.values())) == 2

    def test_setup_is_idempotent_on_already_v2_db(self, temp_dir):
        path = temp_dir / "fresh.db"

        store_a = SQLiteCheckpointer(db_path=str(path))
        store_a.setup()
        store_a.close()

        # Second setup() on an already-migrated DB must be a no-op.
        store_b = SQLiteCheckpointer(db_path=str(path))
        store_b.setup()
        with SQLiteHelper(path) as db:
            row = db.fetchone("PRAGMA user_version")
        assert next(iter(row.values())) == 2
        store_b.close()
