"""Round-trip tests for the Checkpointer protocol against SQLiteCheckpointer.

Phase 8 will parameterize this suite over PostgresCheckpointer too.
"""

from __future__ import annotations

import time
from datetime import timedelta

import pytest

from fastaiagent import Checkpointer, PendingInterrupt, SQLiteCheckpointer
from fastaiagent.chain.checkpoint import Checkpoint


@pytest.fixture
def store(temp_dir) -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=str(temp_dir / "cp.db"))
    cp.setup()
    yield cp
    cp.close()


def _make(execution_id: str, node_id: str, idx: int = 0, **extra) -> Checkpoint:
    return Checkpoint(
        chain_name="protocol-test",
        execution_id=execution_id,
        node_id=node_id,
        node_index=idx,
        state_snapshot={"step": node_id, **extra.pop("state", {})},
        **extra,
    )


class TestProtocolConformance:
    def test_satisfies_checkpointer_protocol(self, store):
        # ``runtime_checkable`` Protocol — isinstance check verifies the
        # method surface matches.
        assert isinstance(store, Checkpointer)

    def test_put_and_get_last(self, store):
        store.put(_make("exec-A", "node-1", 0))
        store.put(_make("exec-A", "node-2", 1))
        latest = store.get_last("exec-A")
        assert latest is not None
        assert latest.node_id == "node-2"
        assert latest.checkpoint_id  # auto-assigned

    def test_get_last_missing_returns_none(self, store):
        assert store.get_last("nope") is None

    def test_get_by_id(self, store):
        cp = _make("exec-B", "node-x", 0)
        store.put(cp)
        fetched = store.get_by_id("exec-B", cp.checkpoint_id)
        assert fetched is not None
        assert fetched.node_id == "node-x"
        assert store.get_by_id("exec-B", "missing-id") is None

    def test_list_orders_by_node_index(self, store):
        store.put(_make("exec-C", "n0", 0))
        store.put(_make("exec-C", "n1", 1))
        store.put(_make("exec-C", "n2", 2))
        rows = store.list("exec-C")
        assert [r.node_id for r in rows] == ["n0", "n1", "n2"]

    def test_list_respects_limit(self, store):
        for i in range(5):
            store.put(_make("exec-D", f"n{i}", i))
        rows = store.list("exec-D", limit=2)
        assert len(rows) == 2

    def test_list_pending_interrupts_empty_in_phase_1(self, store):
        # Phase 1 doesn't write to pending_interrupts itself, but the table
        # exists and the read path returns an empty list cleanly.
        assert store.list_pending_interrupts() == []

    def test_list_pending_interrupts_reads_inserts(self, store):
        # Direct insert simulates what Phase 2's interrupt() will do.
        store._conn().execute(
            """INSERT INTO pending_interrupts
               (execution_id, chain_name, node_id, reason, context,
                agent_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                "exec-PI",
                "protocol-test",
                "approval",
                "manager_approval",
                '{"amount": 50000}',
                None,
                "2026-04-29T00:00:00+00:00",
            ),
        )
        rows = store.list_pending_interrupts()
        assert len(rows) == 1
        assert isinstance(rows[0], PendingInterrupt)
        assert rows[0].reason == "manager_approval"
        assert rows[0].context == {"amount": 50000}

    def test_delete_execution_removes_everything_for_that_id(self, store):
        store.put(_make("exec-E", "n0", 0))
        store.put(_make("exec-E", "n1", 1))
        store.put(_make("exec-F", "n0", 0))  # different execution — keep
        store.put_idempotent("exec-E", "fn:k", {"ok": True})

        store.delete_execution("exec-E")

        assert store.list("exec-E") == []
        assert store.get_last("exec-E") is None
        assert store.get_idempotent("exec-E", "fn:k") is None
        # Other executions untouched.
        assert store.get_last("exec-F") is not None

    def test_prune_drops_old_checkpoints_and_idempotency_rows(self, store):
        old = _make("exec-G", "old-node", 0)
        old.created_at = "2000-01-01T00:00:00+00:00"
        store.put(old)
        store.put(_make("exec-G", "fresh-node", 1))  # auto-stamped now

        store.put_idempotent("exec-G", "stale", {"x": 1})
        # Backdate the idempotency row directly.
        store._conn().execute(
            "UPDATE idempotency_cache SET created_at = ? WHERE function_key = ?",
            ("2000-01-01T00:00:00+00:00", "stale"),
        )
        store.put_idempotent("exec-G", "fresh", {"x": 2})

        deleted = store.prune(timedelta(days=30))
        assert deleted >= 2
        remaining = store.list("exec-G")
        assert all(cp.node_id == "fresh-node" for cp in remaining)
        assert store.get_idempotent("exec-G", "stale") is None
        assert store.get_idempotent("exec-G", "fresh") == {"x": 2}


class TestIdempotencyCache:
    def test_round_trip(self, store):
        assert store.get_idempotent("exec-I", "fn:1") is None
        store.put_idempotent("exec-I", "fn:1", {"answer": 42})
        assert store.get_idempotent("exec-I", "fn:1") == {"answer": 42}

    def test_replace_existing_key(self, store):
        store.put_idempotent("exec-I", "fn:k", {"v": 1})
        store.put_idempotent("exec-I", "fn:k", {"v": 2})
        assert store.get_idempotent("exec-I", "fn:k") == {"v": 2}


class TestSetupIsIdempotent:
    def test_setup_called_twice(self, temp_dir):
        a = SQLiteCheckpointer(db_path=str(temp_dir / "x.db"))
        a.setup()
        a.setup()  # must not error
        a.put(_make("exec-X", "n0", 0))
        assert a.get_last("exec-X") is not None
        a.close()

    def test_put_auto_runs_setup(self, temp_dir):
        a = SQLiteCheckpointer(db_path=str(temp_dir / "y.db"))
        # No explicit setup() — put() should still succeed because _conn()
        # lazily migrates.
        a.put(_make("exec-Y", "n0", 0))
        assert a.get_last("exec-Y") is not None
        a.close()


def test_create_at_is_populated_when_omitted(store):
    cp = _make("exec-T", "n0")
    assert cp.created_at == ""
    store.put(cp)
    fetched = store.get_last("exec-T")
    assert fetched is not None
    assert fetched.created_at  # non-empty timestamp


def test_two_checkpoints_have_distinct_checkpoint_ids(store):
    store.put(_make("exec-U", "n0", 0))
    time.sleep(0.001)
    store.put(_make("exec-U", "n1", 1))
    rows = store.list("exec-U")
    assert len({r.checkpoint_id for r in rows}) == 2
