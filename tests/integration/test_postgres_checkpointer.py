"""Integration tests for ``PostgresCheckpointer`` (spec test #11).

Gated on ``PG_TEST_DSN``. Locally run with::

    docker run -d --name pg-dev -e POSTGRES_PASSWORD=test \\
        -e POSTGRES_DB=fastaiagent_test -p 127.0.0.1:55432:5432 \\
        postgres:16-alpine
    PG_TEST_DSN=postgresql://postgres:test@127.0.0.1:55432/fastaiagent_test \\
        pytest tests/integration/test_postgres_checkpointer.py

CI runs these against a service container (see ``.github/workflows/ci.yml``).

The tests parameterize over both ``SQLiteCheckpointer`` and
``PostgresCheckpointer`` so the same protocol contract is exercised end-to-end
against each backend — drift between them surfaces immediately.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest

from fastaiagent import Checkpointer, PendingInterrupt, SQLiteCheckpointer
from fastaiagent.chain.checkpoint import Checkpoint

PG_DSN = os.environ.get("PG_TEST_DSN")


def _skip_if_no_postgres(backend: str) -> None:
    if backend == "postgres" and not PG_DSN:
        pytest.skip("PG_TEST_DSN not set — skipping Postgres integration tests")


@pytest.fixture
def store(request: pytest.FixtureRequest, tmp_path: Any) -> Iterator[Checkpointer]:
    """Yield a fresh checkpointer for the requested backend."""
    backend = request.param
    _skip_if_no_postgres(backend)

    cp: Checkpointer
    if backend == "sqlite":
        cp = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
        cp.setup()
        try:
            yield cp
        finally:
            cp.close()  # type: ignore[attr-defined]
    else:
        from fastaiagent.checkpointers.postgres import PostgresCheckpointer

        # Each Postgres test runs in its own schema so parallel tests don't
        # collide on the same checkpoints / pending_interrupts rows.
        schema = f"fastaiagent_t_{uuid.uuid4().hex[:8]}"
        pg = PostgresCheckpointer(PG_DSN, schema=schema)
        pg.setup()
        try:
            yield pg
        finally:
            # Drop the per-test schema so we leave the DB clean.
            pool = pg._get_pool()
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                conn.commit()
            pg.close()


def _make(execution_id: str, node_id: str, idx: int = 0, **extra: Any) -> Checkpoint:
    return Checkpoint(
        chain_name="protocol-test",
        execution_id=execution_id,
        node_id=node_id,
        node_index=idx,
        state_snapshot={"step": node_id, **extra.pop("state", {})},
        **extra,
    )


# ---------- Round-trip protocol contract --------------------------------


@pytest.mark.parametrize("store", ["sqlite", "postgres"], indirect=True)
class TestProtocolConformance:
    """Spec test #11 — same protocol surface, both backends."""

    def test_satisfies_checkpointer_protocol(self, store: Checkpointer) -> None:
        assert isinstance(store, Checkpointer)

    def test_put_and_get_last(self, store: Checkpointer) -> None:
        store.put(_make("exec-A", "node-1", 0))
        store.put(_make("exec-A", "node-2", 1))
        latest = store.get_last("exec-A")
        assert latest is not None
        assert latest.node_id == "node-2"
        assert latest.checkpoint_id

    def test_get_last_missing_returns_none(self, store: Checkpointer) -> None:
        assert store.get_last("nope") is None

    def test_get_by_id(self, store: Checkpointer) -> None:
        cp = _make("exec-B", "node-x", 0)
        store.put(cp)
        fetched = store.get_by_id("exec-B", cp.checkpoint_id)
        assert fetched is not None
        assert fetched.node_id == "node-x"
        assert store.get_by_id("exec-B", "missing-id") is None

    def test_list_orders_chronologically(self, store: Checkpointer) -> None:
        store.put(_make("exec-C", "n0", 0))
        store.put(_make("exec-C", "n1", 1))
        store.put(_make("exec-C", "n2", 2))
        rows = store.list("exec-C")
        assert [r.node_id for r in rows] == ["n0", "n1", "n2"]

    def test_list_respects_limit(self, store: Checkpointer) -> None:
        for i in range(5):
            store.put(_make("exec-D", f"n{i}", i))
        rows = store.list("exec-D", limit=2)
        assert len(rows) == 2

    def test_list_pending_interrupts_empty(self, store: Checkpointer) -> None:
        assert store.list_pending_interrupts() == []

    def test_record_and_list_interrupt(self, store: Checkpointer) -> None:
        ckpt = _make("exec-PI", "approval", 0, status="interrupted")
        ckpt.interrupt_reason = "manager_approval"
        ckpt.interrupt_context = {"amount": 50_000}
        ckpt.agent_path = "agent:test/tool:approve"
        pending = PendingInterrupt(
            execution_id="exec-PI",
            chain_name="protocol-test",
            node_id="approval",
            reason="manager_approval",
            context={"amount": 50_000},
            agent_path="agent:test/tool:approve",
        )
        store.record_interrupt(ckpt, pending)

        rows = store.list_pending_interrupts()
        assert any(r.execution_id == "exec-PI" for r in rows)
        latest = store.get_last("exec-PI")
        assert latest is not None
        assert latest.status == "interrupted"
        assert latest.interrupt_reason == "manager_approval"
        assert latest.interrupt_context == {"amount": 50_000}

    def test_delete_pending_interrupt_atomic_claim(self, store: Checkpointer) -> None:
        ckpt = _make("exec-CL", "approval", 0, status="interrupted")
        ckpt.agent_path = "agent:test/tool:approve"
        pending = PendingInterrupt(
            execution_id="exec-CL",
            chain_name="protocol-test",
            node_id="approval",
            reason="r",
            context={"k": "v"},
            agent_path="agent:test/tool:approve",
        )
        store.record_interrupt(ckpt, pending)

        first = store.delete_pending_interrupt_atomic("exec-CL")
        assert first is not None
        assert first.context == {"k": "v"}

        # Second claim returns None.
        assert store.delete_pending_interrupt_atomic("exec-CL") is None

    def test_delete_execution_clears_all_three_tables(self, store: Checkpointer) -> None:
        store.put(_make("exec-E", "n0", 0))
        store.put(_make("exec-E", "n1", 1))
        store.put(_make("exec-F", "n0", 0))  # different execution — keep
        store.put_idempotent("exec-E", "fn:k", {"ok": True})

        store.delete_execution("exec-E")

        assert store.list("exec-E") == []
        assert store.get_last("exec-E") is None
        assert store.get_idempotent("exec-E", "fn:k") is None
        assert store.get_last("exec-F") is not None

    def test_idempotent_round_trip_and_replace(self, store: Checkpointer) -> None:
        assert store.get_idempotent("exec-I", "fn:1") is None
        store.put_idempotent("exec-I", "fn:1", {"answer": 42})
        assert store.get_idempotent("exec-I", "fn:1") == {"answer": 42}
        store.put_idempotent("exec-I", "fn:1", {"answer": 100})
        assert store.get_idempotent("exec-I", "fn:1") == {"answer": 100}

    def test_prune_drops_completed_and_idempotency_rows(self, store: Checkpointer) -> None:
        # Old row — manually backdated.
        old = _make("exec-G", "old-node", 0)
        old.created_at = "2000-01-01T00:00:00+00:00"
        store.put(old)
        store.put(_make("exec-G", "fresh-node", 1))

        store.put_idempotent("exec-G", "stale", {"x": 1})
        store.put_idempotent("exec-G", "fresh", {"x": 2})

        # Backdate the stale idempotency row.
        if isinstance(store, SQLiteCheckpointer):
            with store._conn()._lock:
                conn = store._conn()._get_conn()
                conn.execute(
                    "UPDATE idempotency_cache SET created_at = ? WHERE function_key = ?",
                    ("2000-01-01T00:00:00+00:00", "stale"),
                )
                conn.commit()
        else:
            from fastaiagent.checkpointers.postgres import PostgresCheckpointer

            assert isinstance(store, PostgresCheckpointer)
            pool = store._get_pool()
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {store._t_idempotency} SET created_at = %s WHERE function_key = %s",
                    ("2000-01-01T00:00:00+00:00", "stale"),
                )
                conn.commit()

        deleted = store.prune(timedelta(days=30))
        assert deleted >= 2
        remaining = store.list("exec-G")
        assert all(cp.node_id == "fresh-node" for cp in remaining)
        assert store.get_idempotent("exec-G", "stale") is None
        assert store.get_idempotent("exec-G", "fresh") == {"x": 2}

    def test_prune_preserves_interrupted_checkpoints(self, store: Checkpointer) -> None:
        # An old interrupted checkpoint must NOT be pruned — there's a real
        # human waiting for it.
        old = _make("exec-INT", "approval", 0, status="interrupted")
        old.created_at = "2000-01-01T00:00:00+00:00"
        old.interrupt_reason = "old approval"
        store.put(old)

        store.prune(timedelta(days=1))

        latest = store.get_last("exec-INT")
        assert latest is not None
        assert latest.status == "interrupted"
