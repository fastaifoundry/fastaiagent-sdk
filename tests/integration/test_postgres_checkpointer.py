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


def _sync_errors(store: Any, execution_id: str) -> dict[str, str | None]:
    """``{node_id: sync_error}`` straight from the row, for both backends.

    Read raw rather than through ``Checkpoint``: ``sync_error`` is an outbox
    column, not part of the run state, and deliberately absent from the model.
    """
    if isinstance(store, SQLiteCheckpointer):
        rows = store._conn().fetchall(
            "SELECT node_id, sync_error FROM checkpoints WHERE execution_id = ?",
            (execution_id,),
        )
        return {r["node_id"]: r["sync_error"] for r in rows}
    pool = store._get_pool()
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT node_id, sync_error FROM {store._t_checkpoints} WHERE execution_id = %s",
            (execution_id,),
        )
        return {r[0]: r[1] for r in cur.fetchall()}


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

    def test_reusing_a_checkpoint_id_raises_on_both_backends(self, store: Checkpointer) -> None:
        """Durability audit D3 — the two backends must agree on id reuse.

        SQLite has always raised (its ``id`` is the primary key and ``put`` is a
        plain INSERT). Postgres used to ``ON CONFLICT DO UPDATE``, and the
        divergence was invisible until the plane replica existed: its ingest
        door is INSERT-ONLY, so a rewritten row re-pushed as a duplicate came
        back ``{"ingested": 0}``, the SDK marked it synced, and the plane kept
        the FIRST version. Local said ``completed``, the replica said
        ``interrupted``, and nothing anywhere noticed.

        The exception type is the driver's, not an SDK type — "matching SQLite"
        means matching what it actually does, and SQLite propagates
        ``sqlite3.IntegrityError`` untouched. Both are DBAPI ``IntegrityError``
        subclasses, which is what a portable caller catches.
        """
        first = _make("exec-DUP", "node-1", 0)
        store.put(first)

        rewrite = _make("exec-DUP", "node-1", 0, status="interrupted")
        rewrite.checkpoint_id = first.checkpoint_id
        with pytest.raises(Exception) as excinfo:
            store.put(rewrite)
        assert "IntegrityError" in type(excinfo.value).__mro__[0].__name__ or any(
            "IntegrityError" in c.__name__ for c in type(excinfo.value).__mro__
        ), f"expected an integrity error, got {type(excinfo.value).__name__}"

        # And the stored row is untouched — a refused write must not half-apply.
        stored = store.get_by_id("exec-DUP", first.checkpoint_id)
        assert stored is not None and stored.status == "completed"

    def test_a_distinct_id_for_the_same_node_still_works(self, store: Checkpointer) -> None:
        """Refusing reuse must not refuse a legitimate re-run of the same node.

        A chain cycle re-executes a node and writes a SECOND checkpoint for it
        with a fresh id — the executors never re-use one. That path has to stay
        open, or this fix would break every looping chain.
        """
        store.put(_make("exec-CYCLE", "loop-node", 0))
        store.put(_make("exec-CYCLE", "loop-node", 0))
        assert len(store.list("exec-CYCLE")) == 2

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


# ---------- Replication outbox surface ----------------------------------


@pytest.mark.parametrize("store", ["sqlite", "postgres"], indirect=True)
class TestReplicationOutbox:
    """``ReplicatedCheckpointer`` + ``QuarantinableCheckpointer``, both backends.

    The two backends reach the same states by opposite mechanics — SQLite defaults
    ``synced`` to 0 and back-fills, Postgres defaults it to TRUE and writes FALSE
    on every insert — so parity here is worth asserting rather than assuming. It
    is exactly the kind of drift that let ``put`` upsert on one side and raise on
    the other for a whole release.
    """

    def test_satisfies_the_optional_protocols(self, store: Checkpointer) -> None:
        from fastaiagent.checkpointers.protocol import (
            QuarantinableCheckpointer,
            ReplicatedCheckpointer,
        )

        assert isinstance(store, ReplicatedCheckpointer)
        assert isinstance(store, QuarantinableCheckpointer)

    def test_new_rows_are_push_candidates_and_mark_synced_clears_them(
        self, store: Checkpointer
    ) -> None:
        store.put(_make("exec-OUT", "node-1", 0))
        store.put(_make("exec-OUT", "node-2", 1))

        pending = store.fetch_unsynced(10)  # type: ignore[attr-defined]
        assert len(pending) == 2, "a freshly written checkpoint must be a push candidate"

        store.mark_synced([r["checkpoint_id"] for r in pending])  # type: ignore[attr-defined]
        assert store.fetch_unsynced(10) == []  # type: ignore[attr-defined]

    def test_mark_quarantined_stops_the_resend_and_records_the_reason(
        self, store: Checkpointer
    ) -> None:
        """Durability audit D2 — the poison row must leave the outbox.

        Quarantined and synced share the ``synced`` flag (both mean "no longer a
        re-send candidate"), and ``sync_error`` is what keeps them distinguishable:
        a synced row reached the plane, a quarantined one never will.
        """
        store.put(_make("exec-Q", "poison", 0))
        store.put(_make("exec-Q", "healthy", 1))
        pending = store.fetch_unsynced(10)  # type: ignore[attr-defined]
        by_node = {r["node_id"]: r["checkpoint_id"] for r in pending}

        store.mark_quarantined(  # type: ignore[attr-defined]
            {by_node["poison"]: "state_snapshot is 2000000 bytes; the limit is 1048576"}
        )
        store.mark_synced([by_node["healthy"]])  # type: ignore[attr-defined]

        assert store.fetch_unsynced(10) == [], "a quarantined row must not come back"  # type: ignore[attr-defined]

        errors = _sync_errors(store, "exec-Q")
        assert "2000000 bytes" in (errors["poison"] or "")
        assert errors["healthy"] is None, "a row that reached the plane carries no error"

    def test_mark_quarantined_is_idempotent_and_empty_is_a_no_op(self, store: Checkpointer) -> None:
        store.put(_make("exec-Q2", "poison", 0))
        cid = store.fetch_unsynced(10)[0]["checkpoint_id"]  # type: ignore[attr-defined]

        store.mark_quarantined({})  # type: ignore[attr-defined]
        assert len(store.fetch_unsynced(10)) == 1  # type: ignore[attr-defined]

        store.mark_quarantined({cid: "first"})  # type: ignore[attr-defined]
        store.mark_quarantined({cid: "second"})  # type: ignore[attr-defined]
        assert store.fetch_unsynced(10) == []  # type: ignore[attr-defined]
        assert _sync_errors(store, "exec-Q2")["poison"] == "second"
