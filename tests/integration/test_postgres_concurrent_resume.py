"""Concurrent-resume integration test for ``PostgresCheckpointer``.

Spec test #12: when N threads / processes simultaneously call
``Chain.resume(execution_id, resume_value=Resume(...))`` for the same
suspended workflow, **exactly one** wins the atomic claim and runs the
chain to completion; everyone else sees :class:`AlreadyResumed`.

Postgres MVCC + ``DELETE … RETURNING *`` makes this guarantee easy to test
deterministically: only one transaction's ``DELETE`` succeeds at the
``pending_interrupts`` row; the other transactions' ``RETURNING`` clause
returns no rows.

Gated on ``PG_TEST_DSN``. The same contract is verified for SQLite as a
sanity check, though SQLite's single-writer model makes the race window
smaller in practice.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

import pytest

from fastaiagent import (
    AlreadyResumed,
    Checkpointer,
    PendingInterrupt,
    SQLiteCheckpointer,
)
from fastaiagent.chain.checkpoint import Checkpoint

PG_DSN = os.environ.get("PG_TEST_DSN")


def _skip_if_no_postgres(backend: str) -> None:
    if backend == "postgres" and not PG_DSN:
        pytest.skip("PG_TEST_DSN not set — skipping Postgres integration tests")


@pytest.fixture
def store(request: pytest.FixtureRequest, tmp_path: Any) -> Checkpointer:
    backend = request.param
    _skip_if_no_postgres(backend)

    if backend == "sqlite":
        cp = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
        cp.setup()
        return cp

    from fastaiagent.checkpointers.postgres import PostgresCheckpointer

    schema = f"fastaiagent_t_{uuid.uuid4().hex[:8]}"
    pg = PostgresCheckpointer(PG_DSN, schema=schema)
    pg.setup()

    # Drop the per-test schema on teardown.
    def _cleanup() -> None:
        pool = pg._get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            conn.commit()
        pg.close()

    request.addfinalizer(_cleanup)
    return pg


def _seed_pending(store: Checkpointer, execution_id: str) -> None:
    """Write an interrupted checkpoint + pending_interrupts row."""
    store.record_interrupt(
        Checkpoint(
            chain_name="concurrent-resume",
            execution_id=execution_id,
            node_id="approval",
            node_index=0,
            status="interrupted",
            state_snapshot={"messages": []},
            interrupt_reason="manager_approval",
            interrupt_context={"amount": 50_000},
            agent_path="agent:test/tool:approve",
        ),
        PendingInterrupt(
            execution_id=execution_id,
            chain_name="concurrent-resume",
            node_id="approval",
            reason="manager_approval",
            context={"amount": 50_000},
            agent_path="agent:test/tool:approve",
        ),
    )


@pytest.mark.parametrize("store", ["sqlite", "postgres"], indirect=True)
class TestConcurrentResume:
    """Spec test #12 — exactly-one wins, the rest see AlreadyResumed."""

    def test_eight_threads_one_wins(self, store: Checkpointer) -> None:
        """Eight threads race to claim the pending row; exactly one wins.

        We model what the SDK's ``Chain.resume`` /
        ``Agent.aresume`` do internally: call
        ``store.delete_pending_interrupt_atomic(execution_id)`` and treat
        ``None`` as :class:`AlreadyResumed`.
        """
        execution_id = f"concurrent-{uuid.uuid4().hex[:8]}"
        _seed_pending(store, execution_id)

        winners: list[PendingInterrupt] = []
        losers: list[BaseException] = []
        winners_lock = threading.Lock()
        losers_lock = threading.Lock()
        barrier = threading.Barrier(8)

        def _try_claim() -> None:
            barrier.wait()
            try:
                row = store.delete_pending_interrupt_atomic(execution_id)
                if row is None:
                    raise AlreadyResumed(execution_id)
                with winners_lock:
                    winners.append(row)
            except AlreadyResumed as e:
                with losers_lock:
                    losers.append(e)

        threads = [threading.Thread(target=_try_claim) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(winners) == 1, (
            f"expected exactly one winner, got {len(winners)} winners and {len(losers)} losers"
        )
        assert len(losers) == 7
        assert winners[0].context == {"amount": 50_000}

    def test_second_serial_resume_after_winner_raises(self, store: Checkpointer) -> None:
        """After the winner's claim, a later resume sees no pending row."""
        execution_id = f"serial-{uuid.uuid4().hex[:8]}"
        _seed_pending(store, execution_id)

        first = store.delete_pending_interrupt_atomic(execution_id)
        assert first is not None

        second = store.delete_pending_interrupt_atomic(execution_id)
        assert second is None  # the SDK raises AlreadyResumed at this point
