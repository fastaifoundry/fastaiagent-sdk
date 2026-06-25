"""Checkpointer protocol and shared models.

The :class:`Checkpointer` Protocol defines the storage surface every backend
(SQLite today, Postgres in Phase 8) must satisfy. It is the contract
``Chain``, ``Agent``, ``Swarm``, and ``Supervisor`` rely on for durability.
"""

from __future__ import annotations

import builtins
from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from fastaiagent.chain.checkpoint import Checkpoint


class PendingInterrupt(BaseModel):
    """A workflow that has called ``interrupt()`` and is waiting for resume."""

    execution_id: str
    chain_name: str
    node_id: str
    reason: str
    context: dict[str, Any] = Field(default_factory=dict)
    agent_path: str | None = None
    created_at: str = ""


@runtime_checkable
class Checkpointer(Protocol):
    """Storage protocol shared by every checkpoint backend."""

    def setup(self) -> None:
        """Run idempotent migrations. Called once before first use."""
        ...

    def put(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint."""
        ...

    def put_writes(self, execution_id: str, checkpoint_id: str, writes: list[Any]) -> None:
        """Record per-checkpoint writes (reserved for Phase 2/3)."""
        ...

    def get_last(self, execution_id: str) -> Checkpoint | None:
        """Return the most recent checkpoint for an execution, or None."""
        ...

    def get_by_id(self, execution_id: str, checkpoint_id: str) -> Checkpoint | None:
        """Return a specific checkpoint by id, or None."""
        ...

    def list(  # noqa: A003  (matches v1 spec method name)
        self, execution_id: str, *, limit: int = 100
    ) -> builtins.list[Checkpoint]:
        """Return checkpoints for an execution in chronological order."""
        ...

    def list_pending_interrupts(self, *, limit: int = 100) -> builtins.list[PendingInterrupt]:
        """Return all workflows currently suspended on ``interrupt()``."""
        ...

    def record_interrupt(self, checkpoint: Checkpoint, pending: PendingInterrupt) -> None:
        """Atomically write the interrupted checkpoint + pending_interrupts row.

        Both inserts must commit together so the ``/approvals`` UI never
        observes a half-suspended workflow.
        """
        ...

    def delete_pending_interrupt_atomic(self, execution_id: str) -> PendingInterrupt | None:
        """Claim a pending interrupt by deleting its row.

        Returns the row that was claimed, or ``None`` if no row existed (the
        caller should raise ``AlreadyResumed``). The SELECT and DELETE run in
        a single transaction so concurrent resumers can't both claim the
        same row.
        """
        ...

    def delete_execution(self, execution_id: str) -> None:
        """Delete every checkpoint and any pending interrupt for an execution."""
        ...

    def get_idempotent(self, execution_id: str, function_key: str) -> Any | None:
        """Return the cached result for an ``@idempotent`` function, or None."""
        ...

    def put_idempotent(self, execution_id: str, function_key: str, result: Any) -> None:
        """Cache the result of an ``@idempotent`` function.

        ``result`` must already be JSON-serializable — the decorator runs
        ``pydantic_core.to_jsonable_python`` first.
        """
        ...

    def prune(self, older_than: timedelta) -> int:
        """Delete completed/failed checkpoints + idempotency rows older than ``older_than``.

        Suspended (``interrupted``) checkpoints are preserved — pruning them
        would orphan an active human-in-the-loop. Returns the total number
        of rows deleted across both tables.
        """
        ...


@runtime_checkable
class ReplicatedCheckpointer(Protocol):
    """**Optional** replication surface for the connected-plane checkpoint outbox.

    Deliberately a *separate* protocol — **not** part of :class:`Checkpointer` —
    so adding the connected-durability surface stays additive and never breaks a
    third-party custom checkpointer that doesn't implement it. The SDK's own
    ``SQLiteCheckpointer`` / ``PostgresCheckpointer`` implement it;
    ``fastaiagent.checkpointers.platform_replica`` probes for it and only
    replicates checkpointers that satisfy it. A checkpointer without these methods
    simply does not replicate to the plane (a no-op, never an error).
    """

    def fetch_unsynced(
        self, limit: int, project_id: str | None = None
    ) -> builtins.list[dict[str, Any]]:
        """Return up to ``limit`` un-acked checkpoint rows (oldest first), as
        column dicts. The replicator maps them onto the ``/checkpoints/ingest``
        wire shape, so the exact row keys are an implementation detail shared with
        :mod:`fastaiagent.checkpointers.platform_replica`."""
        ...

    def mark_synced(self, checkpoint_ids: builtins.list[str]) -> None:
        """Mark checkpoints as acked (no longer re-send candidates) after a
        confirmed 2xx ingest. Idempotent."""
        ...
