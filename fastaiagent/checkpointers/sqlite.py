"""SQLite-backed checkpointer.

This is the default backend for local development and tests. The schema
lives in :mod:`fastaiagent.ui.db` (the unified ``local.db``); ``setup()``
delegates to the ``init_local_db`` migration runner so checkpoints share
storage with traces, prompts, and other local artefacts.
"""

from __future__ import annotations

import builtins
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.checkpointers.protocol import PendingInterrupt


class SQLiteCheckpointer:
    """SQLite-backed implementation of the :class:`Checkpointer` Protocol."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or get_config().resolved_checkpoint_db_path
        self._db: SQLiteHelper | None = None
        self._setup_done = False

    # --- lifecycle ----------------------------------------------------

    def setup(self) -> None:
        """Run idempotent migrations against the unified local.db schema."""
        from fastaiagent.ui.db import init_local_db

        # init_local_db is itself idempotent; we still gate to avoid the
        # extra PRAGMA round-trip on every put().
        if self._setup_done:
            return
        # init_local_db opens its own helper and runs migrations; we then
        # reuse our own helper for subsequent reads/writes (both share the
        # same file).
        helper = init_local_db(self.db_path)
        helper.close()
        self._setup_done = True

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _conn(self) -> SQLiteHelper:
        if not self._setup_done:
            self.setup()
        if self._db is None:
            self._db = SQLiteHelper(self.db_path)
        return self._db

    # --- writes -------------------------------------------------------

    def put(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint. Fills checkpoint_id / created_at if missing."""
        if not checkpoint.checkpoint_id:
            checkpoint.checkpoint_id = str(uuid.uuid4())
        if not checkpoint.created_at:
            checkpoint.created_at = datetime.now(tz=timezone.utc).isoformat()

        # ``id`` is the legacy primary key column; mirror checkpoint_id into
        # it so old SELECTs and the migrator's INSERT OR IGNORE still work.
        from fastaiagent._internal.project import safe_get_project_id

        self._conn().execute(
            """INSERT INTO checkpoints
               (id, checkpoint_id, parent_checkpoint_id, chain_name,
                execution_id, node_id, node_index, status,
                state_snapshot, node_input, node_output,
                iteration, iteration_counters,
                interrupt_reason, interrupt_context, agent_path,
                created_at, project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.checkpoint_id,
                checkpoint.checkpoint_id,
                checkpoint.parent_checkpoint_id,
                checkpoint.chain_name,
                checkpoint.execution_id,
                checkpoint.node_id,
                checkpoint.node_index,
                checkpoint.status,
                json.dumps(checkpoint.state_snapshot),
                json.dumps(checkpoint.node_input),
                json.dumps(checkpoint.node_output),
                checkpoint.iteration,
                json.dumps(checkpoint.iteration_counters),
                checkpoint.interrupt_reason,
                json.dumps(checkpoint.interrupt_context),
                checkpoint.agent_path,
                checkpoint.created_at,
                safe_get_project_id(),
            ),
        )

    def put_writes(self, execution_id: str, checkpoint_id: str, writes: list[Any]) -> None:
        """No-op in Phase 1 — Phase 2/3 will wire this up."""
        return None

    # --- reads --------------------------------------------------------

    def get_last(self, execution_id: str) -> Checkpoint | None:
        # Order by ``created_at`` (when the row was written) so multi-level
        # topologies — Swarm/Agent/Chain mixed under one execution — return
        # the most recently committed checkpoint, not the one with the
        # numerically largest ``node_index`` (which only made sense when
        # checkpoints were a flat chain). ``rowid`` breaks same-microsecond
        # ties via SQLite's strictly-monotonic insertion order.
        row = self._conn().fetchone(
            """SELECT * FROM checkpoints
               WHERE execution_id = ?
               ORDER BY created_at DESC, rowid DESC
               LIMIT 1""",
            (execution_id,),
        )
        return self._row_to_checkpoint(row) if row else None

    def get_by_id(self, execution_id: str, checkpoint_id: str) -> Checkpoint | None:
        row = self._conn().fetchone(
            """SELECT * FROM checkpoints
               WHERE execution_id = ? AND checkpoint_id = ?
               LIMIT 1""",
            (execution_id, checkpoint_id),
        )
        return self._row_to_checkpoint(row) if row else None

    def list(  # noqa: A003  (matches Checkpointer protocol method name)
        self, execution_id: str, *, limit: int = 100
    ) -> builtins.list[Checkpoint]:
        rows = self._conn().fetchall(
            """SELECT * FROM checkpoints
               WHERE execution_id = ?
               ORDER BY node_index ASC, created_at ASC
               LIMIT ?""",
            (execution_id, int(limit)),
        )
        return [self._row_to_checkpoint(r) for r in rows]

    def list_pending_interrupts(self, *, limit: int = 100) -> builtins.list[PendingInterrupt]:
        rows = self._conn().fetchall(
            """SELECT execution_id, chain_name, node_id, reason, context,
                      agent_path, created_at
               FROM pending_interrupts
               ORDER BY created_at DESC
               LIMIT ?""",
            (int(limit),),
        )
        return [
            PendingInterrupt(
                execution_id=r["execution_id"],
                chain_name=r["chain_name"],
                node_id=r["node_id"],
                reason=r["reason"],
                context=json.loads(r["context"]) if r["context"] else {},
                agent_path=r["agent_path"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # --- interrupt coordination --------------------------------------

    def record_interrupt(self, checkpoint: Checkpoint, pending: PendingInterrupt) -> None:
        """Insert the interrupted checkpoint and pending row in one txn."""
        if not checkpoint.checkpoint_id:
            checkpoint.checkpoint_id = str(uuid.uuid4())
        if not checkpoint.created_at:
            checkpoint.created_at = datetime.now(tz=timezone.utc).isoformat()
        if not pending.created_at:
            pending.created_at = checkpoint.created_at

        from fastaiagent._internal.project import safe_get_project_id

        pid = safe_get_project_id()
        db = self._conn()
        # Reach for the underlying sqlite3 connection so we can wrap the
        # two inserts in a single explicit transaction. SQLiteHelper.execute
        # auto-commits per call, which would split this into two txns.
        with db._lock:
            conn = db._get_conn()
            try:
                conn.execute("BEGIN")
                conn.execute(
                    """INSERT INTO checkpoints
                       (id, checkpoint_id, parent_checkpoint_id, chain_name,
                        execution_id, node_id, node_index, status,
                        state_snapshot, node_input, node_output,
                        iteration, iteration_counters,
                        interrupt_reason, interrupt_context, agent_path,
                        created_at, project_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.checkpoint_id,
                        checkpoint.parent_checkpoint_id,
                        checkpoint.chain_name,
                        checkpoint.execution_id,
                        checkpoint.node_id,
                        checkpoint.node_index,
                        checkpoint.status,
                        json.dumps(checkpoint.state_snapshot),
                        json.dumps(checkpoint.node_input),
                        json.dumps(checkpoint.node_output),
                        checkpoint.iteration,
                        json.dumps(checkpoint.iteration_counters),
                        checkpoint.interrupt_reason,
                        json.dumps(checkpoint.interrupt_context),
                        checkpoint.agent_path,
                        checkpoint.created_at,
                        pid,
                    ),
                )
                conn.execute(
                    """INSERT INTO pending_interrupts
                       (execution_id, chain_name, node_id, reason, context,
                        agent_path, created_at, project_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pending.execution_id,
                        pending.chain_name,
                        pending.node_id,
                        pending.reason,
                        json.dumps(pending.context),
                        pending.agent_path,
                        pending.created_at,
                        pid,
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def delete_pending_interrupt_atomic(self, execution_id: str) -> PendingInterrupt | None:
        """Claim a pending interrupt by SELECT-then-DELETE in one txn."""
        db = self._conn()
        with db._lock:
            conn = db._get_conn()
            try:
                conn.execute("BEGIN")
                cursor = conn.execute(
                    """SELECT execution_id, chain_name, node_id, reason,
                              context, agent_path, created_at
                       FROM pending_interrupts WHERE execution_id = ?""",
                    (execution_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    conn.commit()
                    return None
                conn.execute(
                    "DELETE FROM pending_interrupts WHERE execution_id = ?",
                    (execution_id,),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return PendingInterrupt(
            execution_id=row["execution_id"],
            chain_name=row["chain_name"],
            node_id=row["node_id"],
            reason=row["reason"],
            context=json.loads(row["context"]) if row["context"] else {},
            agent_path=row["agent_path"],
            created_at=row["created_at"],
        )

    # --- deletes / prune ---------------------------------------------

    def delete_execution(self, execution_id: str) -> None:
        db = self._conn()
        db.execute("DELETE FROM checkpoints WHERE execution_id = ?", (execution_id,))
        db.execute(
            "DELETE FROM pending_interrupts WHERE execution_id = ?",
            (execution_id,),
        )
        db.execute(
            "DELETE FROM idempotency_cache WHERE execution_id = ?",
            (execution_id,),
        )

    def prune(self, older_than: timedelta) -> int:
        cutoff = (datetime.now(tz=timezone.utc) - older_than).isoformat()
        db = self._conn()
        # Skip ``interrupted`` rows so pending HITL workflows are not orphaned
        # when an operator runs prune() on a maintenance schedule.
        cursor = db.execute(
            """DELETE FROM checkpoints
               WHERE created_at < ? AND status IN ('completed', 'failed')""",
            (cutoff,),
        )
        deleted = cursor.rowcount or 0
        cursor = db.execute("DELETE FROM idempotency_cache WHERE created_at < ?", (cutoff,))
        deleted += cursor.rowcount or 0
        return deleted

    # --- idempotency cache (used by Phase 3 ``@idempotent``) ----------

    def get_idempotent(self, execution_id: str, function_key: str) -> Any | None:
        row = self._conn().fetchone(
            """SELECT result FROM idempotency_cache
               WHERE execution_id = ? AND function_key = ?""",
            (execution_id, function_key),
        )
        if row is None:
            return None
        return json.loads(row["result"])

    def put_idempotent(self, execution_id: str, function_key: str, result: Any) -> None:
        # Strict serialization: refuse to silently coerce non-JSON values
        # via ``default=str``. The ``@idempotent`` decorator runs
        # ``pydantic_core.to_jsonable_python`` first, so by the time we get
        # here the value is plain JSON-shaped data.
        from fastaiagent._internal.project import safe_get_project_id

        self._conn().execute(
            """INSERT OR REPLACE INTO idempotency_cache
               (execution_id, function_key, result, created_at, project_id)
               VALUES (?, ?, ?, ?, ?)""",
            (
                execution_id,
                function_key,
                json.dumps(result),
                datetime.now(tz=timezone.utc).isoformat(),
                safe_get_project_id(),
            ),
        )

    # --- helpers ------------------------------------------------------

    @staticmethod
    def _row_to_checkpoint(row: dict[str, Any]) -> Checkpoint:
        # Older rows (pre-v2 migration backfill failed for some reason) may
        # have a NULL checkpoint_id — fall back to ``id``.
        checkpoint_id = row.get("checkpoint_id") or row.get("id") or ""
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            parent_checkpoint_id=row.get("parent_checkpoint_id"),
            chain_name=row["chain_name"],
            execution_id=row["execution_id"],
            node_id=row["node_id"],
            node_index=row["node_index"] or 0,
            status=row.get("status") or "completed",
            state_snapshot=json.loads(row["state_snapshot"] or "{}"),
            node_input=json.loads(row["node_input"] or "{}"),
            node_output=json.loads(row["node_output"] or "{}"),
            iteration=row["iteration"] or 0,
            iteration_counters=json.loads(row["iteration_counters"] or "{}"),
            interrupt_reason=row.get("interrupt_reason"),
            interrupt_context=json.loads(row.get("interrupt_context") or "{}"),
            agent_path=row.get("agent_path"),
            created_at=row.get("created_at") or "",
        )
