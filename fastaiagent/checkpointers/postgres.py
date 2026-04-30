"""Postgres-backed checkpointer for production deployments.

Ships under the ``fastaiagent[postgres]`` extra. Same surface as
:class:`SQLiteCheckpointer` so a chain / agent / swarm / supervisor only
needs to swap the constructor argument:

    chain = Chain("flow", checkpointer=PostgresCheckpointer(
        "postgresql://user:pass@host/db"
    ))

Implementation notes:
    - Uses psycopg3 (not psycopg2) for native ``JSONB`` adaptation and
      modern pooling via ``psycopg_pool.ConnectionPool``.
    - Stores all JSON columns as ``JSONB`` and timestamps as ``TIMESTAMPTZ``.
    - The atomic pending-interrupt claim is a single ``DELETE … RETURNING *``
      — Postgres MVCC guarantees only one of N concurrent resumers wins.
    - Schema is namespaced (default ``fastaiagent``); ``setup()`` is gated on
      the ``schema_version`` table so re-runs are no-ops.
"""

from __future__ import annotations

import builtins
import importlib.resources
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from fastaiagent.chain.checkpoint import Checkpoint
from fastaiagent.checkpointers.protocol import PendingInterrupt

_SCHEMA_VERSION = 1


def _require_psycopg() -> Any:
    """Import psycopg lazily so the SDK still loads without the extra."""
    try:
        import psycopg  # noqa: F401
        from psycopg.types.json import Jsonb
        from psycopg_pool import ConnectionPool
    except ImportError as e:  # pragma: no cover - exercised by extra-missing path
        raise ImportError(
            "PostgresCheckpointer requires the [postgres] extra: "
            "`pip install 'fastaiagent[postgres]'`."
        ) from e
    return Jsonb, ConnectionPool


def _quote_ident(name: str) -> str:
    """Validate + quote a schema identifier. Refuses anything but
    ``[A-Za-z_][A-Za-z0-9_]*`` to keep the migration template safe.
    """
    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid Postgres schema name {name!r}")
    return name


class PostgresCheckpointer:
    """Postgres implementation of the :class:`Checkpointer` Protocol."""

    def __init__(
        self,
        connection_string: str,
        *,
        schema: str = "fastaiagent",
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        self.connection_string = connection_string
        self.schema = _quote_ident(schema)
        # Force the import path so a missing extra fails clearly at construction.
        _require_psycopg()
        # Open lazily on first use. Construction cost is small but real
        # (DNS, TCP handshake), and most short-lived Chain.aexecute calls
        # only need one connection.
        # ``ConnectionPool`` is generic-invariant on its connection type;
        # ``Any`` keeps the field assignable from the unparameterized
        # constructor without giving up usefulness — every method below
        # uses ``pool.connection()`` which is fully typed.
        self._pool: Any = None
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._setup_done = False
        # Fully-qualified table names — quoted once, used everywhere.
        s = self.schema
        self._t_checkpoints = f"{s}.checkpoints"
        self._t_pending = f"{s}.pending_interrupts"
        self._t_idempotency = f"{s}.idempotency_cache"
        self._t_schema_version = f"{s}.schema_version"

    # --- lifecycle ----------------------------------------------------

    def _get_pool(self) -> Any:
        from psycopg_pool import ConnectionPool

        if self._pool is None:
            pool = ConnectionPool(
                conninfo=self.connection_string,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
                open=False,  # open eagerly below so a bad DSN fails fast
            )
            pool.open()
            self._pool = pool
        return self._pool

    def setup(self) -> None:
        """Run idempotent migrations against the configured schema."""
        if self._setup_done:
            return
        sql_template = (
            importlib.resources.files("fastaiagent.checkpointers.migrations")
            .joinpath("postgres_v1.sql")
            .read_text()
        )
        sql = sql_template.replace("{schema}", self.schema)
        pool = self._get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
        self._setup_done = True

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None
            self._setup_done = False

    def _ensure_setup(self) -> None:
        if not self._setup_done:
            self.setup()

    # --- writes -------------------------------------------------------

    def put(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint. Fills checkpoint_id / created_at if missing."""
        import uuid as _uuid

        from psycopg.types.json import Jsonb

        if not checkpoint.checkpoint_id:
            checkpoint.checkpoint_id = str(_uuid.uuid4())
        if not checkpoint.created_at:
            checkpoint.created_at = datetime.now(tz=timezone.utc).isoformat()

        self._ensure_setup()
        pool = self._get_pool()
        # ON CONFLICT DO UPDATE — same checkpoint_id rewriting is rare but
        # cleanly recoverable (the chain executor never re-uses an id, but
        # third-party tools might).
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._t_checkpoints} (
                        checkpoint_id, parent_checkpoint_id, chain_name,
                        execution_id, node_id, node_index, status,
                        state_snapshot, node_input, node_output,
                        iteration, iteration_counters,
                        interrupt_reason, interrupt_context, agent_path,
                        created_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (checkpoint_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        state_snapshot = EXCLUDED.state_snapshot,
                        node_output = EXCLUDED.node_output,
                        iteration_counters = EXCLUDED.iteration_counters,
                        interrupt_reason = EXCLUDED.interrupt_reason,
                        interrupt_context = EXCLUDED.interrupt_context
                    """,
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.parent_checkpoint_id,
                        checkpoint.chain_name,
                        checkpoint.execution_id,
                        checkpoint.node_id,
                        checkpoint.node_index,
                        checkpoint.status,
                        Jsonb(checkpoint.state_snapshot),
                        Jsonb(checkpoint.node_input),
                        Jsonb(checkpoint.node_output),
                        checkpoint.iteration,
                        Jsonb(checkpoint.iteration_counters),
                        checkpoint.interrupt_reason,
                        Jsonb(checkpoint.interrupt_context),
                        checkpoint.agent_path,
                        checkpoint.created_at,
                    ),
                )
            conn.commit()

    def put_writes(self, execution_id: str, checkpoint_id: str, writes: list[Any]) -> None:
        """No-op in Phase 1 — Phase 2/3 will wire this up."""
        return None

    # --- reads --------------------------------------------------------

    def get_last(self, execution_id: str) -> Checkpoint | None:
        self._ensure_setup()
        pool = self._get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {self._t_checkpoints}
                WHERE execution_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (execution_id,),
            )
            row = cur.fetchone()
            cols = [d.name for d in cur.description] if cur.description else []
        return self._row_to_checkpoint(_row_dict(cols, row)) if row else None

    def get_by_id(self, execution_id: str, checkpoint_id: str) -> Checkpoint | None:
        self._ensure_setup()
        pool = self._get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {self._t_checkpoints}
                WHERE execution_id = %s AND checkpoint_id = %s
                LIMIT 1
                """,
                (execution_id, checkpoint_id),
            )
            row = cur.fetchone()
            cols = [d.name for d in cur.description] if cur.description else []
        return self._row_to_checkpoint(_row_dict(cols, row)) if row else None

    def list(  # noqa: A003  (matches Checkpointer protocol method name)
        self, execution_id: str, *, limit: int = 100
    ) -> builtins.list[Checkpoint]:
        self._ensure_setup()
        pool = self._get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {self._t_checkpoints}
                WHERE execution_id = %s
                ORDER BY created_at ASC, node_index ASC
                LIMIT %s
                """,
                (execution_id, int(limit)),
            )
            rows = cur.fetchall()
            cols = [d.name for d in cur.description] if cur.description else []
        return [self._row_to_checkpoint(_row_dict(cols, r)) for r in rows]

    def list_pending_interrupts(self, *, limit: int = 100) -> builtins.list[PendingInterrupt]:
        self._ensure_setup()
        pool = self._get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT execution_id, chain_name, node_id, reason, context,
                       agent_path, created_at
                FROM {self._t_pending}
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (int(limit),),
            )
            rows = cur.fetchall()
        return [
            PendingInterrupt(
                execution_id=r[0],
                chain_name=r[1],
                node_id=r[2],
                reason=r[3],
                context=dict(r[4]) if r[4] is not None else {},
                agent_path=r[5],
                created_at=_isoformat(r[6]),
            )
            for r in rows
        ]

    # --- interrupt coordination --------------------------------------

    def record_interrupt(self, checkpoint: Checkpoint, pending: PendingInterrupt) -> None:
        """Insert the interrupted checkpoint and pending row in one txn."""
        import uuid as _uuid

        from psycopg.types.json import Jsonb

        if not checkpoint.checkpoint_id:
            checkpoint.checkpoint_id = str(_uuid.uuid4())
        if not checkpoint.created_at:
            checkpoint.created_at = datetime.now(tz=timezone.utc).isoformat()
        if not pending.created_at:
            pending.created_at = checkpoint.created_at

        self._ensure_setup()
        pool = self._get_pool()
        # psycopg3 connections start an implicit transaction on the first
        # statement and commit on context exit — so both inserts land
        # together. ``INSERT ... ON CONFLICT (execution_id) DO NOTHING``
        # on the pending row preserves the existing pending interrupt if
        # ``record_interrupt`` is called twice for the same execution
        # (e.g. nested topology re-emitting the signal); the first row
        # wins.
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._t_checkpoints} (
                        checkpoint_id, parent_checkpoint_id, chain_name,
                        execution_id, node_id, node_index, status,
                        state_snapshot, node_input, node_output,
                        iteration, iteration_counters,
                        interrupt_reason, interrupt_context, agent_path,
                        created_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        checkpoint.checkpoint_id,
                        checkpoint.parent_checkpoint_id,
                        checkpoint.chain_name,
                        checkpoint.execution_id,
                        checkpoint.node_id,
                        checkpoint.node_index,
                        checkpoint.status,
                        Jsonb(checkpoint.state_snapshot),
                        Jsonb(checkpoint.node_input),
                        Jsonb(checkpoint.node_output),
                        checkpoint.iteration,
                        Jsonb(checkpoint.iteration_counters),
                        checkpoint.interrupt_reason,
                        Jsonb(checkpoint.interrupt_context),
                        checkpoint.agent_path,
                        checkpoint.created_at,
                    ),
                )
                cur.execute(
                    f"""
                    INSERT INTO {self._t_pending} (
                        execution_id, chain_name, node_id, reason,
                        context, agent_path, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (execution_id) DO NOTHING
                    """,
                    (
                        pending.execution_id,
                        pending.chain_name,
                        pending.node_id,
                        pending.reason,
                        Jsonb(pending.context),
                        pending.agent_path,
                        pending.created_at,
                    ),
                )
            conn.commit()

    def delete_pending_interrupt_atomic(self, execution_id: str) -> PendingInterrupt | None:
        """Claim a pending interrupt via ``DELETE … RETURNING *``.

        Postgres MVCC guarantees that only one of N concurrent resumers
        sees the row; everyone else sees ``None`` and the caller raises
        :class:`AlreadyResumed`.
        """
        self._ensure_setup()
        pool = self._get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                DELETE FROM {self._t_pending}
                WHERE execution_id = %s
                RETURNING execution_id, chain_name, node_id, reason,
                          context, agent_path, created_at
                """,
                (execution_id,),
            )
            row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return PendingInterrupt(
            execution_id=row[0],
            chain_name=row[1],
            node_id=row[2],
            reason=row[3],
            context=dict(row[4]) if row[4] is not None else {},
            agent_path=row[5],
            created_at=_isoformat(row[6]),
        )

    # --- deletes / prune ---------------------------------------------

    def delete_execution(self, execution_id: str) -> None:
        self._ensure_setup()
        pool = self._get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._t_checkpoints} WHERE execution_id = %s",
                    (execution_id,),
                )
                cur.execute(
                    f"DELETE FROM {self._t_pending} WHERE execution_id = %s",
                    (execution_id,),
                )
                cur.execute(
                    f"DELETE FROM {self._t_idempotency} WHERE execution_id = %s",
                    (execution_id,),
                )
            conn.commit()

    def get_idempotent(self, execution_id: str, function_key: str) -> Any | None:
        self._ensure_setup()
        pool = self._get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT result FROM {self._t_idempotency}
                WHERE execution_id = %s AND function_key = %s
                """,
                (execution_id, function_key),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return row[0]

    def put_idempotent(self, execution_id: str, function_key: str, result: Any) -> None:
        from psycopg.types.json import Jsonb

        self._ensure_setup()
        pool = self._get_pool()
        # Strict serialization — the @idempotent decorator already runs the
        # value through pydantic_core.to_jsonable_python, so anything that
        # gets here should round-trip cleanly through Jsonb.
        json.dumps(result)  # raises if non-serializable
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._t_idempotency}
                        (execution_id, function_key, result, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (execution_id, function_key) DO UPDATE SET
                        result = EXCLUDED.result,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        execution_id,
                        function_key,
                        Jsonb(result),
                        datetime.now(tz=timezone.utc),
                    ),
                )
            conn.commit()

    def prune(self, older_than: timedelta) -> int:
        self._ensure_setup()
        cutoff = datetime.now(tz=timezone.utc) - older_than
        pool = self._get_pool()
        deleted = 0
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self._t_checkpoints}
                    WHERE created_at < %s
                      AND status IN ('completed', 'failed')
                    """,
                    (cutoff,),
                )
                deleted += cur.rowcount or 0
                cur.execute(
                    f"DELETE FROM {self._t_idempotency} WHERE created_at < %s",
                    (cutoff,),
                )
                deleted += cur.rowcount or 0
            conn.commit()
        return deleted

    # --- helpers ------------------------------------------------------

    @staticmethod
    def _row_to_checkpoint(row: dict[str, Any]) -> Checkpoint:
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            parent_checkpoint_id=row.get("parent_checkpoint_id"),
            chain_name=row["chain_name"],
            execution_id=row["execution_id"],
            node_id=row["node_id"],
            node_index=row["node_index"] or 0,
            status=row.get("status") or "completed",
            state_snapshot=dict(row["state_snapshot"]) if row["state_snapshot"] else {},
            node_input=dict(row["node_input"]) if row["node_input"] else {},
            node_output=dict(row["node_output"]) if row["node_output"] else {},
            iteration=row["iteration"] or 0,
            iteration_counters=(
                dict(row["iteration_counters"]) if row["iteration_counters"] else {}
            ),
            interrupt_reason=row.get("interrupt_reason"),
            interrupt_context=(dict(row["interrupt_context"]) if row["interrupt_context"] else {}),
            agent_path=row.get("agent_path"),
            created_at=_isoformat(row.get("created_at")),
        )


def _row_dict(cols: list[str], row: tuple[Any, ...] | None) -> dict[str, Any]:
    return {c: row[i] for i, c in enumerate(cols)} if row else {}


def _isoformat(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
