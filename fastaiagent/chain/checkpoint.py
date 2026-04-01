"""Local SQLite checkpointing for chain executions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    chain_name TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_index INTEGER,
    status TEXT DEFAULT 'completed',
    state_snapshot TEXT DEFAULT '{}',
    node_input TEXT DEFAULT '{}',
    node_output TEXT DEFAULT '{}',
    iteration INTEGER DEFAULT 0,
    iteration_counters TEXT DEFAULT '{}',
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_cp_exec ON checkpoints (execution_id);
"""


class Checkpoint(BaseModel):
    """A checkpoint snapshot of chain execution at a node."""

    id: str = ""
    chain_name: str = ""
    execution_id: str = ""
    node_id: str = ""
    node_index: int = 0
    status: str = "completed"
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    node_input: dict[str, Any] = Field(default_factory=dict)
    node_output: dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    iteration_counters: dict[str, int] = Field(default_factory=dict)
    created_at: str = ""


class CheckpointStore:
    """SQLite-backed checkpoint storage for chain execution resume."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_config().checkpoint_db_path
        self._db: SQLiteHelper | None = None

    def _get_db(self) -> SQLiteHelper:
        if self._db is None:
            self._db = SQLiteHelper(self.db_path)
            for stmt in _SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    self._db.execute(stmt)
        return self._db

    def save(
        self,
        chain_name: str,
        execution_id: str,
        node_id: str,
        node_index: int,
        state_snapshot: dict,
        node_input: dict | None = None,
        node_output: dict | None = None,
        iteration: int = 0,
        iteration_counters: dict | None = None,
    ) -> Checkpoint:
        """Save a checkpoint after a node completes."""
        cp = Checkpoint(
            id=str(uuid.uuid4()),
            chain_name=chain_name,
            execution_id=execution_id,
            node_id=node_id,
            node_index=node_index,
            state_snapshot=state_snapshot,
            node_input=node_input or {},
            node_output=node_output or {},
            iteration=iteration,
            iteration_counters=iteration_counters or {},
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        db = self._get_db()
        db.execute(
            """INSERT INTO checkpoints
               (id, chain_name, execution_id, node_id, node_index, status,
                state_snapshot, node_input, node_output, iteration,
                iteration_counters, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cp.id, cp.chain_name, cp.execution_id, cp.node_id,
                cp.node_index, cp.status,
                json.dumps(cp.state_snapshot), json.dumps(cp.node_input),
                json.dumps(cp.node_output), cp.iteration,
                json.dumps(cp.iteration_counters), cp.created_at,
            ),
        )
        return cp

    def load(self, execution_id: str) -> list[Checkpoint]:
        """Load all checkpoints for an execution."""
        db = self._get_db()
        rows = db.fetchall(
            "SELECT * FROM checkpoints WHERE execution_id = ? ORDER BY node_index",
            (execution_id,),
        )
        return [self._row_to_checkpoint(r) for r in rows]

    def get_latest(self, execution_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for an execution."""
        db = self._get_db()
        row = db.fetchone(
            "SELECT * FROM checkpoints WHERE execution_id = ? ORDER BY node_index DESC LIMIT 1",
            (execution_id,),
        )
        return self._row_to_checkpoint(row) if row else None

    def _row_to_checkpoint(self, row: dict) -> Checkpoint:
        return Checkpoint(
            id=row["id"],
            chain_name=row["chain_name"],
            execution_id=row["execution_id"],
            node_id=row["node_id"],
            node_index=row["node_index"],
            status=row["status"],
            state_snapshot=json.loads(row["state_snapshot"]),
            node_input=json.loads(row["node_input"]),
            node_output=json.loads(row["node_output"]),
            iteration=row["iteration"],
            iteration_counters=json.loads(row["iteration_counters"]),
            created_at=row["created_at"],
        )

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
