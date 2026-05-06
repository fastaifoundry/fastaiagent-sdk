"""Persistent storage for facts learned from past traces.

The ``MemoryStore`` is a thin wrapper over the ``learned_memory`` table
introduced in schema migration v8. Facts are scoped (``user`` /
``project`` / ``agent``) and identified by their text + scope combo;
re-extracting the same fact is idempotent (the UNIQUE constraint
deduplicates).

Conflict resolution is by **recency**: when a new fact contradicts an
existing one, the caller marks the old row's ``superseded_by`` to point
at the new row. We never delete — the audit chain stays intact for
human review and rollback.

This module knows nothing about LLMs. The ``extractor`` module is the
piece that turns trace contents into ``Fact`` objects; this module just
stores and retrieves them.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import init_local_db

Scope = Literal["user", "project", "agent"]


@dataclass(slots=True)
class Fact:
    """A single durable fact extracted from a trace.

    ``id`` is assigned by the database on insert; ``None`` for not-yet-stored
    facts. ``superseded_by`` points at the row that replaced this one (None
    if the fact is current).
    """

    scope: Scope
    scope_id: str
    fact: str
    source_trace_id: str | None = None
    confidence: float = 1.0
    created_at: float | None = None
    superseded_by: int | None = None
    project_id: str = ""
    id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryStore:
    """CRUD over the ``learned_memory`` table.

    Construct with no args to point at the default ``local.db`` (resolves
    via :func:`fastaiagent._internal.config.get_config`); pass ``db_path``
    to point elsewhere (used in tests).
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or get_config().local_db_path
        # Ensure migrations have run so the table exists. Idempotent.
        init_local_db(self._db_path).close()

    def _open(self) -> SQLiteHelper:
        return SQLiteHelper(self._db_path)

    # ─── Insert ──────────────────────────────────────────────────────────────

    def add(self, fact: Fact) -> int:
        """Insert a fact. Returns the row id.

        If the same (scope, scope_id, fact, project_id) tuple already exists,
        we return the existing id without modifying anything (idempotent).
        Use :meth:`supersede` to replace a fact with a newer version.
        """
        if not fact.fact.strip():
            raise ValueError("fact text must be non-empty")
        if fact.scope not in ("user", "project", "agent"):
            raise ValueError(f"scope must be one of user|project|agent, got {fact.scope!r}")

        created_at = fact.created_at if fact.created_at is not None else time.time()
        db = self._open()
        try:
            existing = db.fetchone(
                "SELECT id FROM learned_memory "
                "WHERE scope = ? AND scope_id = ? AND fact = ? AND project_id = ?",
                (fact.scope, fact.scope_id, fact.fact, fact.project_id),
            )
            if existing:
                return int(existing["id"])
            cursor = db.execute(
                "INSERT INTO learned_memory "
                "(scope, scope_id, fact, source_trace_id, confidence, created_at, project_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fact.scope,
                    fact.scope_id,
                    fact.fact,
                    fact.source_trace_id,
                    fact.confidence,
                    created_at,
                    fact.project_id,
                ),
            )
            new_id = cursor.lastrowid
            return int(new_id) if new_id is not None else 0
        finally:
            db.close()

    def add_many(self, facts: Iterable[Fact]) -> list[int]:
        """Bulk insert. Returns the list of resulting row ids in input order."""
        return [self.add(f) for f in facts]

    # ─── Read ────────────────────────────────────────────────────────────────

    def list_active(
        self,
        scope: Scope,
        scope_id: str = "",
        project_id: str = "",
        limit: int | None = None,
    ) -> list[Fact]:
        """Return non-superseded facts for the given scope.

        ``scope_id=""`` matches all scope_ids within the scope (useful when
        you want every agent-scoped fact, not just one agent's).
        """
        sql = (
            "SELECT * FROM learned_memory "
            "WHERE scope = ? AND project_id = ? AND superseded_by IS NULL"
        )
        params: list = [scope, project_id]
        if scope_id:
            sql += " AND scope_id = ?"
            params.append(scope_id)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        db = self._open()
        try:
            rows = db.fetchall(sql, tuple(params))
            return [self._row_to_fact(r) for r in rows]
        finally:
            db.close()

    def list_all(self, project_id: str = "", limit: int | None = None) -> list[Fact]:
        """Every fact across every scope (for the UI listing endpoint)."""
        sql = "SELECT * FROM learned_memory WHERE project_id = ? ORDER BY created_at DESC"
        params: tuple = (project_id,)
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        db = self._open()
        try:
            return [self._row_to_fact(r) for r in db.fetchall(sql, params)]
        finally:
            db.close()

    def get(self, fact_id: int) -> Fact | None:
        db = self._open()
        try:
            row = db.fetchone("SELECT * FROM learned_memory WHERE id = ?", (fact_id,))
            return self._row_to_fact(row) if row else None
        finally:
            db.close()

    # ─── Conflict resolution ────────────────────────────────────────────────

    def supersede(self, old_id: int, new_id: int) -> None:
        """Mark ``old_id`` as superseded by ``new_id``.

        Idempotent: if the row is already marked, the UPDATE is a no-op.
        Both rows must exist or the call raises ValueError.
        """
        db = self._open()
        try:
            old = db.fetchone("SELECT id FROM learned_memory WHERE id = ?", (old_id,))
            new = db.fetchone("SELECT id FROM learned_memory WHERE id = ?", (new_id,))
            if not old or not new:
                raise ValueError(
                    f"supersede: missing row(s) old_id={old_id} new_id={new_id}"
                )
            db.execute(
                "UPDATE learned_memory SET superseded_by = ? WHERE id = ?",
                (new_id, old_id),
            )
        finally:
            db.close()

    # ─── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_fact(row: dict) -> Fact:
        return Fact(
            id=int(row["id"]),
            scope=row["scope"],
            scope_id=row["scope_id"],
            fact=row["fact"],
            source_trace_id=row["source_trace_id"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else 1.0,
            created_at=float(row["created_at"]),
            superseded_by=int(row["superseded_by"]) if row["superseded_by"] is not None else None,
            project_id=row["project_id"],
        )
