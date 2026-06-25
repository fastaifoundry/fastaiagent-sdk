"""Non-blocking connected-HITL event export to the FastAIAgent Platform.

When the SDK is connected to an Enterprise control plane, a paused run
(``interrupt()``) and its later resolution (``resume``) are reported to the
plane as **metadata-only** events so it can serve an org-wide pending/paused
status view + a compliance ledger. The plane is an **observer only** — it is
never the approval surface; the customer's own app still resolves the pause.

This mirrors the proven trace outbox (:mod:`fastaiagent.trace.platform_export`):

* The emit helpers (:func:`record_pause_event` / :func:`record_resolution_event`)
  write one row to the local ``hitl_events`` table with ``synced=0`` — the
  durable, local-first source of truth — then kick a fire-and-forget background
  drain. They are **best-effort**: a no-op when not connected, and they never
  raise into the agent hot path.
* :meth:`HitlEventExporter.export` drains ``synced=0`` rows, POSTs them to
  ``/public/v1/hitl/events`` with bounded retry, and marks them ``synced=1``
  **only after a confirmed 2xx**. Un-acked rows stay buffered and re-drain.

The server is **idempotent by ``event_id``** (the SDK-generated row PK), so
at-least-once re-send returns ``{"ingested": 0}`` for duplicates and never
double-counts. Unlike traces, the body is domain-scoped — ``{"events": [...]}``
carries no ``project`` field (the plane scopes from the API key).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from pydantic import BaseModel

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper

if TYPE_CHECKING:
    from fastaiagent.client import _Connection

logger = logging.getLogger(__name__)


class HitlEvent(BaseModel):
    """One connected-HITL event pushed to the plane (metadata only).

    Field names mirror the plane's ``/public/v1/hitl/events`` ingest schema
    exactly, so :meth:`to_wire` serializes straight to the request body.
    """

    event_id: str
    run_id: str
    event_type: str  # "paused" | "resolved"
    kind: str | None = None  # "approval" | "interrupt"
    agent_id: str | None = None
    chain_id: str | None = None
    node: str | None = None
    reason_code: str | None = None
    reason: str | None = None
    status: str | None = None  # "approved" | "rejected" (on "resolved")
    resolver: str | None = None
    occurred_at: str | None = None  # ISO 8601
    # Reserved for future structured metadata; HITL pushes metadata only, so the
    # SDK never populates this (no raw interrupt payloads / PII on the wire).
    context: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


_HITL_SCHEMA = """
CREATE TABLE IF NOT EXISTS hitl_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    kind TEXT,
    agent_id TEXT,
    chain_id TEXT,
    node TEXT,
    reason_code TEXT,
    reason TEXT,
    status TEXT,
    resolver TEXT,
    occurred_at TEXT,
    context TEXT,
    project_id TEXT NOT NULL DEFAULT '',
    synced INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hitl_events_synced ON hitl_events(synced, created_at);
"""


class HitlEventStore:
    """Local SQLite outbox for connected-HITL events.

    Mirrors :class:`fastaiagent.trace.storage.TraceStore`'s platform-export
    buffer (``fetch_unsynced`` / ``mark_synced`` / ``enforce_buffer_bound``) but
    keyed on the ``hitl_events`` table and ``event_id``.
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_config().resolved_trace_db_path
        # Run the full migration ladder so ``hitl_events`` (v12) is in place even
        # when this store is the first thing touched. Falls back to the inline
        # schema only in the unusual install where the UI module is missing.
        try:
            from fastaiagent.ui.db import init_local_db

            self._db = init_local_db(self.db_path)
        except (ImportError, RuntimeError):
            self._db = SQLiteHelper(self.db_path)
            self._init_schema()

    def _init_schema(self) -> None:
        """Legacy fallback — only runs when ``init_local_db`` is unavailable."""
        for stmt in _HITL_SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)

    @staticmethod
    def _row_to_event(row: dict[str, Any]) -> HitlEvent:
        """Deserialize a ``hitl_events`` row into a :class:`HitlEvent`.

        NULLs are coerced to field defaults so one partial buffered row can never
        fail validation and abort the whole bg-thread drain.
        """
        return HitlEvent(
            event_id=row["event_id"],
            run_id=row["run_id"],
            event_type=row["event_type"],
            kind=row["kind"],
            agent_id=row["agent_id"],
            chain_id=row["chain_id"],
            node=row["node"],
            reason_code=row["reason_code"],
            reason=row["reason"],
            status=row["status"],
            resolver=row["resolver"],
            occurred_at=row["occurred_at"],
            context=json.loads(row["context"]) if row["context"] else None,
        )

    def record(self, event: HitlEvent, project_id: str | None = None) -> None:
        """Persist one event with ``synced=0``. Idempotent on ``event_id`` (PK)."""
        self._db.execute(
            """INSERT OR IGNORE INTO hitl_events
               (event_id, run_id, event_type, kind, agent_id, chain_id, node,
                reason_code, reason, status, resolver, occurred_at, context,
                project_id, synced, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (
                event.event_id,
                event.run_id,
                event.event_type,
                event.kind,
                event.agent_id,
                event.chain_id,
                event.node,
                event.reason_code,
                event.reason,
                event.status,
                event.resolver,
                event.occurred_at,
                json.dumps(event.context) if event.context is not None else None,
                project_id or "",
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def fetch_unsynced(self, limit: int, project_id: str | None = None) -> list[HitlEvent]:
        """Return up to ``limit`` un-acked events (``synced=0``), oldest first."""
        if project_id is not None:
            rows = self._db.fetchall(
                "SELECT * FROM hitl_events WHERE synced = 0 AND project_id = ? "
                "ORDER BY created_at LIMIT ?",
                (project_id, limit),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM hitl_events WHERE synced = 0 ORDER BY created_at LIMIT ?",
                (limit,),
            )
        return [self._row_to_event(r) for r in rows]

    def mark_synced(self, event_ids: list[str]) -> None:
        """Mark events as no longer re-send candidates (acked *or* abandoned)."""
        for start in range(0, len(event_ids), 500):
            chunk = event_ids[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            self._db.execute(
                f"UPDATE hitl_events SET synced = 1 WHERE event_id IN ({placeholders})",
                tuple(chunk),
            )

    def count_unsynced(self, project_id: str | None = None) -> int:
        """Count un-acked (``synced=0``) events, optionally scoped to a project."""
        if project_id is not None:
            row = self._db.fetchone(
                "SELECT COUNT(*) AS n FROM hitl_events WHERE synced = 0 AND project_id = ?",
                (project_id,),
            )
        else:
            row = self._db.fetchone("SELECT COUNT(*) AS n FROM hitl_events WHERE synced = 0")
        return int(row["n"]) if row else 0

    def enforce_buffer_bound(
        self, max_unsynced: int, max_age_days: int, project_id: str | None = None
    ) -> int:
        """Bound the re-send queue *without deleting local history*.

        Abandons (marks ``synced=1`` — rows stay in ``local.db``) un-acked events
        older than ``max_age_days`` or beyond the ``max_unsynced`` cap (oldest
        first). For HITL the age bound is the real cap (events are rare +
        audit-significant); the count branch is a hard safety ceiling that a
        permanently-unentitled domain (terminal 403) won't grow past.
        """
        if project_id is None:
            from fastaiagent._internal.project import safe_get_project_id

            project_id = safe_get_project_id()
        abandoned = 0

        # (a) Age bound — abandon events older than the cutoff.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        old = self._db.fetchall(
            "SELECT event_id FROM hitl_events "
            "WHERE synced = 0 AND project_id = ? AND created_at < ?",
            (project_id, cutoff),
        )
        if old:
            self.mark_synced([r["event_id"] for r in old])
            abandoned += len(old)

        # (b) Count bound — abandon the oldest excess beyond the cap.
        remaining = self.count_unsynced(project_id)
        if remaining > max_unsynced:
            excess = remaining - max_unsynced
            rows = self._db.fetchall(
                "SELECT event_id FROM hitl_events WHERE synced = 0 AND project_id = ? "
                "ORDER BY created_at ASC LIMIT ?",
                (project_id, excess),
            )
            if rows:
                self.mark_synced([r["event_id"] for r in rows])
                abandoned += len(rows)

        return abandoned

    def close(self) -> None:
        self._db.close()


class HitlEventExporter(SpanExporter):
    """Drains the local HITL outbox to the plane with retry + idempotent re-send.

    Modeled on :class:`fastaiagent.trace.platform_export.PlatformSpanExporter`.
    Registered as a secondary ``BatchSpanProcessor`` (so its ``export`` rides the
    trace flush cadence + ``disconnect()`` force-flush), and also invoked directly
    on a daemon thread by the emit helpers — see :func:`_kick_drain`. The
    ``spans`` argument is ignored: it is only a flush trigger; the payload is
    drained from the store.
    """

    # Transient-only retry (connection / timeout / 5xx); 4xx (incl. 403) terminal.
    _MAX_ATTEMPTS = 3
    _BACKOFF_BASE = 0.5
    _TIMEOUT = 10
    _DRAIN_LIMIT = 500

    # Gentler than traces: HITL events are rare + audit-significant, so the age
    # bound is generous (30d) and the count ceiling is effectively never hit.
    _MAX_UNSYNCED = 100_000
    _MAX_AGE_DAYS = 30

    def __init__(self) -> None:
        # Lazily opened on first export(); cached thereafter (drain-side store).
        self._store: HitlEventStore | None = None

    def _get_store(self) -> HitlEventStore:
        if self._store is None:
            self._store = HitlEventStore()
        return self._store

    def export(self, spans: Sequence[ReadableSpan] = ()) -> SpanExportResult:
        """Drain buffered HITL events from local SQLite and push them."""
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return SpanExportResult.SUCCESS

        try:
            from fastaiagent._internal.project import safe_get_project_id

            store = self._get_store()
            pid = safe_get_project_id()

            pending = store.fetch_unsynced(limit=self._DRAIN_LIMIT, project_id=pid)
            if pending:
                wire = [e.to_wire() for e in pending]
                if self._post_with_retry(_connection, wire):
                    store.mark_synced([e.event_id for e in pending])

            dropped = store.enforce_buffer_bound(
                self._MAX_UNSYNCED, self._MAX_AGE_DAYS, project_id=pid
            )
            if dropped:
                logger.warning(
                    "HITL platform buffer bound hit: abandoned %d un-acked events "
                    "(kept in local.db, not re-sent)",
                    dropped,
                )
        except Exception:
            # Never let export crash the bg thread; local SQLite still has the data.
            logger.debug("HITL event export drain failed", exc_info=True)

        # Always SUCCESS — the durable buffer owns retry, not the OTel processor.
        return SpanExportResult.SUCCESS

    def _post_with_retry(self, conn: _Connection, wire: list[dict[str, Any]]) -> bool:
        """POST ``wire`` to ``/public/v1/hitl/events``. Return True on 2xx.

        Retries connection errors, timeouts, and HTTP 5xx with exponential
        backoff. Does **not** retry 4xx — a 403 (domain not entitled to
        ``connected_state_plane``) or bad request won't fix on retry; events stay
        buffered and the buffer bound eventually ages them out.
        """
        import httpx

        url = f"{conn.target}/public/v1/hitl/events"
        payload = {"events": wire}

        for attempt in range(self._MAX_ATTEMPTS):
            try:
                with httpx.Client(timeout=self._TIMEOUT, verify=True) as client:
                    resp = client.post(url, json=payload, headers=conn.headers)
                code = resp.status_code
                if 200 <= code < 300:
                    return True
                if 400 <= code < 500:
                    logger.warning(
                        "Platform rejected %d HITL events with HTTP %d — not "
                        "retrying; left buffered for the bound to age out "
                        "(403 = domain not entitled to connected_state_plane).",
                        len(wire),
                        code,
                    )
                    return False
                # 5xx → transient; fall through to backoff + retry.
                logger.debug(
                    "HITL export HTTP %d (attempt %d/%d)",
                    code,
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                )
            except httpx.TransportError:
                logger.debug(
                    "HITL export transient error (attempt %d/%d)",
                    attempt + 1,
                    self._MAX_ATTEMPTS,
                    exc_info=True,
                )

            if attempt < self._MAX_ATTEMPTS - 1:
                time.sleep(self._BACKOFF_BASE * (2**attempt))

        return False

    def shutdown(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                logger.debug("Failed to close exporter HITL store", exc_info=True)
            self._store = None


# --- module singleton + emit helpers --------------------------------------

_exporter: HitlEventExporter | None = None
_exporter_lock = threading.Lock()


def get_hitl_exporter() -> HitlEventExporter:
    """Return the process-wide HITL exporter (shared by the registered
    ``BatchSpanProcessor`` and the per-emit daemon drain)."""
    global _exporter
    if _exporter is None:
        with _exporter_lock:
            if _exporter is None:
                _exporter = HitlEventExporter()
    return _exporter


def _kick_drain() -> None:
    """Fire-and-forget background drain. Runs on a throwaway daemon thread so the
    HTTP POST + retry backoff never block the agent hot path."""
    try:
        get_hitl_exporter().export([])
    except Exception:
        logger.debug("HITL drain kick failed", exc_info=True)


def _emit(
    *,
    event_type: str,
    run_id: str,
    node: str | None,
    reason: str | None,
    reason_code: str | None,
    agent_id: str | None,
    chain_id: str | None,
    kind: str | None,
    status: str | None,
    resolver: str | None,
    occurred_at: str | None,
) -> None:
    """Best-effort emit: persist one event locally, then kick a background drain.

    A strict no-op when not connected, and wrapped so it can NEVER raise into the
    pause/resume hot path.
    """
    try:
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return

        from fastaiagent._internal.project import safe_get_project_id

        event = HitlEvent(
            event_id=str(uuid.uuid4()),
            run_id=run_id,
            event_type=event_type,
            kind=kind,
            agent_id=agent_id,
            chain_id=chain_id,
            node=node,
            reason_code=reason_code,
            reason=reason,
            status=status,
            resolver=resolver,
            occurred_at=occurred_at or datetime.now(timezone.utc).isoformat(),
            context=None,  # metadata only — no raw interrupt payloads on the wire
        )
        pid = safe_get_project_id()
        # Short-lived store for the insert (own per-thread connection); the drain
        # uses the exporter singleton's cached store — keeps close() ownership clean.
        store = HitlEventStore()
        try:
            store.record(event, project_id=pid)
        finally:
            store.close()

        # Drain off-thread so neither the SQLite write nor the POST blocks the
        # caller. The write is committed before the thread starts (WAL makes it
        # visible to the drain connection).
        threading.Thread(target=_kick_drain, daemon=True).start()
    except Exception:
        logger.debug("HITL %s emit failed", event_type, exc_info=True)


def record_pause_event(
    *,
    run_id: str,
    node: str | None = None,
    reason: str | None = None,
    reason_code: str | None = None,
    agent_id: str | None = None,
    chain_id: str | None = None,
    kind: str = "interrupt",
) -> None:
    """Report a run pausing for human input to the plane (best-effort, non-blocking)."""
    _emit(
        event_type="paused",
        run_id=run_id,
        node=node,
        reason=reason,
        reason_code=reason_code,
        agent_id=agent_id,
        chain_id=chain_id,
        kind=kind,
        status=None,
        resolver=None,
        occurred_at=None,
    )


def record_resolution_event(
    *,
    run_id: str,
    approved: bool,
    node: str | None = None,
    reason: str | None = None,
    reason_code: str | None = None,
    resolver: str | None = None,
    agent_id: str | None = None,
    chain_id: str | None = None,
    kind: str = "interrupt",
) -> None:
    """Report a pause resolution (approved/rejected) to the plane (best-effort)."""
    _emit(
        event_type="resolved",
        run_id=run_id,
        node=node,
        reason=reason,
        reason_code=reason_code,
        agent_id=agent_id,
        chain_id=chain_id,
        kind=kind,
        status="approved" if approved else "rejected",
        resolver=resolver,
        occurred_at=None,
    )
