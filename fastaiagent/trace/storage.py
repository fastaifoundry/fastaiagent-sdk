"""Local SQLite trace storage with OTel SpanProcessor."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper

# Opt-in foreign-span normalization. Flipped on by
# ``fastaiagent.enable_otel_capture()``; default ``False`` keeps the write path
# byte-identical. When on, ``on_end`` maps foreign OTel/OpenInference/
# OpenLLMetry attribute conventions onto canonical ``gen_ai.*`` keys before the
# attributes are serialized to SQLite. See :mod:`fastaiagent.trace.normalize`.
_normalize_enabled = False
_framework_override: str | None = None


def set_normalize_enabled(value: bool, *, framework: str | None = None) -> None:
    """Toggle write-time foreign-span normalization (see module docstring).

    ``framework`` is an optional override for the root-span framework badge;
    when omitted it is derived per-span from the instrumentation scope name.
    """
    global _normalize_enabled, _framework_override
    _normalize_enabled = value
    _framework_override = framework if value else None


class SpanData(BaseModel):
    """Stored span data."""

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = "OK"
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)


class TraceData(BaseModel):
    """Complete trace with all spans."""

    trace_id: str
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = "OK"
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[SpanData] = Field(default_factory=list)

    def publish(self) -> None:
        """Publish this trace to the platform (for manual backfill)."""
        from fastaiagent._internal.errors import PlatformNotConnectedError
        from fastaiagent._platform.api import get_platform_api
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            raise PlatformNotConnectedError("Not connected to platform. Call fa.connect() first.")
        api = get_platform_api()
        api.post(
            "/public/v1/traces/ingest",
            {
                "project": _connection.project,
                "spans": [s.model_dump() for s in self.spans],
            },
        )


class TraceSummary(BaseModel):
    """Summary of a trace for listing."""

    trace_id: str
    name: str = ""
    start_time: str = ""
    status: str = "OK"
    span_count: int = 0
    duration_ms: int = 0


_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT,
    start_time TEXT,
    end_time TEXT,
    status TEXT DEFAULT 'OK',
    attributes TEXT DEFAULT '{}',
    events TEXT DEFAULT '[]',
    synced INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans (trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans (start_time);
CREATE INDEX IF NOT EXISTS idx_spans_synced ON spans (synced, start_time);
CREATE TABLE IF NOT EXISTS trace_attachments (
    attachment_id  TEXT PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    span_id        TEXT NOT NULL,
    media_type     TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    thumbnail      BLOB,
    full_data      BLOB,
    metadata_json  TEXT DEFAULT '{}',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_attachments_span ON trace_attachments(trace_id, span_id);
"""


class LocalStorageProcessor:
    """OTel SpanProcessor that writes spans to local SQLite."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_config().resolved_trace_db_path
        self._db: SQLiteHelper | None = None

    def _on_ending(self, span: Any) -> None:
        """Called when a span is ending (before on_end)."""
        pass

    def _get_db(self) -> SQLiteHelper:
        if self._db is None:
            # Run the full migration ladder (incl. v2/v3/v4) so save_span
            # sees a current schema, even when the user never ran
            # ``fastaiagent ui`` first. Falls back to the inline _SCHEMA
            # block on import-cycle / setup errors.
            try:
                from fastaiagent.ui.db import init_local_db

                self._db = init_local_db(self.db_path)
            except (ImportError, RuntimeError):
                self._db = SQLiteHelper(self.db_path)
                for stmt in _SCHEMA.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        self._db.execute(stmt)
        return self._db

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: Any) -> None:
        """Called when a span completes — write to SQLite.

        If an opt-in :class:`RedactionPolicy` is installed (via
        ``set_redaction_policy``) with ``mode in {"capture", "both"}``,
        sensitive attribute values are masked here — *before* the JSON
        blob hits SQLite *and* before any downstream OTel exporter
        attached via :func:`add_exporter` sees the span.
        """
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")

        parent_id = None
        if span.parent and span.parent.span_id:
            parent_id = format(span.parent.span_id, "016x")

        # Convert attributes
        attrs = {}
        if hasattr(span, "attributes") and span.attributes:
            attrs = dict(span.attributes)

        # Opt-in: map foreign OTel/OpenInference/OpenLLMetry conventions onto
        # the canonical gen_ai.* keys the UI/FTS read. No-op unless
        # ``enable_otel_capture()`` flipped the flag. Runs *before* redaction so
        # the added canonical keys are themselves redaction-eligible. Only fills
        # absent keys, so native fastaiagent spans are unaffected.
        # Effective flags: a job_scope() override (ContextVar) wins over the
        # module globals so concurrent runner jobs don't clobber each other.
        from fastaiagent._internal.scope import UNSET, scoped_framework, scoped_normalize

        _sn = scoped_normalize.get()
        normalize_on = _normalize_enabled if _sn is UNSET else _sn
        if normalize_on:
            from fastaiagent.trace.normalize import normalize_attributes

            _sf = scoped_framework.get()
            framework_override = _framework_override if _sf is UNSET else _sf
            scope_name = getattr(getattr(span, "instrumentation_scope", None), "name", None)
            attrs = normalize_attributes(
                attrs,
                scope_name=scope_name,
                is_root=(parent_id is None),
                framework_override=framework_override,
            )

        # Apply capture-mode redaction (no-op when no policy is installed).
        from fastaiagent.trace.redaction import _capture_redact

        attrs = _capture_redact(attrs)

        events = []
        if hasattr(span, "events") and span.events:
            for e in span.events:
                # Preserve event attributes — OTel's record_exception stores
                # exception.type / exception.message / exception.stacktrace on
                # the event, which lets the UI render a useful traceback panel
                # instead of just "exception @ <ts>".
                raw_attrs = getattr(e, "attributes", None)
                event_attrs = {str(k): v for k, v in raw_attrs.items()} if raw_attrs else {}
                events.append(
                    {
                        "name": e.name,
                        "timestamp": str(e.timestamp),
                        "attributes": event_attrs,
                    }
                )

        start_ns = span.start_time or 0
        end_ns = span.end_time or 0
        start_time = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()
        end_time = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat()

        status = "OK"
        if hasattr(span, "status") and span.status:
            status = (
                span.status.status_code.name
                if hasattr(span.status.status_code, "name")
                else str(span.status.status_code)
            )

        from fastaiagent._internal.project import safe_get_project_id

        db = self._get_db()
        db.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_span_id, name,
                start_time, end_time, status, attributes, events,
                project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                span_id,
                trace_id,
                parent_id,
                span.name,
                start_time,
                end_time,
                status,
                json.dumps(attrs, default=str),
                json.dumps(events, default=str),
                safe_get_project_id(),
            ),
        )

    def shutdown(self) -> None:
        if self._db:
            self._db.close()
            self._db = None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class TraceStore:
    """Query interface for locally stored traces."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_config().resolved_trace_db_path
        # Run the full migration ladder (incl. v2/v3/v4) so spans /
        # checkpoints / trace_attachments / project_id columns are in
        # place even when TraceStore is the first thing the user calls.
        # Falls back to the inline ``_SCHEMA`` block if the UI module
        # isn't importable (which only happens in unusual install
        # configurations — every supported install has it).
        try:
            from fastaiagent.ui.db import init_local_db

            self._db = init_local_db(self.db_path)
        except (ImportError, RuntimeError):
            self._db = SQLiteHelper(self.db_path)
            self._init_schema()

    def _init_schema(self) -> None:
        """Legacy fallback — only runs when ``init_local_db`` failed."""
        for stmt in _SCHEMA.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._db.execute(stmt)

    @classmethod
    def default(cls) -> TraceStore:
        return cls()

    @staticmethod
    def _row_to_span(row: dict[str, Any]) -> SpanData:
        """Deserialize a ``spans`` row (dict) into a :class:`SpanData`.

        Shared by :meth:`get_trace` and :meth:`fetch_unsynced` so the buffered
        re-send path reconstructs spans identically to the read path.
        """
        # Coerce NULLs to the field defaults. ``on_end`` always writes these, but
        # the drain path processes arbitrary buffered rows in the bg thread — one
        # partial row must not fail validation and abort the whole export.
        return SpanData(
            span_id=row["span_id"],
            trace_id=row["trace_id"],
            parent_span_id=row["parent_span_id"],
            name=row["name"] or "",
            start_time=row["start_time"] or "",
            end_time=row["end_time"] or "",
            status=row["status"] or "OK",
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            events=json.loads(row["events"]) if row["events"] else [],
        )

    def get_trace(self, trace_id: str) -> TraceData:
        """Get a complete trace with all its spans."""
        rows = self._db.fetchall(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        )
        if not rows:
            from fastaiagent._internal.errors import TraceError

            raise TraceError(
                f"Trace '{trace_id}' not found in local storage.\n"
                f"Use TraceStore.list_traces() to see available traces, or check that "
                f"tracing was enabled when the agent ran (trace=True)."
            )

        spans = [self._row_to_span(row) for row in rows]

        root = spans[0] if spans else SpanData(span_id="", trace_id=trace_id)
        return TraceData(
            trace_id=trace_id,
            name=root.name,
            start_time=root.start_time,
            end_time=spans[-1].end_time if spans else "",
            status=root.status,
            spans=spans,
        )

    def list_traces(self, last_hours: int = 24, **filters: Any) -> list[TraceSummary]:
        """List recent traces."""
        rows = self._db.fetchall(
            """SELECT trace_id,
                      MIN(name) as name,
                      MIN(start_time) as start_time,
                      MIN(status) as status,
                      COUNT(*) as span_count
               FROM spans
               GROUP BY trace_id
               ORDER BY start_time DESC
               LIMIT 100""",
        )
        return [
            TraceSummary(
                trace_id=row["trace_id"],
                name=row["name"],
                start_time=row["start_time"],
                status=row["status"],
                span_count=row["span_count"],
            )
            for row in rows
        ]

    def search(self, query: str | None = None, **filters: Any) -> list[TraceSummary]:
        """Search traces by name or attributes."""
        if query:
            rows = self._db.fetchall(
                """SELECT trace_id,
                          MIN(name) as name,
                          MIN(start_time) as start_time,
                          MIN(status) as status,
                          COUNT(*) as span_count
                   FROM spans
                   WHERE name LIKE ? OR attributes LIKE ?
                   GROUP BY trace_id
                   ORDER BY start_time DESC
                   LIMIT 100""",
                (f"%{query}%", f"%{query}%"),
            )
        else:
            return self.list_traces()
        return [
            TraceSummary(
                trace_id=row["trace_id"],
                name=row["name"],
                start_time=row["start_time"],
                status=row["status"],
                span_count=row["span_count"],
            )
            for row in rows
        ]

    def export(self, trace_id: str, format: str = "json") -> str:
        """Export a trace as JSON."""
        trace = self.get_trace(trace_id)
        return trace.model_dump_json(indent=2)

    # --- Platform-export buffer ------------------------------------------
    # ``LocalStorageProcessor`` writes every span with ``synced=0``;
    # ``PlatformSpanExporter`` (bg thread) drains synced=0 rows, pushes them,
    # then marks ``synced=1`` only after a confirmed 2xx. See platform_export.py.

    def fetch_unsynced(self, limit: int, project_id: str | None = None) -> list[SpanData]:
        """Return up to ``limit`` un-acked spans (``synced=0``), oldest first.

        Scoped to ``project_id`` when given — rows are stamped with the *local*
        project id (``safe_get_project_id``), not the platform UUID. Oldest-first
        so a backlog drains in production order.
        """
        if project_id is not None:
            rows = self._db.fetchall(
                "SELECT * FROM spans WHERE synced = 0 AND project_id = ? "
                "ORDER BY start_time LIMIT ?",
                (project_id, limit),
            )
        else:
            rows = self._db.fetchall(
                "SELECT * FROM spans WHERE synced = 0 ORDER BY start_time LIMIT ?",
                (limit,),
            )
        return [self._row_to_span(r) for r in rows]

    def mark_synced(self, span_ids: list[str]) -> None:
        """Mark spans as no longer re-send candidates (acked *or* abandoned).

        Chunked to stay under SQLite's bound-variable limit (999).
        """
        for start in range(0, len(span_ids), 500):
            chunk = span_ids[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            self._db.execute(
                f"UPDATE spans SET synced = 1 WHERE span_id IN ({placeholders})",
                tuple(chunk),
            )

    def count_unsynced(self, project_id: str | None = None) -> int:
        """Count un-acked (``synced=0``) spans, optionally scoped to a project."""
        if project_id is not None:
            row = self._db.fetchone(
                "SELECT COUNT(*) AS n FROM spans WHERE synced = 0 AND project_id = ?",
                (project_id,),
            )
        else:
            row = self._db.fetchone("SELECT COUNT(*) AS n FROM spans WHERE synced = 0")
        return int(row["n"]) if row else 0

    def enforce_buffer_bound(
        self, max_unsynced: int, max_age_days: int, project_id: str | None = None
    ) -> int:
        """Bound the platform re-send queue *without deleting local history*.

        Abandons (marks ``synced=1`` — rows stay in ``local.db`` so the Local UI
        still shows them) un-acked spans that are either older than
        ``max_age_days`` or beyond the ``max_unsynced`` cap (oldest first).
        Returns the count abandoned so the caller can log the drop. Each branch
        is gated on a cheap read so a healthy steady-state export does no write.
        """
        if project_id is None:
            from fastaiagent._internal.project import safe_get_project_id

            project_id = safe_get_project_id()
        abandoned = 0

        # (a) Age bound — abandon spans older than the cutoff.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        old = self._db.fetchall(
            "SELECT span_id FROM spans WHERE synced = 0 AND project_id = ? AND start_time < ?",
            (project_id, cutoff),
        )
        if old:
            self.mark_synced([r["span_id"] for r in old])
            abandoned += len(old)

        # (b) Count bound — abandon the oldest excess beyond the cap.
        remaining = self.count_unsynced(project_id)
        if remaining > max_unsynced:
            excess = remaining - max_unsynced
            rows = self._db.fetchall(
                "SELECT span_id FROM spans WHERE synced = 0 AND project_id = ? "
                "ORDER BY start_time ASC LIMIT ?",
                (project_id, excess),
            )
            if rows:
                self.mark_synced([r["span_id"] for r in rows])
                abandoned += len(rows)

        return abandoned

    def close(self) -> None:
        self._db.close()
