"""Local SQLite trace storage with OTel SpanProcessor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper


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
    events: list[dict] = Field(default_factory=list)


class TraceData(BaseModel):
    """Complete trace with all spans."""

    trace_id: str
    name: str = ""
    start_time: str = ""
    end_time: str = ""
    status: str = "OK"
    metadata: dict[str, Any] = Field(default_factory=dict)
    spans: list[SpanData] = Field(default_factory=list)


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
    events TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans (trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans (start_time);
"""


class LocalStorageProcessor:
    """OTel SpanProcessor that writes spans to local SQLite."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_config().trace_db_path
        self._db: SQLiteHelper | None = None

    def _on_ending(self, span: Any) -> None:
        """Called when a span is ending (before on_end)."""
        pass

    def _get_db(self) -> SQLiteHelper:
        if self._db is None:
            self._db = SQLiteHelper(self.db_path)
            for stmt in _SCHEMA.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    self._db.execute(stmt)
        return self._db

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        pass

    def on_end(self, span: Any) -> None:
        """Called when a span completes — write to SQLite."""
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

        events = []
        if hasattr(span, "events") and span.events:
            for e in span.events:
                events.append({"name": e.name, "timestamp": str(e.timestamp)})

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

        db = self._get_db()
        db.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_span_id, name,
                start_time, end_time, status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        self.db_path = db_path or get_config().trace_db_path
        self._db = SQLiteHelper(self.db_path)

    @classmethod
    def default(cls) -> TraceStore:
        return cls()

    def get_trace(self, trace_id: str) -> TraceData:
        """Get a complete trace with all its spans."""
        rows = self._db.fetchall(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        )
        if not rows:
            from fastaiagent._internal.errors import TraceError

            raise TraceError(f"Trace '{trace_id}' not found")

        spans = []
        for row in rows:
            spans.append(
                SpanData(
                    span_id=row["span_id"],
                    trace_id=row["trace_id"],
                    parent_span_id=row["parent_span_id"],
                    name=row["name"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    status=row["status"],
                    attributes=json.loads(row["attributes"]) if row["attributes"] else {},
                    events=json.loads(row["events"]) if row["events"] else [],
                )
            )

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

    def close(self) -> None:
        self._db.close()
