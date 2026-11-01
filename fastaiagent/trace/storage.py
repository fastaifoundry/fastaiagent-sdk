"""Local SQLite trace storage with OTel SpanProcessor."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Collection
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper

logger = logging.getLogger(__name__)

#: ``fastaiagent.trace.provenance.LOCAL_ONLY_ATTR``, repeated to keep this
#: module free of an import cycle (provenance imports nothing from here).
_LOCAL_ONLY_ATTR = "fastaiagent.export.local_only"

#: ``spans.synced``: 0 = queued for the plane, 1 = the plane confirmed it (a
#: 2xx), 2 = never sent and kept here — local-only, recorded offline, dropped by
#: the buffer bound, or recorded while export was paused (1.97.0). ``traces
#: prune`` deletes only 1; until 1.97.0 every "not queued" span was also 1, so
#: prune deleted traces that existed nowhere else.
SYNC_QUEUED, SYNC_ACKED, SYNC_KEPT = 0, 1, 2


def _exporting_to_plane() -> bool:
    """True while ``connect()`` has a platform span exporter registered (1.95.0).

    A span recorded while this is False was never meant for a plane: not
    connected, or ``connect(export_traces=False)``. Queuing it anyway made the
    first later ``connect()`` upload the user's offline history.
    """
    from fastaiagent.client import _connection

    return _connection.is_connected and _connection._platform_processor is not None


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


class SpanRecord(NamedTuple):
    """One span from :meth:`TraceStore.list_spans`, with its tail cursor.

    The cursor is deliberately **not** a field on :class:`SpanData`. That model
    is dumped whole by ``trace.platform_export.to_wire``, so any field added to
    it becomes a new key on the span payload the plane receives — a wire event
    under the cross-repo contract, for something the plane has no use for. A
    local read cursor has no business on the egress model, so it rides
    alongside instead.
    """

    cursor: int
    span: SpanData


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


def _parents_first(spans: list[SpanData]) -> list[SpanData]:
    """Order spans by start time, a parent before its children on a tie (1.89.0).

    Spans are stored when they end, so a child is written before its parent.
    Where the clock is coarse (Windows), a parent and its child often share a
    start time, and the tie fell to write order: the child came first, so the
    trace's "root" was a child and Replay's steps came out of order. ``spans``
    arrives ordered by ``(start_time, rowid)``; the stable sort keeps that order
    for everything but the tie it fixes.
    """
    by_id = {s.span_id: s for s in spans}

    def depth(span: SpanData) -> int:
        d, seen, parent = 0, set(), span.parent_span_id
        while parent and parent in by_id and parent not in seen:
            seen.add(parent)
            d += 1
            parent = by_id[parent].parent_span_id
        return d

    return sorted(spans, key=lambda s: (s.start_time or "", depth(s)))


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
CREATE TABLE IF NOT EXISTS trace_export_pause (
    scope       TEXT PRIMARY KEY,
    resets_at   TEXT NOT NULL,
    paused_at   TEXT NOT NULL
);
"""


class LocalStorageProcessor:
    """OTel SpanProcessor that writes spans to local SQLite."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_config().resolved_trace_db_path
        self._db: SQLiteHelper | None = None
        # Write failures already logged, by error type — one warning each, not
        # one per span.
        self._warned: set[str] = set()
        # Split-DB sink (1.89.0): spans started inside a provenance scope with a
        # ``trace_db`` are written there. span_id -> path, set at start.
        self._span_sinks: dict[int, str] = {}
        self._sink_dbs: dict[str, SQLiteHelper] = {}

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
        # Provenance (1.89.0): inside a ``provenance_scope`` every span is
        # stamped with its origin while it is still mutable. See
        # ``fastaiagent.trace.provenance``.
        from fastaiagent.trace.provenance import current_provenance, stamp_on_start

        stamp_on_start(span)
        prov = current_provenance()
        if prov is not None and prov.trace_db:
            try:
                self._span_sinks[span.get_span_context().span_id] = prov.trace_db
            except Exception:  # pragma: no cover — a sink is best-effort routing
                logger.debug("Could not record the span's trace sink", exc_info=True)

    def _sink_db(self, path: str) -> SQLiteHelper:
        """The store a scoped span is written to (the default one when equal)."""
        import os

        if os.path.abspath(path) == os.path.abspath(self.db_path):
            return self._get_db()
        db = self._sink_dbs.get(path)
        if db is None:
            from fastaiagent.ui.db import init_local_db

            db = init_local_db(path)
            self._sink_dbs[path] = db
        return db

    def on_end(self, span: Any) -> None:
        """Write a finished span to SQLite — and never raise into the caller.

        OTel runs span processors synchronously inside whatever ended the span,
        so an exception here surfaces in *that* code: the user's agent run, or —
        since FastAPI 0.143 traces every request through the global provider by
        default — an HTTP request of any FastAPI app in the process. A local
        trace store that cannot be written (disk full, a deleted directory, a
        locked file) is a lost span, not a failed request. Logged once per error
        type.
        """
        try:
            self._write(span)
        except Exception as e:  # noqa: BLE001 — tracing must never break the app
            kind = type(e).__name__
            if kind not in self._warned:
                self._warned.add(kind)
                logger.warning(
                    "fastaiagent: could not write span %r to %s (%s: %s); the span "
                    "is dropped. Further %s errors for this store are not logged.",
                    getattr(span, "name", "?"),
                    self.db_path,
                    kind,
                    e,
                    kind,
                )

    def _write(self, span: Any) -> None:
        """Called when a span completes — write to SQLite.

        If an opt-in :class:`RedactionPolicy` is installed (via
        ``set_redaction_policy``) with ``mode in {"capture", "both"}``,
        sensitive attribute values are masked here before the JSON blob hits
        SQLite.

        Note: this masks the *local* copy only. Redaction for spans that leave
        the machine is applied independently at the egress boundary — the
        control-plane exporter and :func:`fastaiagent.trace.otel.add_exporter`
        both run :func:`fastaiagent.trace.redaction.apply_export_policy`
        (security_audit_2 N4). That is also where the payload gate
        (``FASTAIAGENT_TRACE_PAYLOADS=0``) drops payloads from export while
        local.db keeps full fidelity.
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
            # 1.89.0 (P9): an exception message can carry a credential: a
            # provider echoing a rejected key, or user code formatting one in.
            # Scrubbed before it is stored, so neither the UI nor the plane
            # drain ever reads it.
            from fastaiagent._internal.scrub import scrub_event_dicts

            events = scrub_event_dicts(events)

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

        sink = self._span_sinks.pop(ctx.span_id, None)
        db = self._sink_db(sink) if sink else self._get_db()
        db.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_span_id, name,
                start_time, end_time, status, attributes, events,
                project_id, synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                # Queued for the plane (``synced=0``) only while trace export is
                # on. A local-only span (a replay, a Playground call, 1.89.0) or
                # one recorded offline (1.95.0) is written as kept, so no
                # exporter, this SDK's or an older one sharing the store, ever
                # sends it, and ``traces prune`` never deletes it.
                SYNC_QUEUED
                if _exporting_to_plane() and not attrs.get(_LOCAL_ONLY_ATTR)
                else SYNC_KEPT,
            ),
        )

    def shutdown(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
        for db in self._sink_dbs.values():
            try:
                db.close()
            except Exception:  # pragma: no cover — shutdown never raises
                logger.debug("Could not close a trace sink", exc_info=True)
        self._sink_dbs.clear()
        self._span_sinks.clear()

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
        """The configured store, or the scoped one inside a provenance scope
        with a ``trace_db`` (a replay written to another store, 1.89.0)."""
        from fastaiagent.trace.provenance import current_provenance

        prov = current_provenance()
        if prov is not None and prov.trace_db:
            return cls(db_path=prov.trace_db)
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

    def list_spans(
        self,
        *,
        since: int = 0,
        limit: int = 100,
        trace_id: str | None = None,
        execution_id: str | None = None,
    ) -> list[SpanRecord]:
        """Return spans across traces, oldest first, for tailing as they land.

        Every "show spans as they arrive" use case previously had to walk trace
        by trace: two streaming demos assumed a method like this one and were
        dead because of it, and example 08 dropped to raw ``sqlite3``.

        ``since`` is a cursor, not a timestamp. Pass ``0`` for the first call,
        then the ``cursor`` of the last record you handled; each call returns
        only rows written after it. SQLite rowids are assigned in insert order
        and never reused within a table, so this reads each span exactly once
        even while ``on_end`` keeps writing. Ordering by rowid rather than
        ``start_time`` matters here: a long span *starts* before a short one
        that finishes first, so start-time order would step over rows that
        arrive later with an earlier timestamp.

        ``trace_id`` narrows to one trace. ``execution_id`` matches the
        ``chain.execution_id`` attribute, so a durable run can be followed
        across the traces it spans.

        Returns at most ``limit`` records. An empty list means "nothing new
        yet", not "end of stream" — keep the cursor and call again.
        """
        if limit <= 0:
            return []

        where = ["rowid > ?"]
        params: list[Any] = [since]
        if trace_id is not None:
            where.append("trace_id = ?")
            params.append(trace_id)
        if execution_id is not None:
            # The attribute key contains a dot, so it must be quoted inside the
            # JSON path or SQLite reads it as a nested object.
            where.append("json_extract(attributes, '$.\"chain.execution_id\"') = ?")
            params.append(execution_id)
        params.append(limit)

        rows = self._db.fetchall(
            f"SELECT rowid, * FROM spans WHERE {' AND '.join(where)} "
            f"ORDER BY rowid LIMIT ?",
            tuple(params),
        )
        return [SpanRecord(cursor=r["rowid"], span=self._row_to_span(r)) for r in rows]

    def get_trace(self, trace_id: str) -> TraceData:
        """Get a complete trace with all its spans."""
        rows = self._db.fetchall(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time, rowid",
            (trace_id,),
        )
        if not rows:
            from fastaiagent._internal.errors import TraceError

            raise TraceError(
                f"Trace '{trace_id}' not found in local storage.\n"
                f"Use TraceStore.list_traces() to see available traces, or check that "
                f"tracing was enabled when the agent ran (trace=True)."
            )

        spans = _parents_first([self._row_to_span(row) for row in rows])

        root = spans[0] if spans else SpanData(span_id="", trace_id=trace_id)
        return TraceData(
            trace_id=trace_id,
            name=root.name,
            start_time=root.start_time,
            end_time=spans[-1].end_time if spans else "",
            status=root.status,
            spans=spans,
        )

    #: How a trace gets its label.
    #:
    #: The root span names the run; ``MIN(name)`` is only the fallback. Until
    #: 1.68.0 the fallback *was* the rule, so the label was the lexicographic
    #: minimum of every span in the trace — measured: a trace whose spans are
    #: ``['swarm.pair', 'agent.alpha']`` listed as ``agent.alpha``. Since
    #: ``agent.*`` and ``chain.*`` both sort ahead of ``supervisor.*`` and
    #: ``swarm.*``, that was most multi-agent traces, and 1.67.0 made it more
    #: visible by giving root spans to three paths that previously had none.
    #:
    #: The fallback still matters: a trace can legitimately arrive with no root
    #: (a sampled export, a foreign ingest, a crash between a child's ``on_end``
    #: and its parent's), and an unnamed row in the list is worse than an
    #: approximate one. The empty-string arm covers direct ingests that write
    #: ``''`` where OTel capture writes NULL.
    TRACE_NAME_SQL = (
        "COALESCE(MIN(CASE WHEN parent_span_id IS NULL OR parent_span_id = '' "
        "THEN name END), MIN(name))"
    )

    #: A trace's status is its **root** span's status (1.89.0), with the same
    #: fallback as :attr:`TRACE_NAME_SQL`. ``MIN(status)`` over every span made a
    #: run that completed after a ``warn`` guardrail (an ERROR guardrail span
    #: under an UNSET root) report ERROR, and a run whose only spans were OK and
    #: UNSET report OK however its root ended. The plane reads the root too.
    TRACE_STATUS_SQL = (
        "COALESCE(MIN(CASE WHEN parent_span_id IS NULL OR parent_span_id = '' "
        "THEN status END), MIN(status))"
    )

    def list_traces(
        self,
        last_hours: float | None = 24,
        *,
        limit: int | None = 100,
        name_filter: str | None = None,
        agent_name: str | None = None,
        exclude_sources: bool | Collection[str] = False,
        **filters: Any,
    ) -> list[TraceSummary]:
        """List traces, newest first.

        Args:
            last_hours: only traces that started within this many hours;
                ``None`` for all time.
            limit: at most this many traces; ``None`` for no limit.
            name_filter: only traces whose name (the root span's) starts with
                this, e.g. ``"chain.sales-sdr"``.
            agent_name: only traces in which an agent of this name ran — at
                the root, or as a child span (a swarm, a chain, a supervisor).
            exclude_sources: ``True`` leaves out every trace with a
                non-production span (a replay, a Playground call, a learn or
                eval run; see :mod:`fastaiagent.trace.provenance`). A
                collection leaves out only those sources. Default: everything.
        """
        if filters:
            warnings.warn(
                f"list_traces() ignores unknown filters: {sorted(filters)}",
                UserWarning,
                stacklevel=2,
            )
        where: list[str] = []
        having: list[str] = []
        params: list[Any] = []
        if agent_name is not None:
            where.append(
                "trace_id IN (SELECT trace_id FROM spans WHERE "
                """json_extract(attributes, '$."agent.name"') = ? OR """
                """json_extract(attributes, '$."fastaiagent.agent.name"') = ?)"""
            )
            params += [agent_name, agent_name]
        if exclude_sources:
            from fastaiagent.trace.provenance import NON_PRODUCTION_SOURCES, non_production_filter

            keep: bool | set[str] = (
                False
                if exclude_sources is True
                else set(NON_PRODUCTION_SOURCES) - set(exclude_sources)
            )
            np_clause, np_params = non_production_filter(include=keep)
            if np_clause:
                where.append(np_clause.removeprefix("AND "))
                params += list(np_params)
        if last_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=last_hours)
            having.append("MIN(start_time) >= ?")
            params.append(cutoff.isoformat())
        if name_filter is not None:
            having.append(f"{self.TRACE_NAME_SQL} LIKE ? ESCAPE '\\'")
            escaped = name_filter.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(escaped + "%")
        sql = f"""SELECT trace_id,
                      {self.TRACE_NAME_SQL} as name,
                      MIN(start_time) as start_time,
                      {self.TRACE_STATUS_SQL} as status,
                      COUNT(*) as span_count
               FROM spans
               {"WHERE " + " AND ".join(where) if where else ""}
               GROUP BY trace_id
               {"HAVING " + " AND ".join(having) if having else ""}
               ORDER BY start_time DESC"""
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self._db.fetchall(sql, tuple(params))
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
                f"""SELECT trace_id,
                          {self.TRACE_NAME_SQL} as name,
                          MIN(start_time) as start_time,
                          {self.TRACE_STATUS_SQL} as status,
                          COUNT(*) as span_count
                   FROM spans
                   WHERE trace_id IN (
                       SELECT trace_id FROM spans WHERE name LIKE ? OR attributes LIKE ?
                   )
                   GROUP BY trace_id
                   ORDER BY start_time DESC
                   LIMIT 100""",
                (f"%{query}%", f"%{query}%"),
            )
        else:
            return self.list_traces(last_hours=None)
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
    # ``LocalStorageProcessor`` writes a span with ``synced=0`` while trace
    # export to a plane is on (``synced=2``, kept, otherwise);
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
        """Mark spans the plane confirmed (a 2xx) — the only ones ``prune`` deletes.

        Chunked to stay under SQLite's bound-variable limit (999).
        """
        self._set_synced(span_ids, SYNC_ACKED)

    def mark_kept(self, span_ids: list[str]) -> None:
        """Take spans off the re-send queue without claiming the plane has them
        (1.97.0): they stay in ``local.db`` and ``prune`` keeps them."""
        self._set_synced(span_ids, SYNC_KEPT)

    def _set_synced(self, span_ids: list[str], value: int) -> None:
        for start in range(0, len(span_ids), 500):
            chunk = span_ids[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            self._db.execute(
                f"UPDATE spans SET synced = ? WHERE span_id IN ({placeholders})",
                (value, *chunk),
            )

    def export_pause(self, scope: str) -> str | None:
        """When trace export to ``scope`` resumes (ISO 8601 UTC), or ``None`` (1.94.0)."""
        row = self._db.fetchone(
            "SELECT resets_at FROM trace_export_pause WHERE scope = ?", (scope,)
        )
        return str(row["resets_at"]) if row else None

    def set_export_pause(self, scope: str, resets_at: str, paused_at: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO trace_export_pause (scope, resets_at, paused_at) "
            "VALUES (?, ?, ?)",
            (scope, resets_at, paused_at),
        )

    def clear_export_pause(self, scope: str) -> None:
        self._db.execute("DELETE FROM trace_export_pause WHERE scope = ?", (scope,))

    def mark_synced_before(self, start_time: str, project_id: str | None = None) -> int:
        """Mark un-acked spans that started before ``start_time`` as done; return
        how many. Spans recorded while export is paused at the plane's limit are
        never sent (1.94.0, S7), so they are kept, not acked (1.97.0)."""
        sql = "UPDATE spans SET synced = ? WHERE synced = 0 AND start_time < ?"
        params: tuple[Any, ...] = (SYNC_KEPT, start_time)
        if project_id is not None:
            sql += " AND project_id = ?"
            params += (project_id,)
        return int(self._db.execute(sql, params).rowcount)

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

        Abandons (marks ``synced=2``, kept — rows stay in ``local.db`` so the
        Local UI still shows them and ``prune`` keeps them) un-acked spans that
        are either older than ``max_age_days`` or beyond the ``max_unsynced``
        cap (oldest first).
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
            self.mark_kept([r["span_id"] for r in old])
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
                self.mark_kept([r["span_id"] for r in rows])
                abandoned += len(rows)

        return abandoned

    def prune_acked(
        self, older_than_days: int | None = None, project_id: str | None = None
    ) -> int:
        """Delete spans the plane confirmed (``synced=1``) to reclaim local.db space.

        A span that never reached the plane (``synced=2``: local-only, offline,
        abandoned by ``enforce_buffer_bound``) and one still queued (``0``) are
        kept: this copy is the only one. Distinct from ``purge``, which deletes
        by trace scope. ``older_than_days`` restricts to rows older than the
        cutoff; ``project_id`` scopes to one project. Returns rows deleted.
        """
        db = self._db
        clauses = ["synced = 1"]
        params: list[Any] = []
        if older_than_days is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=older_than_days)
            ).isoformat()
            clauses.append("start_time < ?")
            params.append(cutoff)
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        where = " AND ".join(clauses)

        row = db.fetchone(
            f"SELECT COUNT(*) AS n FROM spans WHERE {where}", tuple(params)
        )
        n = int(row["n"]) if row else 0
        if n:
            db.execute(f"DELETE FROM spans WHERE {where}", tuple(params))
        return n

    def close(self) -> None:
        self._db.close()
