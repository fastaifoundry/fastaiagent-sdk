"""Trace dashboard endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api", tags=["traces"])


class TraceRow(BaseModel):
    trace_id: str
    name: str
    start_time: str
    end_time: str | None
    status: str
    span_count: int
    duration_ms: int | None
    agent_name: str | None = None
    thread_id: str | None = None
    total_cost_usd: float | None = None
    total_tokens: int | None = None
    runner_type: str = "agent"
    runner_name: str | None = None


class TracesPage(BaseModel):
    rows: list[TraceRow]
    total: int
    page: int
    page_size: int


class SpanRow(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    start_time: str
    end_time: str
    status: str
    attributes: dict[str, Any]
    events: list[dict[str, Any]]


class SpanTreeNode(BaseModel):
    span: SpanRow
    children: list[SpanTreeNode]


SpanTreeNode.model_rebuild()


def _ms(start: str, end: str) -> int | None:
    try:
        a = datetime.fromisoformat(start)
        b = datetime.fromisoformat(end)
        return int((b - a).total_seconds() * 1000)
    except (ValueError, TypeError):
        return None


def _row_attrs(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("attributes") or "{}"
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _summarize_trace(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a trace's span rows into the fields the UI card needs."""
    agent_name: str | None = None
    thread_id: str | None = None
    total_cost = 0.0
    total_tokens = 0
    start_time = ""
    end_time = ""
    root_name = ""
    status = "OK"
    from fastaiagent.ui.attrs import attr

    runner_type: str | None = None
    runner_name: str | None = None
    for sp in spans:
        attrs = _row_attrs(sp)
        span_name = sp.get("name") or ""
        if runner_type is None:
            explicit = attr(attrs, "runner.type")
            if explicit:
                runner_type = str(explicit)
            elif span_name.startswith("chain."):
                runner_type = "chain"
            elif span_name.startswith("swarm."):
                runner_type = "swarm"
            elif span_name.startswith("supervisor."):
                runner_type = "supervisor"
            elif span_name.startswith("agent."):
                runner_type = "agent"
        if runner_name is None and runner_type:
            runner_name = attr(attrs, f"{runner_type}.name") or attr(attrs, "agent.name")
        if agent_name is None:
            agent_name = attr(attrs, "agent.name")
        if thread_id is None:
            thread_id = attr(attrs, "thread.id") or attr(attrs, "agent.thread_id")
        from fastaiagent.ui.attrs import attr, trace_cost_usd

        reported_cost = trace_cost_usd(attrs)
        input_tokens = int(attrs.get("gen_ai.usage.input_tokens") or 0)
        output_tokens = int(attrs.get("gen_ai.usage.output_tokens") or 0)
        # The SDK's root agent span also carries a rolled-up total.
        agent_total = attr(attrs, "agent.tokens_used")
        if agent_total is not None:
            try:
                total_tokens += int(agent_total)
            except (TypeError, ValueError):
                pass
        else:
            total_tokens += input_tokens + output_tokens
        if reported_cost is not None:
            total_cost += reported_cost
        elif input_tokens or output_tokens:
            from fastaiagent.ui.pricing import compute_cost_usd

            estimated = compute_cost_usd(
                attrs.get("gen_ai.request.model"), input_tokens, output_tokens
            )
            if estimated is not None:
                total_cost += estimated
        if not start_time or (sp.get("start_time") and sp["start_time"] < start_time):
            start_time = sp.get("start_time", "")
        if sp.get("end_time", "") > end_time:
            end_time = sp["end_time"]
        if not sp.get("parent_span_id") and not root_name:
            root_name = sp.get("name") or ""
            status = sp.get("status") or status
    return {
        "agent_name": agent_name,
        "thread_id": thread_id,
        "total_cost_usd": total_cost if total_cost else None,
        "total_tokens": total_tokens or None,
        "start_time": start_time,
        "end_time": end_time,
        "root_name": root_name,
        "status": status,
        "runner_type": runner_type or "agent",
        "runner_name": runner_name or agent_name,
    }


@router.get("/traces", response_model=TracesPage)
def list_traces(
    request: Request,
    _user: str = Depends(require_session),
    agent: str | None = Query(default=None),
    trace_status: str | None = Query(default=None, alias="status"),
    q: str | None = Query(default=None, description="Full-text match across span content"),
    thread_id: str | None = Query(default=None),
    runner_type: str | None = Query(
        default=None,
        pattern="^(agent|chain|swarm|supervisor)$",
        description="Filter by root runner type (agent|chain|swarm|supervisor).",
    ),
    runner_name: str | None = Query(
        default=None,
        description="Filter by specific chain/swarm/supervisor name (pairs with runner_type).",
    ),
    min_duration_ms: int | None = Query(default=None),
    max_duration_ms: int | None = Query(default=None),
    min_cost: float | None = Query(default=None),
    min_tokens: int | None = Query(default=None),
    since: str | None = Query(default=None, description="ISO timestamp lower bound"),
    until: str | None = Query(default=None, description="ISO timestamp upper bound"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> TracesPage:
    ctx = get_context(request)
    db = ctx.db()
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if since:
            clauses.append("start_time >= ?")
            params.append(since)
        if until:
            clauses.append("start_time <= ?")
            params.append(until)
        if trace_status:
            clauses.append("status = ?")
            params.append(trace_status)
        if q:
            clauses.append("(name LIKE ? OR attributes LIKE ? OR events LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        trace_rows = db.fetchall(
            f"""SELECT trace_id, MAX(start_time) AS latest
               FROM spans
               {where_sql}
               GROUP BY trace_id
               ORDER BY latest DESC""",
            tuple(params),
        )
        total = len(trace_rows)
        offset = (page - 1) * page_size
        paginated = trace_rows[offset : offset + page_size]

        out: list[TraceRow] = []
        for row in paginated:
            spans = db.fetchall(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (row["trace_id"],),
            )
            summary = _summarize_trace(spans)

            if agent and summary["agent_name"] != agent:
                continue
            if thread_id and summary["thread_id"] != thread_id:
                continue
            if runner_type and summary["runner_type"] != runner_type:
                continue
            if runner_name and summary["runner_name"] != runner_name:
                continue
            duration = _ms(summary["start_time"], summary["end_time"])
            if min_duration_ms is not None and (duration is None or duration < min_duration_ms):
                continue
            if max_duration_ms is not None and (duration is None or duration > max_duration_ms):
                continue
            if min_cost is not None and (
                summary["total_cost_usd"] is None or summary["total_cost_usd"] < min_cost
            ):
                continue
            if min_tokens is not None and (
                summary["total_tokens"] is None or summary["total_tokens"] < min_tokens
            ):
                continue

            out.append(
                TraceRow(
                    trace_id=row["trace_id"],
                    name=summary["root_name"] or row["trace_id"],
                    start_time=summary["start_time"],
                    end_time=summary["end_time"] or None,
                    status=summary["status"],
                    span_count=len(spans),
                    duration_ms=duration,
                    agent_name=summary["agent_name"],
                    thread_id=summary["thread_id"],
                    total_cost_usd=summary["total_cost_usd"],
                    total_tokens=summary["total_tokens"],
                    runner_type=summary["runner_type"],
                    runner_name=summary["runner_name"],
                )
            )
        return TracesPage(rows=out, total=total, page=page, page_size=page_size)
    finally:
        db.close()


@router.get("/traces/threads")
def list_threads(request: Request, _user: str = Depends(require_session)) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        from fastaiagent.ui.attrs import attr

        rows = db.fetchall("SELECT trace_id, attributes FROM spans WHERE parent_span_id IS NULL")
        groups: dict[str, list[str]] = {}
        for row in rows:
            attrs = _row_attrs(row)
            tid = attr(attrs, "thread.id") or attr(attrs, "agent.thread_id")
            if not tid:
                continue
            groups.setdefault(tid, []).append(row["trace_id"])
        return {"threads": [{"thread_id": k, "trace_ids": v} for k, v in groups.items()]}
    finally:
        db.close()


@router.get("/traces/compare")
def compare_traces(
    request: Request,
    a: str,
    b: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:

        def spans_for(trace_id: str) -> list[SpanRow]:
            rows = db.fetchall(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (trace_id,),
            )
            return [_row_to_span(r) for r in rows]

        return {
            "a": [s.model_dump() for s in spans_for(a)],
            "b": [s.model_dump() for s in spans_for(b)],
        }
    finally:
        db.close()


@router.get("/traces/{trace_id}/scores")
def get_trace_scores(
    request: Request,
    trace_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Return every score-ish artifact that points at this trace.

    - Guardrail events where ``guardrail_events.trace_id == trace_id``.
    - Eval cases where ``eval_cases.trace_id == trace_id`` (joined with
      their run for context).
    """
    ctx = get_context(request)
    db = ctx.db()
    try:
        guardrail_rows = db.fetchall(
            """SELECT event_id, guardrail_name, guardrail_type, position,
                      outcome, score, message, agent_name, timestamp
               FROM guardrail_events
               WHERE trace_id = ?
               ORDER BY timestamp""",
            (trace_id,),
        )
        eval_rows = db.fetchall(
            """SELECT c.case_id, c.run_id, c.ordinal, c.per_scorer,
                      c.input, c.expected_output, c.actual_output,
                      r.run_name, r.dataset_name, r.started_at
               FROM eval_cases c
               LEFT JOIN eval_runs r ON c.run_id = r.run_id
               WHERE c.trace_id = ?
               ORDER BY r.started_at DESC""",
            (trace_id,),
        )
        for row in eval_rows:
            for key in ("per_scorer", "input", "expected_output", "actual_output"):
                if row.get(key):
                    try:
                        row[key] = json.loads(row[key])
                    except json.JSONDecodeError:
                        pass
        return {
            "trace_id": trace_id,
            "guardrail_events": guardrail_rows,
            "eval_cases": eval_rows,
        }
    finally:
        db.close()


@router.get("/threads/{thread_id}")
def get_thread(
    request: Request,
    thread_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """List every trace sharing ``thread_id``, newest first, with summaries."""
    ctx = get_context(request)
    db = ctx.db()
    try:
        # Accept every prefix variant so traces from older SDK releases still
        # surface here. Three LIKEs rather than one regex — SQLite LIKE is
        # cheap on the small tables this tool targets.
        trace_rows = db.fetchall(
            """SELECT DISTINCT trace_id
               FROM spans
               WHERE attributes LIKE ?
                  OR attributes LIKE ?
                  OR attributes LIKE ?""",
            (
                f'%"fastaiagent.thread.id": "{thread_id}"%',
                f'%"thread.id": "{thread_id}"%',
                f'%"agent.thread_id": "{thread_id}"%',
            ),
        )
        out: list[dict[str, Any]] = []
        for row in trace_rows:
            spans = db.fetchall(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (row["trace_id"],),
            )
            summary = _summarize_trace(spans)
            duration = _ms(summary["start_time"], summary["end_time"])
            out.append(
                {
                    "trace_id": row["trace_id"],
                    "name": summary["root_name"] or row["trace_id"],
                    "start_time": summary["start_time"],
                    "end_time": summary["end_time"] or None,
                    "status": summary["status"],
                    "span_count": len(spans),
                    "duration_ms": duration,
                    "agent_name": summary["agent_name"],
                    "thread_id": summary["thread_id"],
                    "total_cost_usd": summary["total_cost_usd"],
                    "total_tokens": summary["total_tokens"],
                    "runner_type": summary["runner_type"],
                    "runner_name": summary["runner_name"],
                }
            )
        out.sort(key=lambda r: r["start_time"], reverse=True)
        return {"thread_id": thread_id, "traces": out}
    finally:
        db.close()


@router.get("/traces/{trace_id}")
def get_trace(
    request: Request,
    trace_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        rows = db.fetchall(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        )
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Trace '{trace_id}' not found")
        spans = [_row_to_span(r) for r in rows]
        summary = _summarize_trace(rows)
        return {
            "trace_id": trace_id,
            "name": summary["root_name"] or trace_id,
            "status": summary["status"],
            "start_time": summary["start_time"],
            "end_time": summary["end_time"],
            "agent_name": summary["agent_name"],
            "thread_id": summary["thread_id"],
            "total_cost_usd": summary["total_cost_usd"],
            "total_tokens": summary["total_tokens"],
            "span_count": len(spans),
            "runner_type": summary["runner_type"],
            "runner_name": summary["runner_name"],
            "spans": [s.model_dump() for s in spans],
        }
    finally:
        db.close()


@router.get("/traces/{trace_id}/spans")
def get_spans(
    request: Request,
    trace_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        rows = db.fetchall(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        )
        if not rows:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Trace '{trace_id}' not found")
        return {"tree": _build_tree(rows).model_dump()}
    finally:
        db.close()


@router.get("/traces/{trace_id}/spans/{span_id}/attachments")
def list_span_attachments(
    request: Request,
    trace_id: str,
    span_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Return the attachment metadata for a span (no binary payload).

    Frontend uses this to render thumbnail tiles in the trace inspector;
    each row's ``attachment_id`` is then passed to the binary endpoint
    below for the thumbnail or full image bytes.
    """
    from fastaiagent.trace.attachments import list_attachments_for_span

    ctx = get_context(request)
    db = ctx.db()
    try:
        records = list_attachments_for_span(db=db, trace_id=trace_id, span_id=span_id)
        return {
            "attachments": [
                {
                    "attachment_id": r.attachment_id,
                    "media_type": r.media_type,
                    "size_bytes": r.size_bytes,
                    "metadata": r.metadata,
                    "has_full_data": r.full_data is not None,
                    "created_at": r.created_at,
                }
                for r in records
            ]
        }
    finally:
        db.close()


@router.get("/traces/{trace_id}/spans/{span_id}/attachments/{attachment_id}")
def get_span_attachment(
    request: Request,
    trace_id: str,
    span_id: str,
    attachment_id: str,
    full: bool = Query(default=False, description="Return original bytes when available."),
    _user: str = Depends(require_session),
) -> Any:
    """Stream the binary payload (thumbnail by default, original when ?full=1).

    The thumbnail is always available for image / PDF attachments — PDFs
    render as a JPEG of page 1. ``full=1`` returns ``full_data`` when the
    SDK was running with ``trace_full_images=True``; otherwise 404.
    """
    from fastapi.responses import Response

    from fastaiagent.trace.attachments import get_attachment

    ctx = get_context(request)
    db = ctx.db()
    try:
        record = get_attachment(db=db, attachment_id=attachment_id)
        if record is None or record.trace_id != trace_id or record.span_id != span_id:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Attachment '{attachment_id}' not found on span '{span_id}'",
            )
        if full:
            if record.full_data is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    "full_data not stored — set fa.config.trace_full_images=True before the run",
                )
            return Response(content=record.full_data, media_type=record.media_type)
        if record.thumbnail is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no thumbnail for this attachment")
        return Response(content=record.thumbnail, media_type="image/jpeg")
    finally:
        db.close()


@router.get("/traces/{trace_id}/export")
def export_trace(
    request: Request,
    trace_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    return get_trace(request, trace_id, _user)


class BulkDeleteRequest(BaseModel):
    trace_ids: list[str]


def _delete_traces(db: Any, trace_ids: list[str]) -> int:
    """Delete spans, notes, favorites, and guardrail rows for the given traces.

    Returns how many traces had at least one span removed.
    """
    if not trace_ids:
        return 0
    deleted = 0
    for tid in trace_ids:
        before = db.fetchone("SELECT COUNT(*) AS n FROM spans WHERE trace_id = ?", (tid,))
        if not before or (before.get("n") or 0) == 0:
            continue
        db.execute("DELETE FROM spans WHERE trace_id = ?", (tid,))
        db.execute("DELETE FROM trace_notes WHERE trace_id = ?", (tid,))
        db.execute("DELETE FROM trace_favorites WHERE trace_id = ?", (tid,))
        db.execute("DELETE FROM guardrail_events WHERE trace_id = ?", (tid,))
        # Detach eval cases but keep the run — evals aren't owned by traces.
        db.execute("UPDATE eval_cases SET trace_id = NULL WHERE trace_id = ?", (tid,))
        deleted += 1
    return deleted


@router.delete("/traces/{trace_id}")
def delete_trace(
    request: Request,
    trace_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        count = _delete_traces(db, [trace_id])
        if count == 0:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Trace '{trace_id}' not found")
        return {"deleted": count}
    finally:
        db.close()


@router.post("/traces/bulk-delete")
def bulk_delete_traces(
    request: Request,
    body: BulkDeleteRequest,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        count = _delete_traces(db, body.trace_ids)
        return {"deleted": count, "requested": len(body.trace_ids)}
    finally:
        db.close()


class NoteRequest(BaseModel):
    note: str


@router.post("/traces/{trace_id}/notes")
def set_note(
    request: Request,
    trace_id: str,
    body: NoteRequest,
    _user: str = Depends(require_session),
) -> dict[str, str]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        now = datetime.now(tz=timezone.utc).isoformat()
        db.execute(
            """INSERT INTO trace_notes (trace_id, note, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(trace_id) DO UPDATE SET
                 note = excluded.note,
                 updated_at = excluded.updated_at""",
            (trace_id, body.note, now),
        )
        return {"status": "ok"}
    finally:
        db.close()


@router.post("/traces/{trace_id}/favorite")
def toggle_favorite(
    request: Request,
    trace_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        existing = db.fetchone(
            "SELECT trace_id FROM trace_favorites WHERE trace_id = ?",
            (trace_id,),
        )
        if existing:
            db.execute("DELETE FROM trace_favorites WHERE trace_id = ?", (trace_id,))
            return {"favorited": False}
        db.execute(
            "INSERT INTO trace_favorites (trace_id, created_at) VALUES (?, ?)",
            (trace_id, datetime.now(tz=timezone.utc).isoformat()),
        )
        return {"favorited": True}
    finally:
        db.close()


def _row_to_span(row: dict[str, Any]) -> SpanRow:
    return SpanRow(
        span_id=row["span_id"],
        trace_id=row["trace_id"],
        parent_span_id=row.get("parent_span_id"),
        name=row.get("name") or "",
        start_time=row.get("start_time") or "",
        end_time=row.get("end_time") or "",
        status=row.get("status") or "OK",
        attributes=_row_attrs(row),
        events=json.loads(row.get("events") or "[]"),
    )


def _build_tree(rows: list[dict[str, Any]]) -> SpanTreeNode:
    spans = [_row_to_span(r) for r in rows]
    by_id = {s.span_id: SpanTreeNode(span=s, children=[]) for s in spans}
    root: SpanTreeNode | None = None
    for span in spans:
        node = by_id[span.span_id]
        if span.parent_span_id and span.parent_span_id in by_id:
            by_id[span.parent_span_id].children.append(node)
        else:
            root = root or node
    if root is None:
        root = next(iter(by_id.values()))
    return root
