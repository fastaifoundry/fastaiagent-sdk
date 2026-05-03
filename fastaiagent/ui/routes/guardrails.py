"""Guardrail events list, detail, and false-positive annotation endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from fastaiagent.ui.deps import get_context, project_filter, require_session

router = APIRouter(prefix="/api/guardrail-events", tags=["guardrails"])


class GuardrailEvent(BaseModel):
    event_id: str
    trace_id: str | None
    span_id: str | None
    guardrail_name: str
    guardrail_type: str | None
    position: str | None
    outcome: str | None
    score: float | None
    message: str | None
    agent_name: str | None
    timestamp: str | None
    metadata: dict[str, Any]
    false_positive: bool = False
    false_positive_at: str | None = None


def _row_to_event(r: dict[str, Any]) -> GuardrailEvent:
    """Project a SQLite row onto the public GuardrailEvent shape."""
    return GuardrailEvent(
        event_id=r["event_id"],
        trace_id=r.get("trace_id"),
        span_id=r.get("span_id"),
        guardrail_name=r["guardrail_name"],
        guardrail_type=r.get("guardrail_type"),
        position=r.get("position"),
        outcome=r.get("outcome"),
        score=r.get("score"),
        message=r.get("message"),
        agent_name=r.get("agent_name"),
        timestamp=r.get("timestamp"),
        metadata=json.loads(r.get("metadata") or "{}"),
        false_positive=bool(r.get("false_positive") or 0),
        false_positive_at=r.get("false_positive_at"),
    )


@router.get("")
def list_events(
    request: Request,
    _user: str = Depends(require_session),
    rule: str | None = Query(default=None),
    outcome: str | None = Query(default=None),
    agent: str | None = Query(default=None),
    type: str | None = Query(
        default=None,
        description=(
            "Filter by ``guardrail_type``: code / regex / llm_judge / "
            "schema / classifier."
        ),
    ),
    position: str | None = Query(
        default=None,
        description="Filter by ``position``: input / output / tool_call / tool_result.",
    ),
    false_positive: bool | None = Query(
        default=None,
        description="When set, restrict to events flagged as false positive.",
    ),
    since: str | None = Query(default=None),
    until: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if rule:
            clauses.append("guardrail_name = ?")
            params.append(rule)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        if agent:
            clauses.append("agent_name = ?")
            params.append(agent)
        if type:
            clauses.append("guardrail_type = ?")
            params.append(type)
        if position:
            clauses.append("position = ?")
            params.append(position)
        if false_positive is not None:
            clauses.append("false_positive = ?")
            params.append(1 if false_positive else 0)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if ctx.project_id:
            clauses.append("project_id = ?")
            params.append(ctx.project_id)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        count_row = db.fetchone(
            f"SELECT COUNT(*) AS n FROM guardrail_events {where_sql}", tuple(params)
        )
        total = int((count_row or {}).get("n") or 0)

        rows = db.fetchall(
            f"""SELECT * FROM guardrail_events
                {where_sql}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?""",
            tuple(params) + (page_size, (page - 1) * page_size),
        )

        events = [_row_to_event(r).model_dump() for r in rows]
        return {"rows": events, "total": total, "page": page, "page_size": page_size}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Detail endpoint
# ---------------------------------------------------------------------------


_AGENT_INPUT_KEYS = (
    "fastaiagent.agent.input",
    "agent.input",
    "gen_ai.request.messages",
    "tool.args",
    "fastaiagent.tool.args",
    "tool.input",
)

_AGENT_OUTPUT_KEYS = (
    "fastaiagent.agent.output",
    "agent.output",
    "gen_ai.response.content",
    "tool.result",
    "fastaiagent.tool.result",
    "tool.output",
)


def _pick(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in attrs and attrs[k] is not None:
            return attrs[k]
    return None


def _coerce_text(value: Any) -> str | None:
    """Render a span-attribute value as a readable string for the UI."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


def _load_triggering_span(
    db: Any, span_id: str | None, project_id: str
) -> dict[str, Any] | None:
    """Fetch the span that this guardrail evaluated, if we can find it.

    Reads the recorded span_id from the event row. Returns ``None`` when
    the span was never persisted (e.g. event recorded outside an OTel
    context, or before tracing was enabled).
    """
    if not span_id:
        return None
    pid_clause, pid_params = (
        ("AND project_id = ?", (project_id,))
        if project_id
        else ("", ())
    )
    row = db.fetchone(
        f"SELECT * FROM spans WHERE span_id = ? {pid_clause} LIMIT 1",
        (span_id, *pid_params),
    )
    return dict(row) if row else None


def _surrounding_spans(
    db: Any, trace_id: str | None, project_id: str, limit: int = 8
) -> list[dict[str, Any]]:
    """Up to ``limit`` spans from the same trace, ordered chronologically.

    Used by the UI's "context" section to render the conversation around
    the trigger so devs can judge whether the guardrail fired correctly.
    """
    if not trace_id:
        return []
    pid_clause, pid_params = (
        ("AND project_id = ?", (project_id,))
        if project_id
        else ("", ())
    )
    rows = db.fetchall(
        f"""SELECT span_id, name, start_time, end_time, status, attributes
            FROM spans
            WHERE trace_id = ? {pid_clause}
            ORDER BY start_time ASC
            LIMIT ?""",
        (trace_id, *pid_params, limit),
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            attrs = json.loads(r.get("attributes") or "{}")
        except json.JSONDecodeError:
            attrs = {}
        out.append(
            {
                "span_id": r["span_id"],
                "name": r["name"],
                "start_time": r.get("start_time"),
                "end_time": r.get("end_time"),
                "status": r.get("status"),
                "input": _coerce_text(_pick(attrs, _AGENT_INPUT_KEYS)),
                "output": _coerce_text(_pick(attrs, _AGENT_OUTPUT_KEYS)),
            }
        )
    return out


def _sibling_events(
    db: Any,
    event_id: str,
    trace_id: str | None,
    span_id: str | None,
    project_id: str,
) -> list[dict[str, Any]]:
    """Other guardrails that ran on the same content.

    Surfaces both passed and failed siblings so the developer can see, for
    example, that a PII filter blocked while a toxicity check passed —
    important for understanding *which* rule actually mattered.
    """
    if not trace_id:
        return []
    clauses = ["trace_id = ?", "event_id != ?"]
    params: list[Any] = [trace_id, event_id]
    if span_id:
        clauses.append("span_id = ?")
        params.append(span_id)
    if project_id:
        clauses.append("project_id = ?")
        params.append(project_id)
    rows = db.fetchall(
        f"""SELECT * FROM guardrail_events
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp ASC""",
        tuple(params),
    )
    return [_row_to_event(r).model_dump() for r in rows]


def _trigger_payload(
    span: dict[str, Any] | None, position: str | None
) -> dict[str, Any]:
    """Reconstruct what the guardrail evaluated.

    Pulls input vs output off the triggering span based on the guardrail's
    declared position. ``position == "input"`` maps to span input; output
    / tool_result map to span output. Tool-call positions render whichever
    side carries arguments. Returns a structured payload the UI can lay out.
    """
    if span is None:
        return {"kind": "unknown", "text": None, "content_type": None}
    try:
        attrs = json.loads(span.get("attributes") or "{}")
    except json.JSONDecodeError:
        attrs = {}
    if position in ("output", "tool_result"):
        keys = _AGENT_OUTPUT_KEYS
        kind = "agent_output" if position == "output" else "tool_result"
    else:
        keys = _AGENT_INPUT_KEYS
        kind = "agent_input" if position == "input" else "tool_call"
    raw = _pick(attrs, keys)
    return {
        "kind": kind,
        "text": _coerce_text(raw),
        "content_type": _content_type_for(position),
        "span_name": span.get("name"),
        "status": span.get("status"),
    }


def _content_type_for(position: str | None) -> str:
    return {
        "input": "user_input",
        "output": "agent_output",
        "tool_call": "tool_call",
        "tool_result": "tool_response",
    }.get(position or "", "unknown")


@router.get("/{event_id}")
def get_event_detail(
    request: Request,
    event_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Full detail for a single guardrail event.

    Joins the event row with its triggering span (if any), surrounding
    span context, and sibling events that ran on the same content. This
    is the data behind the three-panel detail view (trigger / rule /
    outcome) plus the conversation-context section below.
    """
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        row = db.fetchone(
            f"SELECT * FROM guardrail_events WHERE event_id = ? {pid_clause}",
            (event_id, *pid_params),
        )
        if row is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Guardrail event '{event_id}' not found"
            )
        event = _row_to_event(row)
        triggering_span = _load_triggering_span(
            db, event.span_id, ctx.project_id
        )
        return {
            "event": event.model_dump(),
            "trigger": _trigger_payload(triggering_span, event.position),
            "context": {
                "spans": _surrounding_spans(
                    db, event.trace_id, ctx.project_id
                ),
                "sibling_events": _sibling_events(
                    db, event.event_id, event.trace_id, event.span_id, ctx.project_id
                ),
            },
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# False-positive annotation
# ---------------------------------------------------------------------------


class FalsePositiveRequest(BaseModel):
    false_positive: bool
    note: str | None = None  # accepted for forward compatibility; not stored yet


@router.patch("/{event_id}/false-positive")
def mark_false_positive(
    request: Request,
    event_id: str,
    body: FalsePositiveRequest,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Toggle the ``false_positive`` flag on a guardrail event.

    Project-scoped so cross-project event ids can't be flipped. The flag
    persists across server restarts (lives on the event row), survives
    page refreshes, and feeds into the existing list filter.
    """
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        existing = db.fetchone(
            f"SELECT event_id FROM guardrail_events WHERE event_id = ? {pid_clause}",
            (event_id, *pid_params),
        )
        if existing is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Guardrail event '{event_id}' not found"
            )
        now = datetime.now(tz=timezone.utc).isoformat()
        db.execute(
            f"""UPDATE guardrail_events
                SET false_positive = ?, false_positive_at = ?
                WHERE event_id = ? {pid_clause}""",
            (1 if body.false_positive else 0, now, event_id, *pid_params),
        )
        return {
            "event_id": event_id,
            "false_positive": body.false_positive,
            "false_positive_at": now,
        }
    finally:
        db.close()
