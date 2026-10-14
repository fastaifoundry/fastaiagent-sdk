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
    # What the failure cost. ``action`` is what the rule was configured to do,
    # ``action_taken`` what it actually did — they differ whenever the action
    # could not get what it asked for (an errored check always blocks; a mask
    # with nothing to mask blocks). Both are None for events recorded before
    # 1.57.0.
    action: str | None = None
    action_taken: str | None = None
    severity: str | None = None
    floor: bool = False
    # 1.89.0 (local.db v25). All None on rows written before 1.89.0, which the
    # UI shows as legacy. ``effect`` is what the check did to the run; see
    # fastaiagent.guardrail.actions.effect_of.
    effect: str | None = None
    final_reason: str | None = None
    latency_ms: int | None = None
    attempt: int | None = None
    batch_id: str | None = None
    guardrail_span_id: str | None = None
    judged_ref: str | None = None
    rule: dict[str, Any] | None = None
    blocking: bool | None = None
    on_error: str | None = None
    origin: str | None = None


def _parse_rule(raw: Any) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        rule = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return rule if isinstance(rule, dict) else None


def _row_to_event(r: dict[str, Any]) -> GuardrailEvent:
    """Project a SQLite row onto the public GuardrailEvent shape."""
    rule = _parse_rule(r.get("rule"))
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
        action=r.get("action"),
        action_taken=r.get("action_taken"),
        severity=r.get("severity"),
        floor=bool(r.get("floor") or 0),
        effect=r.get("effect"),
        final_reason=r.get("final_reason"),
        latency_ms=r.get("latency_ms"),
        attempt=r.get("attempt"),
        batch_id=r.get("batch_id"),
        guardrail_span_id=r.get("guardrail_span_id"),
        judged_ref=r.get("judged_ref"),
        rule=rule,
        blocking=(bool(rule["blocking"]) if rule and "blocking" in rule else None),
        on_error=(rule or {}).get("on_error"),
        origin=(rule or {}).get("origin"),
    )


@router.get("")
def list_events(
    request: Request,
    _user: str = Depends(require_session),
    rule: str | None = Query(default=None),
    outcome: str | None = Query(
        default=None,
        description=(
            "Filter by ``outcome``: passed / blocked / warned / errored / "
            "filtered (the payload was masked or overridden and the run continued) / "
            "reasked (the model was asked again; 1.89.0)."
        ),
    ),
    hide_passes: bool = Query(
        default=False,
        description="Leave out clean passes (outcome = passed). The UI sets this by default.",
    ),
    exclude_experiments: bool = Query(
        default=False,
        description=(
            "Leave out checks from replay reruns, Playground calls and other "
            "non-production traces (1.89.0)."
        ),
    ),
    agent: str | None = Query(default=None),
    type: str | None = Query(
        default=None,
        description=(
            "Filter by ``guardrail_type``: code / regex / llm_judge / "
            "schema / classifier / content_safety / groundedness / topic / pii / secrets."
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
        if hide_passes:
            clauses.append("(outcome IS NULL OR outcome != 'passed')")
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
        if exclude_experiments:
            from fastaiagent.trace.provenance import non_production_filter

            np_clause, np_params = non_production_filter()
            # A row with no trace has no provenance and is kept.
            where_sql = f"{where_sql or 'WHERE 1=1'} AND (trace_id IS NULL OR (1=1 {np_clause}))"
            params.extend(np_params)
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
        body: dict[str, Any] = {
            "rows": events,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
        if total == 0:
            # Rows are written by default only since 1.89.0. Say so when a
            # database holds guardrail spans from before then.
            pid_clause, pid_params = project_filter(ctx)
            legacy = db.fetchone(
                f"SELECT COUNT(*) AS n FROM spans WHERE name LIKE 'guardrail.%' {pid_clause}",
                tuple(pid_params),
            )
            body["legacy_guardrail_spans"] = int((legacy or {}).get("n") or 0)
        return body
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


def _load_triggering_span(db: Any, span_id: str | None, project_id: str) -> dict[str, Any] | None:
    """Fetch the span that this guardrail evaluated, if we can find it.

    Reads the recorded span_id from the event row. Returns ``None`` when
    the span was never persisted (e.g. event recorded outside an OTel
    context, or before tracing was enabled).
    """
    if not span_id:
        return None
    pid_clause, pid_params = ("AND project_id = ?", (project_id,)) if project_id else ("", ())
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
    pid_clause, pid_params = ("AND project_id = ?", (project_id,)) if project_id else ("", ())
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
    batch_id: str | None = None,
) -> list[dict[str, Any]]:
    """Other guardrails that ran on the same content.

    Surfaces both passed and failed siblings so the developer can see, for
    example, that a PII filter blocked while a toxicity check passed —
    important for understanding *which* rule actually mattered.

    Since 1.89.0 a position pass shares a ``batch_id``; rows written before
    then are grouped by trace and span.
    """
    if batch_id:
        clauses = ["batch_id = ?", "event_id != ?"]
        params: list[Any] = [batch_id, event_id]
    elif not trace_id:
        return []
    else:
        clauses = ["trace_id = ?", "event_id != ?"]
        params = [trace_id, event_id]
        if span_id:
            clauses.append("span_id = ?")
            params.append(span_id)
    if project_id:
        clauses.append("project_id = ?")
        params.append(project_id)
    rows = db.fetchall(
        f"""SELECT * FROM guardrail_events
            WHERE {" AND ".join(clauses)}
            ORDER BY timestamp ASC""",
        tuple(params),
    )
    return [_row_to_event(r).model_dump() for r in rows]


def _trigger_payload(span: dict[str, Any] | None, position: str | None) -> dict[str, Any]:
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


def _judged(db: Any, judged_ref: str | None) -> dict[str, Any]:
    """The text the rule judged, from ``guardrail_payloads`` (1.89.0).

    ``recorded`` is False for a row written before 1.89.0, or one whose payload
    was purged: the UI then says so instead of guessing from the parent span.
    """
    if not judged_ref:
        return {"text": None, "recorded": False}
    try:
        row = db.fetchone("SELECT text FROM guardrail_payloads WHERE payload_id = ?", (judged_ref,))
    except Exception:
        row = None
    if row is None:
        return {"text": None, "recorded": False}
    return {"text": row.get("text"), "recorded": True}


def _read_redact_event(detail: dict[str, Any]) -> dict[str, Any]:
    """Apply read-mode redaction to the judged text and the before/after diff."""
    from fastaiagent.trace.redaction import _read_redact

    judged = detail.get("judged") or {}
    if judged.get("text") is not None:
        judged["text"] = _read_redact({"t": judged["text"]}).get("t")
    trigger = detail.get("trigger") or {}
    if trigger.get("text") is not None:
        trigger["text"] = _read_redact({"t": trigger["text"]}).get("t")
    for event in (detail.get("event"), *detail.get("context", {}).get("sibling_events", [])):
        if not event:
            continue
        meta = event.get("metadata") or {}
        for key in ("before", "after"):
            if isinstance(meta.get(key), str):
                meta[key] = _read_redact({"t": meta[key]}).get("t")
    return detail


@router.get("/{event_id}")
def get_event_detail(
    request: Request,
    event_id: str,
    redact: bool = Query(
        False, description="Apply read-mode redaction (as /api/traces/{id}?redact=true)."
    ),
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Full detail for a single guardrail event.

    Joins the event row with the text the rule judged, its triggering span (if
    any), surrounding span context, and sibling events that ran on the same
    content. This is the data behind the three-panel detail view (trigger /
    rule / outcome) plus the conversation-context section below.

    ``judged`` (1.89.0) is exactly what the rule saw. ``trigger`` is kept for
    compatibility: it reads the parent span, which may hold the text after a
    rewrite, or nothing when the run was blocked.
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
        triggering_span = _load_triggering_span(db, event.span_id, ctx.project_id)
        detail = {
            "event": event.model_dump(),
            "judged": _judged(db, event.judged_ref),
            "trigger": _trigger_payload(triggering_span, event.position),
            "context": {
                "spans": _surrounding_spans(db, event.trace_id, ctx.project_id),
                "sibling_events": _sibling_events(
                    db,
                    event.event_id,
                    event.trace_id,
                    event.span_id,
                    ctx.project_id,
                    batch_id=event.batch_id,
                ),
            },
        }
        return _read_redact_event(detail) if redact else detail
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
        # Clearing the flag clears its timestamp (1.89.0): a cleared flag kept
        # the time it was cleared as "flagged at".
        flagged_at = datetime.now(tz=timezone.utc).isoformat() if body.false_positive else None
        db.execute(
            f"""UPDATE guardrail_events
                SET false_positive = ?, false_positive_at = ?
                WHERE event_id = ? {pid_clause}""",
            (1 if body.false_positive else 0, flagged_at, event_id, *pid_params),
        )
        return {
            "event_id": event_id,
            "false_positive": body.false_positive,
            "false_positive_at": flagged_at,
        }
    finally:
        db.close()
