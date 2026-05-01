"""Guardrail events list endpoint (read-only)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from fastaiagent.ui.deps import get_context, require_session

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


@router.get("")
def list_events(
    request: Request,
    _user: str = Depends(require_session),
    rule: str | None = Query(default=None),
    outcome: str | None = Query(default=None),
    agent: str | None = Query(default=None),
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

        events = [
            GuardrailEvent(
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
            ).model_dump()
            for r in rows
        ]
        return {"rows": events, "total": total, "page": page, "page_size": page_size}
    finally:
        db.close()
