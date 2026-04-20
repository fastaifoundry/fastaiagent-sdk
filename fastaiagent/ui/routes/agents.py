"""Agent directory derived from span attributes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/agents", tags=["agents"])


def _aggregate(
    spans: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bucket root spans by agent name, computing per-agent stats."""
    by_agent: dict[str, dict[str, Any]] = {}
    for span in spans:
        try:
            attrs = json.loads(span.get("attributes") or "{}")
        except json.JSONDecodeError:
            attrs = {}
        name = attrs.get("fastai.agent.name")
        if not name:
            continue
        bucket = by_agent.setdefault(
            name,
            {
                "agent_name": name,
                "run_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_duration_ms": 0,
                "total_cost_usd": 0.0,
                "last_run": "",
            },
        )
        bucket["run_count"] += 1
        if span.get("status") == "OK":
            bucket["success_count"] += 1
        else:
            bucket["error_count"] += 1
        start = span.get("start_time") or ""
        end = span.get("end_time") or ""
        try:
            from datetime import datetime

            a = datetime.fromisoformat(start)
            b = datetime.fromisoformat(end)
            bucket["total_duration_ms"] += int((b - a).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass
        bucket["total_cost_usd"] += float(attrs.get("fastai.cost.total_usd") or 0.0)
        if start > bucket["last_run"]:
            bucket["last_run"] = start
    return by_agent


def _format(bucket: dict[str, Any]) -> dict[str, Any]:
    run_count = bucket["run_count"] or 1
    return {
        "agent_name": bucket["agent_name"],
        "run_count": bucket["run_count"],
        "success_rate": bucket["success_count"] / run_count,
        "error_count": bucket["error_count"],
        "avg_latency_ms": bucket["total_duration_ms"] / run_count,
        "avg_cost_usd": bucket["total_cost_usd"] / run_count,
        "last_run": bucket["last_run"],
    }


@router.get("")
def list_agents(
    request: Request, _user: str = Depends(require_session)
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        rows = db.fetchall(
            "SELECT * FROM spans WHERE parent_span_id IS NULL OR parent_span_id = ''"
        )
        by_agent = _aggregate(rows)
        return {"agents": [_format(b) for b in by_agent.values()]}
    finally:
        db.close()


@router.get("/{name}")
def get_agent(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        rows = db.fetchall(
            "SELECT * FROM spans WHERE parent_span_id IS NULL OR parent_span_id = ''"
        )
        by_agent = _aggregate(rows)
        if name not in by_agent:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Agent '{name}' not found"
            )
        return _format(by_agent[name])
    finally:
        db.close()
