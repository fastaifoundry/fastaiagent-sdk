"""Overview / home dashboard endpoint."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview")
def overview(
    request: Request, _user: str = Depends(require_session)
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        now = datetime.now(tz=timezone.utc)
        day_ago = (now - timedelta(hours=24)).isoformat()
        week_ago = (now - timedelta(days=7)).isoformat()

        total_day = db.fetchone(
            "SELECT COUNT(DISTINCT trace_id) AS n FROM spans WHERE start_time >= ?",
            (day_ago,),
        )
        failing_day = db.fetchone(
            """SELECT COUNT(DISTINCT trace_id) AS n FROM spans
               WHERE status != 'OK' AND start_time >= ?""",
            (day_ago,),
        )
        runs_week = db.fetchone(
            "SELECT COUNT(*) AS n FROM eval_runs WHERE started_at >= ?",
            (week_ago,),
        )
        avg_pass = db.fetchone(
            "SELECT AVG(pass_rate) AS pr FROM eval_runs WHERE started_at >= ?",
            (week_ago,),
        )
        recent_traces = db.fetchall(
            """SELECT trace_id, MIN(name) AS name, MIN(start_time) AS start_time,
                      MIN(status) AS status
               FROM spans
               GROUP BY trace_id
               ORDER BY start_time DESC
               LIMIT 5"""
        )
        recent_runs = db.fetchall(
            """SELECT run_id, run_name, dataset_name, pass_rate, started_at
               FROM eval_runs
               ORDER BY started_at DESC
               LIMIT 5"""
        )
        prompt_changes = db.fetchall(
            """SELECT slug, version, created_at
               FROM prompt_versions
               WHERE created_at >= ?
               ORDER BY created_at DESC
               LIMIT 10""",
            (week_ago,),
        )
        recent_errors = db.fetchall(
            """SELECT trace_id, name, start_time, status, attributes
               FROM spans
               WHERE status != 'OK' AND start_time >= ?
               ORDER BY start_time DESC
               LIMIT 10""",
            (day_ago,),
        )
        agents_with_errors: dict[str, int] = {}
        for row in recent_errors:
            attrs_raw = row.get("attributes") or "{}"
            try:
                attrs = json.loads(attrs_raw)
            except json.JSONDecodeError:
                attrs = {}
            name = attrs.get("fastai.agent.name")
            if name:
                agents_with_errors[name] = agents_with_errors.get(name, 0) + 1

        return {
            "traces_last_24h": total_day["n"] if total_day else 0,
            "failing_traces_last_24h": failing_day["n"] if failing_day else 0,
            "eval_runs_last_7d": runs_week["n"] if runs_week else 0,
            "avg_pass_rate_last_7d": (avg_pass["pr"] or 0.0) if avg_pass else 0.0,
            "recent_traces": recent_traces,
            "recent_eval_runs": recent_runs,
            "prompt_changes_last_7d": prompt_changes,
            "agents_with_errors": [
                {"agent_name": k, "error_count": v}
                for k, v in sorted(
                    agents_with_errors.items(), key=lambda kv: -kv[1]
                )
            ],
        }
    finally:
        db.close()
