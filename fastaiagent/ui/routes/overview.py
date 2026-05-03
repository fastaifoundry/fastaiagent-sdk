"""Overview / home dashboard endpoint."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request

from fastaiagent.ui.deps import get_context, project_filter, require_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview")
def overview(request: Request, _user: str = Depends(require_session)) -> dict[str, Any]:
    ctx = get_context(request)
    pid_clause, pid_params = project_filter(ctx)
    db = ctx.db()
    try:
        now = datetime.now(tz=timezone.utc)
        day_ago = (now - timedelta(hours=24)).isoformat()
        week_ago = (now - timedelta(days=7)).isoformat()

        total_day = db.fetchone(
            f"SELECT COUNT(DISTINCT trace_id) AS n FROM spans "
            f"WHERE start_time >= ? {pid_clause}",
            (day_ago, *pid_params),
        )
        failing_day = db.fetchone(
            f"""SELECT COUNT(DISTINCT trace_id) AS n FROM spans
               WHERE status != 'OK' AND start_time >= ? {pid_clause}""",
            (day_ago, *pid_params),
        )
        runs_week = db.fetchone(
            f"SELECT COUNT(*) AS n FROM eval_runs WHERE started_at >= ? {pid_clause}",
            (week_ago, *pid_params),
        )
        avg_pass = db.fetchone(
            f"SELECT AVG(pass_rate) AS pr FROM eval_runs WHERE started_at >= ? {pid_clause}",
            (week_ago, *pid_params),
        )
        # GROUP BY trace_id queries can't take a WHERE on project_id without
        # a clause — inject it via the same helper at the outermost WHERE.
        where_for_recent = (
            "WHERE project_id = ?" if ctx.project_id else ""
        )
        recent_traces = db.fetchall(
            f"""SELECT trace_id, MIN(name) AS name, MIN(start_time) AS start_time,
                      MIN(status) AS status
               FROM spans
               {where_for_recent}
               GROUP BY trace_id
               ORDER BY start_time DESC
               LIMIT 5""",
            (ctx.project_id,) if ctx.project_id else (),
        )
        recent_runs = db.fetchall(
            f"""SELECT run_id, run_name, dataset_name, pass_rate, started_at
               FROM eval_runs
               {where_for_recent}
               ORDER BY started_at DESC
               LIMIT 5""",
            (ctx.project_id,) if ctx.project_id else (),
        )
        prompt_changes = db.fetchall(
            f"""SELECT slug, version, created_at
               FROM prompt_versions
               WHERE created_at >= ? {pid_clause}
               ORDER BY created_at DESC
               LIMIT 10""",
            (week_ago, *pid_params),
        )
        recent_errors = db.fetchall(
            f"""SELECT trace_id, name, start_time, status, attributes
               FROM spans
               WHERE status != 'OK' AND start_time >= ? {pid_clause}
               ORDER BY start_time DESC
               LIMIT 10""",
            (day_ago, *pid_params),
        )
        from fastaiagent.ui.attrs import attr

        agents_with_errors: dict[str, int] = {}
        for row in recent_errors:
            attrs_raw = row.get("attributes") or "{}"
            try:
                attrs = json.loads(attrs_raw)
            except json.JSONDecodeError:
                attrs = {}
            name = attr(attrs, "agent.name")
            if name:
                agents_with_errors[name] = agents_with_errors.get(name, 0) + 1

        # Phase 10 — durability KPIs for the Home page.
        pending_approvals = db.fetchone(
            f"SELECT COUNT(*) AS n FROM pending_interrupts {where_for_recent}",
            (ctx.project_id,) if ctx.project_id else (),
        )
        failed_executions = db.fetchone(
            f"""SELECT COUNT(DISTINCT execution_id) AS n FROM checkpoints
               WHERE status IN ('failed', 'interrupted') {pid_clause}""",
            tuple(pid_params),
        )

        return {
            "traces_last_24h": total_day["n"] if total_day else 0,
            "failing_traces_last_24h": failing_day["n"] if failing_day else 0,
            "eval_runs_last_7d": runs_week["n"] if runs_week else 0,
            "avg_pass_rate_last_7d": (avg_pass["pr"] or 0.0) if avg_pass else 0.0,
            "pending_approvals_count": (pending_approvals["n"] if pending_approvals else 0),
            "failed_executions_count": (failed_executions["n"] if failed_executions else 0),
            "recent_traces": recent_traces,
            "recent_eval_runs": recent_runs,
            "prompt_changes_last_7d": prompt_changes,
            "agents_with_errors": [
                {"agent_name": k, "error_count": v}
                for k, v in sorted(agents_with_errors.items(), key=lambda kv: -kv[1])
            ],
        }
    finally:
        db.close()
