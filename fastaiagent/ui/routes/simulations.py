"""Agent-simulation endpoints.

Read-only surface over ``sim_runs`` + ``sim_cases``. Mirrors ``routes/evals.py``:
sync handlers, a per-request ``SQLiteHelper`` from the app context, project
scoping via ``project_filter`` / ``ctx.project_id``, and JSON unpacking of the
stored transcript + per-criterion verdicts so the SPA can render them directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from fastaiagent.ui.deps import get_context, project_filter, require_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/simulations", tags=["simulations"])


def _unpack(row: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON-encoded columns on a sim run / case row."""
    out = dict(row)
    for key in ("metadata", "criteria", "per_criterion", "transcript"):
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON for simulation field %r", key, exc_info=True)
    return out


@router.get("")
def list_runs(
    request: Request,
    _user: str = Depends(require_session),
    agent: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if agent:
            clauses.append("agent_name = ?")
            params.append(agent)
        if ctx.project_id:
            clauses.append("project_id = ?")
            params.append(ctx.project_id)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total_row = db.fetchone(
            f"SELECT COUNT(*) AS n FROM sim_runs {where_sql}", tuple(params)
        )
        total = int((total_row or {}).get("n") or 0)
        rows = db.fetchall(
            f"""SELECT * FROM sim_runs {where_sql}
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?""",
            tuple(params) + (page_size, (page - 1) * page_size),
        )
        return {
            "rows": [_unpack(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    finally:
        db.close()


@router.get("/{run_id}")
def get_run(
    request: Request,
    run_id: str,
    _user: str = Depends(require_session),
    outcome: str | None = Query(
        default=None,
        pattern="^(passed|failed)$",
        description="Filter scenarios by overall outcome.",
    ),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        run_row = db.fetchone(
            f"SELECT * FROM sim_runs WHERE run_id = ? {pid_clause}",
            (run_id, *pid_params),
        )
        if run_row is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Simulation run '{run_id}' not found"
            )
        case_rows = db.fetchall(
            f"""SELECT * FROM sim_cases
               WHERE run_id = ? {pid_clause}
               ORDER BY ordinal""",
            (run_id, *pid_params),
        )
        all_cases = [_unpack(c) for c in case_rows]

        def keep(case: dict[str, Any]) -> bool:
            if outcome:
                want = 1 if outcome == "passed" else 0
                if int(case.get("passed") or 0) != want:
                    return False
            return True

        filtered = [c for c in all_cases if keep(c)]
        return {
            "run": _unpack(run_row),
            "cases": filtered,
            "total_cases": len(all_cases),
        }
    finally:
        db.close()
