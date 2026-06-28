"""Optimize-run endpoints.

Read-only surface over ``optimize_runs`` + ``optimize_iterations``. Mirrors
``routes/simulations.py`` / ``routes/evals.py``: sync handlers, a per-request
``SQLiteHelper`` from the app context, project scoping via ``project_filter`` /
``ctx.project_id``, and JSON unpacking of the stored ``levers`` / ``config`` /
``best_candidate`` columns so the SPA can render them directly.

Each iteration row carries an ``eval_run_id`` linking into the ``eval_runs`` row
that candidate already produced via ``aevaluate(persist=)`` — the UI drills from a
trajectory row straight into the existing eval data (no duplicate storage).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from fastaiagent.ui.deps import get_context, project_filter, require_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimizes", tags=["optimizes"])

_JSON_COLUMNS = ("metadata", "levers", "config", "best_candidate")


def _unpack(row: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON-encoded columns on an optimize run row."""
    out = dict(row)
    for key in _JSON_COLUMNS:
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON for optimize field %r", key, exc_info=True)
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
            f"SELECT COUNT(*) AS n FROM optimize_runs {where_sql}", tuple(params)
        )
        total = int((total_row or {}).get("n") or 0)
        rows = db.fetchall(
            f"""SELECT * FROM optimize_runs {where_sql}
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
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        run_row = db.fetchone(
            f"SELECT * FROM optimize_runs WHERE run_id = ? {pid_clause}",
            (run_id, *pid_params),
        )
        if run_row is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Optimize run '{run_id}' not found"
            )
        iter_rows = db.fetchall(
            f"""SELECT * FROM optimize_iterations
               WHERE run_id = ? {pid_clause}
               ORDER BY ordinal""",
            (run_id, *pid_params),
        )
        return {
            "run": _unpack(run_row),
            "iterations": [dict(r) for r in iter_rows],
            "total_iterations": len(iter_rows),
        }
    finally:
        db.close()
