"""Eval runs endpoints."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/evals", tags=["evals"])


def _unpack(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("scorers", "metadata", "per_scorer", "input", "expected_output", "actual_output"):
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out


@router.get("")
def list_runs(
    request: Request,
    _user: str = Depends(require_session),
    dataset: str | None = Query(default=None),
    agent: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if dataset:
            clauses.append("dataset_name = ?")
            params.append(dataset)
        if agent:
            clauses.append("agent_name = ?")
            params.append(agent)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total_row = db.fetchone(
            f"SELECT COUNT(*) AS n FROM eval_runs {where_sql}", tuple(params)
        )
        total = int((total_row or {}).get("n") or 0)
        rows = db.fetchall(
            f"""SELECT * FROM eval_runs {where_sql}
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


@router.get("/trend")
def trend(
    request: Request,
    dataset: str | None = Query(default=None),
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        clause = "WHERE dataset_name = ?" if dataset else ""
        rows = db.fetchall(
            f"""SELECT started_at, pass_rate, dataset_name
                FROM eval_runs
                {clause}
                ORDER BY started_at ASC""",
            (dataset,) if dataset else (),
        )
        return {"points": rows}
    finally:
        db.close()


@router.get("/compare")
def compare(
    request: Request,
    a: str,
    b: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        def get_run(run_id: str) -> dict[str, Any]:
            row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
            if row is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, f"Eval run '{run_id}' not found"
                )
            return _unpack(row)

        def get_cases(run_id: str) -> list[dict[str, Any]]:
            return [
                _unpack(r)
                for r in db.fetchall(
                    """SELECT * FROM eval_cases
                       WHERE run_id = ?
                       ORDER BY ordinal""",
                    (run_id,),
                )
            ]

        run_a = get_run(a)
        run_b = get_run(b)
        cases_a = get_cases(a)
        cases_b = get_cases(b)

        regressed: list[dict[str, Any]] = []
        improved: list[dict[str, Any]] = []
        for ca, cb in zip(cases_a, cases_b):
            a_ok = all(v.get("passed") for v in (ca.get("per_scorer") or {}).values())
            b_ok = all(v.get("passed") for v in (cb.get("per_scorer") or {}).values())
            if a_ok and not b_ok:
                regressed.append({"a": ca, "b": cb})
            elif b_ok and not a_ok:
                improved.append({"a": ca, "b": cb})
        return {
            "run_a": run_a,
            "run_b": run_b,
            "regressed": regressed,
            "improved": improved,
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
    try:
        run_row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        if run_row is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Eval run '{run_id}' not found"
            )
        cases = db.fetchall(
            """SELECT * FROM eval_cases
               WHERE run_id = ?
               ORDER BY ordinal""",
            (run_id,),
        )
        return {
            "run": _unpack(run_row),
            "cases": [_unpack(c) for c in cases],
        }
    finally:
        db.close()
