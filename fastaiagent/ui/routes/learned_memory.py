"""Read-only browse endpoint for the ``learned_memory`` table.

The Trace Learning Loop (``fastaiagent learn``) populates this table
offline; this endpoint lets the local UI inspect and audit the facts
that will be re-injected into future runs by ``PersistentFactBlock``.

Read-only by design — we never mutate facts from the UI in v1. Manual
conflict resolution goes through the CLI (``fastaiagent learn supersede
<old_id> <new_id>``) so the action is captured in shell history rather
than buried in the UI.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from fastaiagent.ui.deps import get_context, project_filter, require_session

router = APIRouter(prefix="/api/learned_memory", tags=["learned_memory"])


@router.get("")
def list_learned_memory(
    request: Request,
    scope: str | None = Query(
        None,
        description="Optional filter: 'user' | 'project' | 'agent'.",
    ),
    scope_id: str | None = Query(None, description="Optional scope-id filter."),
    include_superseded: bool = Query(
        False, description="Include rows that have been replaced by newer facts."
    ),
    limit: int = Query(200, ge=1, le=2000),
    redact: bool = Query(
        False,
        description=(
            "Mask fact text using the installed read-mode RedactionPolicy. "
            "No-op when no policy is installed (mirrors the trace endpoints)."
        ),
    ),
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Return rows from ``learned_memory`` filtered by the active project.

    Active facts (``superseded_by IS NULL``) come first by default; with
    ``include_superseded=true`` the audit history is included with a
    ``superseded_by`` column populated.
    """
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)

    where = ["1=1"]
    params: list = []

    where.append(pid_clause if pid_clause else "1=1")
    params.extend(pid_params)

    if not include_superseded:
        where.append("superseded_by IS NULL")
    if scope:
        if scope not in ("user", "project", "agent"):
            return {"rows": [], "total": 0, "error": f"invalid scope: {scope!r}"}
        where.append("scope = ?")
        params.append(scope)
    if scope_id:
        where.append("scope_id = ?")
        params.append(scope_id)

    sql = (
        "SELECT id, scope, scope_id, fact, source_trace_id, confidence, "
        "       created_at, superseded_by, project_id "
        "FROM learned_memory "
        f"WHERE {' AND '.join(where)} "
        "ORDER BY created_at DESC "
        f"LIMIT {int(limit)}"
    )
    try:
        rows = db.fetchall(sql, tuple(params))
        total_row = db.fetchone(
            f"SELECT COUNT(*) AS n FROM learned_memory WHERE {' AND '.join(where)}",
            tuple(params),
        )
        total = int(total_row["n"]) if total_row else 0
    finally:
        db.close()

    out_rows = [dict(r) for r in rows]
    if redact:
        # Read-mode masking: only fires when a RedactionPolicy(mode in
        # {"read","both"}) is installed; otherwise leaves facts unchanged.
        from fastaiagent.trace.redaction import get_redaction_policy

        policy = get_redaction_policy()
        if policy is not None and policy.mode in ("read", "both"):
            for row in out_rows:
                if isinstance(row.get("fact"), str):
                    row["fact"] = policy.redact_string(row["fact"])

    return {
        "rows": out_rows,
        "total": total,
        "filters": {
            "scope": scope,
            "scope_id": scope_id,
            "include_superseded": include_superseded,
        },
    }


@router.get("/scopes")
def list_scopes(
    request: Request,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Distinct (scope, scope_id) pairs in the active project — for UI filter chips."""
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    sql = (
        "SELECT scope, scope_id, COUNT(*) AS n FROM learned_memory "
        f"WHERE superseded_by IS NULL {('AND ' + pid_clause) if pid_clause else ''} "
        "GROUP BY scope, scope_id ORDER BY n DESC"
    )
    try:
        rows = db.fetchall(sql, tuple(pid_params))
    finally:
        db.close()
    return {"scopes": [dict(r) for r in rows]}
