"""Eval runs endpoints.

Read-only surface over ``eval_runs`` + ``eval_cases``. Each eval run also
drills into the ``spans`` table (via ``eval_cases.trace_id``) to aggregate
cost and latency, so the UI can show cost-per-run + avg case latency
without any extra bookkeeping at write time.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from fastaiagent.ui.deps import get_context, project_filter, require_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evals", tags=["evals"])


def _unpack(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "scorers",
        "metadata",
        "per_scorer",
        "input",
        "expected_output",
        "actual_output",
    ):
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                logger.debug("Failed to parse JSON for eval field %r", key, exc_info=True)
    return out


def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    """``b[key] - a[key]``; ``None`` when either side is unknown."""
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        return None
    return round(float(vb) - float(va), 8)


def _gate_of(run: dict[str, Any]) -> dict[str, Any] | None:
    """The Agent-CI evidence recorded on a run (E10, 1.89.0), or ``None``.

    Read from ``metadata``: the outcome and thresholds the gate demanded, the
    per-threshold checks with their actual values, the error-rate bound, the
    baseline comparison and git provenance. Every part is optional.
    """
    meta = run.get("metadata")
    if not isinstance(meta, dict) or "gate_outcome" not in meta:
        return None
    return {
        "outcome": meta.get("gate_outcome"),
        "thresholds": meta.get("thresholds") or {},
        "checks": meta.get("gate_checks") or [],
        "error": meta.get("gate_error"),
        "invalid_reasons": meta.get("gate_invalid_reasons") or [],
        "baseline": (
            {**(meta.get("baseline") or {}), **(meta.get("gate_baseline") or {})}
            if meta.get("baseline")
            else None
        ),
        "git_sha": meta.get("git_sha"),
        "git_branch": meta.get("git_branch"),
    }


def _run_with_aggregates(
    db: Any, run: dict[str, Any], *, project_id: str = ""
) -> dict[str, Any]:
    """Decorate a run row with cost_usd + avg_latency_ms derived from its cases."""
    from fastaiagent.ui.trace_cost import sum_rollups, trace_rollups

    pid_extra = "AND project_id = ?" if project_id else ""
    pid_params: tuple = (project_id,) if project_id else ()
    case_rows = db.fetchall(
        f"SELECT trace_id, scoring_trace_id FROM eval_cases WHERE run_id = ? {pid_extra}",
        (run["run_id"], *pid_params),
    )
    agent_ids = [r["trace_id"] for r in case_rows if r["trace_id"]]
    scoring_ids = [r["scoring_trace_id"] for r in case_rows if r["scoring_trace_id"]]
    # 1.89.0: cost lives on the child llm.* spans; reading the root span only
    # made every run's cost 0 ("—"). The judges' spend is the eval cost, kept
    # apart from the agent's.
    rollups = trace_rollups(db, [*agent_ids, *scoring_ids], project_id=project_id)
    agent_cost, agent_known, latency = sum_rollups(rollups, agent_ids)
    eval_cost, eval_known, _ = sum_rollups(rollups, scoring_ids)
    out = dict(run)
    out["agent_cost_usd"] = agent_cost
    out["agent_cost_known"] = agent_known
    out["eval_cost_usd"] = eval_cost
    out["eval_cost_known"] = eval_known
    out["cost_usd"] = round(agent_cost, 6) if agent_cost is not None else 0.0
    out["avg_latency_ms"] = round(latency, 2) if latency is not None else 0.0
    out["traced_case_count"] = len(agent_ids)
    if out.get("case_count") is None:
        # A row written by an SDK older than 1.89.0 into this database has no
        # case columns: count its stored cases. ``case_count`` used to be
        # pass_count + fail_count, which counted verdicts, not cases.
        out.update(_case_columns(db, run["run_id"], project_id=project_id))
    return out


def _case_columns(db: Any, run_id: str, *, project_id: str = "") -> dict[str, Any]:
    from fastaiagent.eval.compare import summarize_cases

    pid_extra = "AND project_id = ?" if project_id else ""
    rows = db.fetchall(
        f"SELECT error, per_scorer FROM eval_cases WHERE run_id = ? {pid_extra}",
        (run_id, *((project_id,) if project_id else ())),
    )
    cases = []
    for r in rows:
        try:
            per = json.loads(r.get("per_scorer") or "{}")
        except (ValueError, TypeError):
            per = {}
        cases.append({"error": r.get("error"), "per_scorer": per})
    s = summarize_cases(cases)
    return {k: v for k, v in s.items() if k.startswith("case_")}


def _scorer_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-scorer {pass, fail, errored, skipped} counts across a run's cases.

    ``errored`` and ``skipped`` (1.89.0) were counted as fails.
    """
    from fastaiagent.eval.scorer import verdict_outcome

    keys = {"passed": "pass", "failed": "fail", "errored": "errored", "skipped": "skipped"}
    summary: dict[str, dict[str, int]] = {}
    for case in cases:
        per = case.get("per_scorer") or {}
        if not isinstance(per, dict):
            continue
        for scorer, result in per.items():
            bucket = summary.setdefault(scorer, {"pass": 0, "fail": 0, "errored": 0, "skipped": 0})
            bucket[keys[verdict_outcome(result)]] += 1
    return summary


def _case_outcome(case: dict[str, Any]) -> str:
    """'errored' for infra-failed (unscored) cases, else 'passed'/'failed'.

    Before the ``error`` column existed (schema v16) an errored case rendered
    as "passed" — an outage could look green. Logic lives in
    ``fastaiagent.eval.compare.case_outcome`` (shared with pytest/CLI gating).
    """
    from fastaiagent.eval.compare import case_outcome

    return case_outcome(case)


@router.get("")
def list_runs(
    request: Request,
    _user: str = Depends(require_session),
    dataset: str | None = Query(default=None),
    agent: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    kind: str | None = Query(
        default=None,
        description=(
            "Comma-separated run kinds to keep: evaluate, cli, pytest, optimize, ui, "
            "ui-echo. ``evaluate`` also matches runs recorded before kinds existed."
        ),
    ),
    exclude_kind: str | None = Query(default=None, description="Comma-separated kinds to drop."),
    q: str | None = Query(default=None, description="Substring of run, dataset or agent name."),
    gate: str | None = Query(
        default=None,
        pattern="^(passed|failed|invalid|none)(,(passed|failed|invalid|none))*$",
        description="Comma-separated recorded gate outcomes; ``none`` = never gated.",
    ),
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
        for value, negate in ((kind, False), (exclude_kind, True)):
            kinds = [k.strip() for k in (value or "").split(",") if k.strip()]
            if kinds:
                cond = f"kind IN ({','.join('?' * len(kinds))})"
                if "evaluate" in kinds:
                    cond = f"({cond} OR kind IS NULL)"
                clauses.append(f"NOT {cond}" if negate else cond)
                params.extend(kinds)
        if q:
            clauses.append("(run_name LIKE ? OR dataset_name LIKE ? OR agent_name LIKE ?)")
            params.extend([f"%{q}%"] * 3)
        if gate:
            outcome_sql = (
                "CASE WHEN json_valid(metadata) "
                "THEN json_extract(metadata, '$.gate_outcome') END"
            )
            wanted = [g for g in gate.split(",") if g]
            named = [g for g in wanted if g != "none"]
            parts = [f"{outcome_sql} IS NULL"] if "none" in wanted else []
            if named:
                parts.append(f"{outcome_sql} IN ({','.join('?' * len(named))})")
                params.extend(named)
            clauses.append(f"({' OR '.join(parts)})")
        if ctx.project_id:
            clauses.append("project_id = ?")
            params.append(ctx.project_id)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total_row = db.fetchone(f"SELECT COUNT(*) AS n FROM eval_runs {where_sql}", tuple(params))
        total = int((total_row or {}).get("n") or 0)
        rows = db.fetchall(
            f"""SELECT * FROM eval_runs {where_sql}
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?""",
            tuple(params) + (page_size, (page - 1) * page_size),
        )
        enriched = [
            _run_with_aggregates(db, _unpack(r), project_id=ctx.project_id)
            for r in rows
        ]
        return {
            "rows": enriched,
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
    include: str | None = Query(
        default=None,
        pattern="^all$",
        description="``all`` keeps dataset checks (ui-echo) and AutoLLM candidate runs.",
    ),
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Quality over time, one series per dataset (1.89.0).

    Each point carries the **cases** rate (``case_pass_rate``) first and the
    checks rate (``pass_rate``) second. A dataset check (``ui-echo``, an echo
    agent) and an AutoLLM candidate run are not quality measurements of the
    agent, so they are left out unless ``include=all``. The series key is the
    dataset, or the run name for a pytest session.
    """
    ctx = get_context(request)
    db = ctx.db()
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if dataset:
            clauses.append("dataset_name = ?")
            params.append(dataset)
        if include != "all":
            clauses.append("(kind IS NULL OR kind NOT IN ('ui-echo', 'optimize'))")
        if ctx.project_id:
            clauses.append("project_id = ?")
            params.append(ctx.project_id)
        clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = db.fetchall(
            f"""SELECT * FROM eval_runs {clause} ORDER BY started_at ASC""",
            tuple(params),
        )
        points = []
        for r in rows:
            run = _unpack(r)
            if run.get("case_count") is None:
                run.update(_case_columns(db, run["run_id"], project_id=ctx.project_id))
            raw_meta = run.get("metadata")
            meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            points.append(
                {
                    "started_at": run.get("started_at"),
                    "run_id": run.get("run_id"),
                    "run_name": run.get("run_name"),
                    "kind": run.get("kind"),
                    "dataset_name": run.get("dataset_name"),
                    "series": (
                        run.get("run_name")
                        if run.get("kind") == "pytest"
                        else run.get("dataset_name") or run.get("run_name")
                    ),
                    "case_pass_rate": run.get("case_pass_rate"),
                    "pass_rate": run.get("pass_rate"),
                    "gate_outcome": meta.get("gate_outcome"),
                }
            )
        return {"points": points}
    finally:
        db.close()


@router.get("/compare")
def compare(
    request: Request,
    a: str,
    b: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Compare two eval runs case-by-case.

    Cases are matched by ``ordinal`` when both runs hold the same input there
    and by ``input`` otherwise, so runs that share a dataset but reordered
    cases still align. Each
    regressed / improved entry includes a ``scorer_deltas`` list so the UI
    can highlight *which scorer* changed on that case.
    """
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:

        def get_run(run_id: str) -> dict[str, Any]:
            row = db.fetchone(
                f"SELECT * FROM eval_runs WHERE run_id = ? {pid_clause}",
                (run_id, *pid_params),
            )
            if row is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, f"Eval run '{run_id}' not found")
            return _run_with_aggregates(db, _unpack(row), project_id=ctx.project_id)

        def get_cases(run_id: str) -> list[dict[str, Any]]:
            return [
                _unpack(r)
                for r in db.fetchall(
                    f"""SELECT * FROM eval_cases
                       WHERE run_id = ? {pid_clause}
                       ORDER BY ordinal""",
                    (run_id, *pid_params),
                )
            ]

        from fastaiagent.eval.compare import compare_runs

        run_a = get_run(a)
        run_b = get_run(b)

        # Matching (ordinal-first, input fallback) + bucketing live in
        # fastaiagent.eval.compare, shared with the pytest plugin and CLI.
        cmp = compare_runs(
            {"run": run_a, "cases": get_cases(a)}, {"run": run_b, "cases": get_cases(b)}
        )

        return {
            "run_a": run_a,
            "run_b": run_b,
            "regressed": cmp.regressed,
            "improved": cmp.improved,
            "unchanged_pass": cmp.unchanged_pass,
            "unchanged_fail": cmp.unchanged_fail,
            # 1.89.0: what used to be dropped silently.
            "errored": cmp.errored,
            "only_in_a": cmp.only_in_a,
            "only_in_b": cmp.only_in_b,
            "drifted": cmp.drifted,
            "case_pass_rate_delta": cmp.case_pass_rate_delta,
            "pass_rate_comparable": cmp.pass_rate_comparable,
            # None when either run has no checks rate (nothing applied): the
            # library keeps 0.0 for its float callers, the UI must not read
            # that as "no change" (1.89.0).
            "pass_rate_delta": (
                cmp.pass_rate_delta
                if run_a.get("pass_rate") is not None and run_b.get("pass_rate") is not None
                else None
            ),
            "cost_delta_usd": round(
                (run_b.get("cost_usd") or 0.0) - (run_a.get("cost_usd") or 0.0),
                6,
            ),
            "agent_cost_delta_usd": _delta(run_a, run_b, "agent_cost_usd"),
            "eval_cost_delta_usd": _delta(run_a, run_b, "eval_cost_usd"),
        }
    finally:
        db.close()


@router.get("/{run_id}/cases/{case_id}/media/{idx}")
def get_case_media(
    request: Request,
    run_id: str,
    case_id: str,
    idx: int,
    _user: str = Depends(require_session),
) -> Any:
    """A case image's or PDF's thumbnail (JPEG), from ``eval_case_media``.

    Only the thumbnail is stored (1.89.0); the case itself holds a descriptor
    (type, media type, size, sha256), never the original bytes.
    """
    from fastapi.responses import Response

    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        owner = db.fetchone(
            f"SELECT case_id FROM eval_cases WHERE case_id = ? AND run_id = ? {pid_clause}",
            (case_id, run_id, *pid_params),
        )
        row = (
            db.fetchone(
                "SELECT thumbnail FROM eval_case_media WHERE case_id = ? AND idx = ?",
                (case_id, idx),
            )
            if owner
            else None
        )
        if row is None or not row.get("thumbnail"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No thumbnail for this case item")
        return Response(content=row["thumbnail"], media_type="image/jpeg")
    finally:
        db.close()


@router.get("/{run_id}")
def get_run(
    request: Request,
    run_id: str,
    _user: str = Depends(require_session),
    scorer: str | None = Query(
        default=None,
        description="Only include cases that have this scorer.",
    ),
    outcome: str | None = Query(
        default=None,
        pattern="^(passed|failed|errored)$",
        description="Filter cases by overall outcome.",
    ),
    q: str | None = Query(
        default=None,
        description="Substring match across input/expected/actual.",
    ),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    pid_clause, pid_params = project_filter(ctx)
    try:
        run_row = db.fetchone(
            f"SELECT * FROM eval_runs WHERE run_id = ? {pid_clause}",
            (run_id, *pid_params),
        )
        if run_row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Eval run '{run_id}' not found")
        case_rows = db.fetchall(
            f"""SELECT * FROM eval_cases
               WHERE run_id = ? {pid_clause}
               ORDER BY ordinal""",
            (run_id, *pid_params),
        )
        all_cases = [_unpack(c) for c in case_rows]
        # Scorer-summary is over ALL cases, not the filtered subset, so the
        # header stat doesn't shift as the user filters.
        summary = _scorer_summary(all_cases)

        def keep(case: dict[str, Any]) -> bool:
            if scorer and scorer not in (case.get("per_scorer") or {}):
                return False
            if outcome and _case_outcome(case) != outcome:
                return False
            if q:
                needle = q.lower()
                hay = (
                    json.dumps(case.get("input") or "")
                    + json.dumps(case.get("expected_output") or "")
                    + json.dumps(case.get("actual_output") or "")
                ).lower()
                if needle not in hay:
                    return False
            return True

        filtered = [c for c in all_cases if keep(c)]
        enriched = _run_with_aggregates(db, _unpack(run_row), project_id=ctx.project_id)
        enriched["scorer_summary"] = summary
        enriched["gate"] = _gate_of(enriched)
        return {
            "run": enriched,
            "cases": filtered,
            "total_cases": len(all_cases),
        }
    finally:
        db.close()
