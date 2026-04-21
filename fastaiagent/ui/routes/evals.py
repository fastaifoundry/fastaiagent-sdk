"""Eval runs endpoints.

Read-only surface over ``eval_runs`` + ``eval_cases``. Each eval run also
drills into the ``spans`` table (via ``eval_cases.trace_id``) to aggregate
cost and latency, so the UI can show cost-per-run + avg case latency
without any extra bookkeeping at write time.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from fastaiagent.ui.deps import get_context, require_session

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
                pass
    return out


def _trace_cost_and_latency(
    db: Any, trace_ids: list[str]
) -> tuple[float, float]:
    """Return (total_cost_usd, avg_latency_ms) for the given trace_ids.

    Scans root spans only (parent_span_id IS NULL) since those carry the
    total cost/latency of the run. Traces without a recorded cost are
    excluded from the average.
    """
    if not trace_ids:
        return 0.0, 0.0
    from datetime import datetime

    from fastaiagent.ui.attrs import trace_cost_usd
    from fastaiagent.ui.pricing import compute_cost_usd

    placeholders = ",".join("?" * len(trace_ids))
    rows = db.fetchall(
        f"""SELECT trace_id, attributes, start_time, end_time
            FROM spans
            WHERE parent_span_id IS NULL AND trace_id IN ({placeholders})""",
        tuple(trace_ids),
    )
    total_cost = 0.0
    total_latency_ms = 0.0
    latency_count = 0
    for row in rows:
        try:
            attrs = json.loads(row["attributes"] or "{}")
        except json.JSONDecodeError:
            attrs = {}
        reported = trace_cost_usd(attrs)
        if reported is None:
            reported = compute_cost_usd(
                attrs.get("gen_ai.request.model"),
                attrs.get("gen_ai.usage.input_tokens"),
                attrs.get("gen_ai.usage.output_tokens"),
            )
        if reported is not None:
            total_cost += reported
        try:
            a = datetime.fromisoformat(row["start_time"])
            b = datetime.fromisoformat(row["end_time"])
            total_latency_ms += (b - a).total_seconds() * 1000
            latency_count += 1
        except (ValueError, TypeError):
            pass
    avg_latency = total_latency_ms / latency_count if latency_count else 0.0
    return total_cost, avg_latency


def _run_with_aggregates(
    db: Any, run: dict[str, Any]
) -> dict[str, Any]:
    """Decorate a run row with cost_usd + avg_latency_ms derived from its cases."""
    case_rows = db.fetchall(
        "SELECT trace_id FROM eval_cases WHERE run_id = ? AND trace_id IS NOT NULL",
        (run["run_id"],),
    )
    trace_ids = [r["trace_id"] for r in case_rows if r["trace_id"]]
    cost, latency = _trace_cost_and_latency(db, trace_ids)
    out = dict(run)
    out["cost_usd"] = round(cost, 6)
    out["avg_latency_ms"] = round(latency, 2)
    out["case_count"] = (out.get("pass_count") or 0) + (out.get("fail_count") or 0)
    return out


def _scorer_summary(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-scorer {pass, fail} counts across a run's cases."""
    summary: dict[str, dict[str, int]] = {}
    for case in cases:
        per = case.get("per_scorer") or {}
        if not isinstance(per, dict):
            continue
        for scorer, result in per.items():
            bucket = summary.setdefault(scorer, {"pass": 0, "fail": 0})
            if isinstance(result, dict) and result.get("passed"):
                bucket["pass"] += 1
            else:
                bucket["fail"] += 1
    return summary


def _case_outcome(case: dict[str, Any]) -> str:
    """Return 'passed' if every scorer passed, else 'failed'."""
    per = case.get("per_scorer") or {}
    if not isinstance(per, dict) or not per:
        return "passed"
    return (
        "passed"
        if all(
            isinstance(v, dict) and v.get("passed") for v in per.values()
        )
        else "failed"
    )


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
        enriched = [_run_with_aggregates(db, _unpack(r)) for r in rows]
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
    """Compare two eval runs case-by-case.

    Cases are matched by ``ordinal`` first and by ``input`` as a fallback,
    so runs that share a dataset but reordered cases still align. Each
    regressed / improved entry includes a ``scorer_deltas`` list so the UI
    can highlight *which scorer* changed on that case.
    """
    ctx = get_context(request)
    db = ctx.db()
    try:
        def get_run(run_id: str) -> dict[str, Any]:
            row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
            if row is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, f"Eval run '{run_id}' not found"
                )
            return _run_with_aggregates(db, _unpack(row))

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

        # Match by ordinal first; fall back to input equality if the cases
        # have been reordered between runs.
        index_b: dict[Any, dict[str, Any]] = {}
        for c in cases_b:
            key = c.get("ordinal")
            if key is not None:
                index_b[key] = c
        by_input: dict[str, dict[str, Any]] = {}
        for c in cases_b:
            try:
                by_input[json.dumps(c.get("input"), sort_keys=True)] = c
            except (TypeError, ValueError):
                pass

        regressed: list[dict[str, Any]] = []
        improved: list[dict[str, Any]] = []
        unchanged_pass = 0
        unchanged_fail = 0

        for ca in cases_a:
            cb = index_b.get(ca.get("ordinal"))
            if cb is None:
                try:
                    cb = by_input.get(json.dumps(ca.get("input"), sort_keys=True))
                except (TypeError, ValueError):
                    cb = None
            if cb is None:
                continue
            a_ok = _case_outcome(ca) == "passed"
            b_ok = _case_outcome(cb) == "passed"
            deltas = _scorer_deltas(ca, cb)
            entry = {"a": ca, "b": cb, "scorer_deltas": deltas}
            if a_ok and not b_ok:
                regressed.append(entry)
            elif b_ok and not a_ok:
                improved.append(entry)
            elif a_ok and b_ok:
                unchanged_pass += 1
            else:
                unchanged_fail += 1

        return {
            "run_a": run_a,
            "run_b": run_b,
            "regressed": regressed,
            "improved": improved,
            "unchanged_pass": unchanged_pass,
            "unchanged_fail": unchanged_fail,
            "pass_rate_delta": round(
                (run_b.get("pass_rate") or 0.0) - (run_a.get("pass_rate") or 0.0),
                4,
            ),
            "cost_delta_usd": round(
                (run_b.get("cost_usd") or 0.0) - (run_a.get("cost_usd") or 0.0),
                6,
            ),
        }
    finally:
        db.close()


def _scorer_deltas(
    a: dict[str, Any], b: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-scorer {passed_before, passed_after, changed} list for one case pair."""
    per_a = a.get("per_scorer") or {}
    per_b = b.get("per_scorer") or {}
    if not isinstance(per_a, dict) or not isinstance(per_b, dict):
        return []
    out = []
    for scorer in sorted(set(per_a) | set(per_b)):
        ra = per_a.get(scorer) or {}
        rb = per_b.get(scorer) or {}
        pa = bool(ra.get("passed")) if isinstance(ra, dict) else False
        pb = bool(rb.get("passed")) if isinstance(rb, dict) else False
        out.append(
            {
                "scorer": scorer,
                "passed_before": pa,
                "passed_after": pb,
                "changed": pa != pb,
            }
        )
    return out


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
        pattern="^(passed|failed)$",
        description="Filter cases by overall outcome.",
    ),
    q: str | None = Query(
        default=None,
        description="Substring match across input/expected/actual.",
    ),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        run_row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        if run_row is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, f"Eval run '{run_id}' not found"
            )
        case_rows = db.fetchall(
            """SELECT * FROM eval_cases
               WHERE run_id = ?
               ORDER BY ordinal""",
            (run_id,),
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
        enriched = _run_with_aggregates(db, _unpack(run_row))
        enriched["scorer_summary"] = summary
        return {
            "run": enriched,
            "cases": filtered,
            "total_cases": len(all_cases),
        }
    finally:
        db.close()
