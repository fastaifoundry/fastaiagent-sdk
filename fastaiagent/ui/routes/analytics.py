"""Analytics endpoint — latency percentiles, cost over time, error rate, top-N agents.

All aggregations are computed in Python over the spans table. SQLite lacks
``PERCENTILE_CONT``; the dataset is small enough (local, single-user) that
pulling the needed columns into Python and sorting is fine.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from fastaiagent.ui.attrs import attr, trace_cost_usd
from fastaiagent.ui.deps import get_context, require_session
from fastaiagent.ui.pricing import compute_cost_usd

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_PERIOD_HOURS = {"1d": 24, "7d": 168, "30d": 720, "all": 24 * 90}


def _period_to_hours(period: str) -> int:
    return _PERIOD_HOURS.get(period, 168)


def _span_cost(attrs: dict[str, Any]) -> float:
    reported = trace_cost_usd(attrs)
    if reported is not None:
        return reported
    return (
        compute_cost_usd(
            attrs.get("gen_ai.request.model"),
            attrs.get("gen_ai.usage.input_tokens"),
            attrs.get("gen_ai.usage.output_tokens"),
        )
        or 0.0
    )


def _bucket_key(ts: str, granularity: str) -> str | None:
    """Round ``ts`` down to the nearest hour/day so the frontend charts
    can sum into tidy buckets."""
    try:
        dt = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    if granularity == "day":
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        dt = dt.replace(minute=0, second=0, microsecond=0)
    return dt.isoformat()


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    frac = rank - lower
    return values[lower] + (values[upper] - values[lower]) * frac


@router.get("")
def analytics(
    request: Request,
    _user: str = Depends(require_session),
    hours: int = Query(default=168, ge=1, le=24 * 90, description="Window in hours (default 7d)"),
    granularity: str = Query(default="hour", pattern="^(hour|day)$"),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        if ctx.project_id:
            rows = db.fetchall(
                """SELECT trace_id, parent_span_id, name, start_time, end_time,
                          status, attributes
                   FROM spans
                   WHERE start_time >= ? AND project_id = ?""",
                (since.isoformat(), ctx.project_id),
            )
        else:
            rows = db.fetchall(
                """SELECT trace_id, parent_span_id, name, start_time, end_time,
                          status, attributes
                   FROM spans
                   WHERE start_time >= ?""",
                (since.isoformat(),),
            )

        # Roots define a trace for this aggregation (one duration per trace).
        trace_stats: dict[str, dict[str, Any]] = {}
        # Per-span LLM cost goes into both global and per-agent buckets.
        series: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"durations_ms": [], "cost_usd": 0.0, "errors": 0, "total": 0}
        )
        agent_totals: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"run_count": 0, "total_duration_ms": 0, "total_cost_usd": 0.0, "errors": 0}
        )

        for row in rows:
            try:
                attrs = json.loads(row.get("attributes") or "{}") or {}
            except json.JSONDecodeError:
                attrs = {}

            is_root = not row.get("parent_span_id")
            bucket = _bucket_key(row["start_time"], granularity)
            if bucket is None:
                continue

            # Cost per span (reported or computed).
            reported = trace_cost_usd(attrs)
            if reported is not None:
                span_cost = reported
            else:
                span_cost = (
                    compute_cost_usd(
                        attrs.get("gen_ai.request.model"),
                        attrs.get("gen_ai.usage.input_tokens"),
                        attrs.get("gen_ai.usage.output_tokens"),
                    )
                    or 0.0
                )
            series[bucket]["cost_usd"] += span_cost

            if is_root:
                start = row.get("start_time")
                end = row.get("end_time")
                dur_ms = None
                if start and end:
                    try:
                        dur_ms = (
                            datetime.fromisoformat(end) - datetime.fromisoformat(start)
                        ).total_seconds() * 1000.0
                    except (ValueError, TypeError):
                        pass

                trace_stats[row["trace_id"]] = {
                    "duration_ms": dur_ms,
                    "status": row.get("status") or "OK",
                    "agent_name": attr(attrs, "agent.name"),
                    "bucket": bucket,
                }

                if dur_ms is not None:
                    series[bucket]["durations_ms"].append(dur_ms)
                series[bucket]["total"] += 1
                if (row.get("status") or "OK") != "OK":
                    series[bucket]["errors"] += 1

                agent = attr(attrs, "agent.name")
                if agent:
                    agent_totals[agent]["run_count"] += 1
                    if dur_ms is not None:
                        agent_totals[agent]["total_duration_ms"] += dur_ms
                    if (row.get("status") or "OK") != "OK":
                        agent_totals[agent]["errors"] += 1

            # Per-agent cost still picks up from every span.
            cost_agent = attr(attrs, "agent.name")
            if cost_agent and span_cost:
                agent_totals[cost_agent]["total_cost_usd"] += span_cost

        # Assemble time-series.
        points: list[dict[str, Any]] = []
        for bucket in sorted(series.keys()):
            s = series[bucket]
            points.append(
                {
                    "bucket": bucket,
                    "trace_count": s["total"],
                    "error_count": s["errors"],
                    "error_rate": (s["errors"] / s["total"]) if s["total"] else 0.0,
                    "cost_usd": round(s["cost_usd"], 6),
                    "p50_ms": _percentile(s["durations_ms"], 0.50),
                    "p95_ms": _percentile(s["durations_ms"], 0.95),
                    "p99_ms": _percentile(s["durations_ms"], 0.99),
                }
            )

        # Top agents by two axes.
        top_slowest = sorted(
            [
                {
                    "agent_name": name,
                    "run_count": stats["run_count"],
                    "avg_latency_ms": (
                        stats["total_duration_ms"] / stats["run_count"]
                        if stats["run_count"]
                        else 0.0
                    ),
                    "total_cost_usd": stats["total_cost_usd"],
                    "error_count": stats["errors"],
                }
                for name, stats in agent_totals.items()
                if stats["run_count"] > 0
            ],
            key=lambda a: a["avg_latency_ms"],
            reverse=True,
        )[:5]
        top_priciest = sorted(
            [
                {
                    "agent_name": name,
                    "run_count": stats["run_count"],
                    "total_cost_usd": stats["total_cost_usd"],
                    "avg_cost_usd": (
                        stats["total_cost_usd"] / stats["run_count"] if stats["run_count"] else 0.0
                    ),
                    "error_count": stats["errors"],
                }
                for name, stats in agent_totals.items()
                if stats["total_cost_usd"] > 0
            ],
            key=lambda a: a["total_cost_usd"],
            reverse=True,
        )[:5]

        # Global summary numbers.
        total_traces = len(trace_stats)
        total_errors = sum(1 for t in trace_stats.values() if t["status"] != "OK")
        durations = [t["duration_ms"] for t in trace_stats.values() if t["duration_ms"] is not None]
        total_cost = sum(s["cost_usd"] for s in series.values())

        return {
            "window_hours": hours,
            "granularity": granularity,
            "summary": {
                "trace_count": total_traces,
                "error_count": total_errors,
                "error_rate": (total_errors / total_traces) if total_traces else 0.0,
                "total_cost_usd": round(total_cost, 6),
                "p50_ms": _percentile(durations, 0.50),
                "p95_ms": _percentile(durations, 0.95),
                "p99_ms": _percentile(durations, 0.99),
            },
            "points": points,
            "top_slowest_agents": top_slowest,
            "top_priciest_agents": top_priciest,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------


@router.get("/costs")
def cost_breakdown(
    request: Request,
    group_by: str = Query(
        "model",
        pattern="^(model|agent|node)$",
        description="Bucket spans by model, agent, or chain node.",
    ),
    period: str = Query(
        "7d",
        pattern="^(1d|7d|30d|all)$",
        description="Lookback window: 1d / 7d / 30d / all.",
    ),
    chain_name: str | None = Query(
        None, description="Required when group_by=node — restricts to one chain."
    ),
    agent: str | None = Query(
        None, description="Optional filter (group_by=agent or model)."
    ),
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Three breakdowns over the spans table for the cost dashboard.

    By model: groups every LLM span by ``gen_ai.request.model`` and sums
    tokens + cost.

    By agent: groups root agent spans, plus their descendant LLM spans for
    cost.

    By node: groups by ``chain.node_id`` for one named chain.
    """
    if group_by == "node" and not chain_name:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "chain_name is required when group_by=node",
        )

    ctx = get_context(request)
    db = ctx.db()
    try:
        since = datetime.now(tz=timezone.utc) - timedelta(hours=_period_to_hours(period))
        if ctx.project_id:
            rows = db.fetchall(
                """SELECT trace_id, parent_span_id, name, start_time, end_time,
                          status, attributes
                   FROM spans
                   WHERE start_time >= ? AND project_id = ?""",
                (since.isoformat(), ctx.project_id),
            )
        else:
            rows = db.fetchall(
                """SELECT trace_id, parent_span_id, name, start_time, end_time,
                          status, attributes
                   FROM spans
                   WHERE start_time >= ?""",
                (since.isoformat(),),
            )

        if group_by == "model":
            return {
                "group_by": "model",
                "period": period,
                "rows": _by_model(rows, agent_filter=agent),
            }
        if group_by == "agent":
            return {
                "group_by": "agent",
                "period": period,
                "rows": _by_agent(rows),
            }
        return {
            "group_by": "node",
            "period": period,
            "chain_name": chain_name,
            "rows": _by_node(rows, chain_name=chain_name),
        }
    finally:
        db.close()


def _by_model(rows: list[dict[str, Any]], *, agent_filter: str | None) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    )
    for row in rows:
        try:
            attrs = json.loads(row.get("attributes") or "{}") or {}
        except json.JSONDecodeError:
            attrs = {}
        model = attrs.get("gen_ai.request.model")
        if not model:
            continue
        if agent_filter and attr(attrs, "agent.name") != agent_filter:
            continue
        b = buckets[str(model)]
        b["calls"] += 1
        b["input_tokens"] += int(attrs.get("gen_ai.usage.input_tokens") or 0)
        b["output_tokens"] += int(attrs.get("gen_ai.usage.output_tokens") or 0)
        b["cost_usd"] += _span_cost(attrs)
    return sorted(
        [
            {
                "model": name,
                "calls": stats["calls"],
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "cost_usd": round(stats["cost_usd"], 6),
            }
            for name, stats in buckets.items()
        ],
        key=lambda r: -r["cost_usd"],
    )


def _by_agent(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"runs": 0, "tokens": 0, "cost": 0.0}
    )
    for row in rows:
        try:
            attrs = json.loads(row.get("attributes") or "{}") or {}
        except json.JSONDecodeError:
            attrs = {}
        agent_name = attr(attrs, "agent.name")
        if not agent_name:
            continue
        bucket = runs[agent_name]
        if not row.get("parent_span_id"):
            # root agent span — count one run
            bucket["runs"] += 1
        bucket["tokens"] += int(attrs.get("gen_ai.usage.input_tokens") or 0) + int(
            attrs.get("gen_ai.usage.output_tokens") or 0
        )
        bucket["cost"] += _span_cost(attrs)
    return sorted(
        [
            {
                "agent": name,
                "runs": stats["runs"],
                "avg_tokens": (
                    round(stats["tokens"] / stats["runs"]) if stats["runs"] else 0
                ),
                "avg_cost_usd": (
                    round(stats["cost"] / stats["runs"], 6) if stats["runs"] else 0.0
                ),
                "total_cost_usd": round(stats["cost"], 6),
            }
            for name, stats in runs.items()
        ],
        key=lambda r: -r["total_cost_usd"],
    )


def _by_node(
    rows: list[dict[str, Any]], *, chain_name: str | None
) -> list[dict[str, Any]]:
    """Group LLM/tool spans by chain.node_id, scoped to one chain.

    The SDK stamps ``chain.node_id`` on every span emitted from inside a
    chain node. We bucket by that key and roll up executions, average
    duration, average cost, and the percent contribution to the chain's
    total cost.
    """
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"executions": 0, "total_duration_ms": 0.0, "cost": 0.0}
    )
    total_cost = 0.0
    for row in rows:
        try:
            attrs = json.loads(row.get("attributes") or "{}") or {}
        except json.JSONDecodeError:
            attrs = {}
        node = attr(attrs, "chain.node_id")
        if not node:
            continue
        belongs = attr(attrs, "chain.name")
        if chain_name and belongs and belongs != chain_name:
            continue
        bucket = buckets[str(node)]
        bucket["executions"] += 1
        try:
            dur = (
                datetime.fromisoformat(row["end_time"])
                - datetime.fromisoformat(row["start_time"])
            ).total_seconds() * 1000.0
            bucket["total_duration_ms"] += dur
        except (TypeError, ValueError, KeyError):
            pass
        cost = _span_cost(attrs)
        bucket["cost"] += cost
        total_cost += cost
    out = [
        {
            "node": name,
            "executions": stats["executions"],
            "avg_duration_ms": (
                round(stats["total_duration_ms"] / stats["executions"])
                if stats["executions"]
                else 0
            ),
            "avg_cost_usd": (
                round(stats["cost"] / stats["executions"], 6) if stats["executions"] else 0.0
            ),
            "percent_of_total": (
                round(100.0 * stats["cost"] / total_cost, 1) if total_cost else 0.0
            ),
        }
        for name, stats in buckets.items()
    ]
    out.sort(key=lambda r: -r["avg_cost_usd"])
    return out
