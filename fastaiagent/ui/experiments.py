"""Production views leave experiments out, and say what they left out (1.89.0).

Analytics, Overview and Agents used to count replay reruns, Playground calls,
learn passes and eval runs as production traffic: a rerun was a "run" of the
agent, its tokens were production spend, and an interrupted rerun was a
production failure. These views now exclude every trace with a non-production
span (see :mod:`fastaiagent.trace.provenance`) unless the caller asks for
``include_experiments``, and report the excluded traces separately, so real
spend is never hidden.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from fastaiagent.trace.provenance import SOURCE_ATTR, is_non_production


def _attrs(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("attributes")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def experiment_trace_sources(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """``{trace_id: source}`` for every trace with a non-production span."""
    out: dict[str, str] = {}
    for row in rows:
        attrs = _attrs(row)
        if is_non_production(attrs):
            out.setdefault(row["trace_id"], str(attrs.get(SOURCE_ATTR)))
    return out


def split_experiments(
    rows: list[dict[str, Any]], *, include_experiments: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """``(production_rows, experiment_rows, {trace_id: source})``.

    With ``include_experiments`` every row is production and nothing is split
    out, but the experiment map is still returned for the summary.
    """
    sources = experiment_trace_sources(rows)
    if include_experiments or not sources:
        return rows, [r for r in rows if r["trace_id"] in sources], sources
    prod = [r for r in rows if r["trace_id"] not in sources]
    exp = [r for r in rows if r["trace_id"] in sources]
    return prod, exp, sources


def experiments_summary(
    exp_rows: list[dict[str, Any]],
    sources: dict[str, str],
    *,
    span_cost: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """``{trace_count, cost_usd, by_source}`` for the experiment traces."""
    by_source: dict[str, dict[str, Any]] = {}
    cost_total = 0.0
    for row in exp_rows:
        source = sources.get(row["trace_id"], "unknown")
        entry = by_source.setdefault(source, {"trace_count": 0, "cost_usd": 0.0, "_ids": set()})
        entry["_ids"].add(row["trace_id"])
        cost = span_cost(_attrs(row)) or 0.0
        entry["cost_usd"] += cost
        cost_total += cost
    for entry in by_source.values():
        entry["trace_count"] = len(entry.pop("_ids"))
        entry["cost_usd"] = round(entry["cost_usd"], 6)
    return {
        "trace_count": len(sources),
        "cost_usd": round(cost_total, 6),
        "by_source": by_source,
    }


__all__ = ["experiment_trace_sources", "experiments_summary", "split_experiments"]
