"""Cost and token roll-ups over a trace's spans (1.89.0).

Two readers got this wrong in opposite directions:

* the eval routes read **root spans only**, and the SDK puts cost on the child
  ``llm.*`` spans, so every eval run's cost read ``0`` ("—");
* the Traces summary added the root agent span's rolled-up
  ``agent.tokens_used`` **and** every child span's tokens, so a trace reported
  twice its tokens.

The rule here: per-call spans are the source of truth. A roll-up attribute on
an ancestor (``agent.tokens_used``, ``agent.cost_usd``) is used only when no
span in the trace carries per-call numbers. Unknown is not zero: a trace whose
model calls cannot be priced has ``cost_known=False``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastaiagent.ui.attrs import attr

_CHUNK = 500


@dataclass
class TraceRollup:
    cost_usd: float | None
    """``None`` when nothing in the trace could be priced."""
    cost_known: bool
    """Every model call in the trace was priced (and there was at least one)."""
    tokens: int | None
    duration_ms: float | None
    """The root span's wall-clock duration."""


def _attrs(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rollup_spans(spans: Iterable[dict[str, Any]]) -> TraceRollup:
    """Roll up one trace. ``spans`` are rows with ``attributes`` (JSON or dict),
    ``parent_span_id``, ``start_time`` and ``end_time``."""
    from fastaiagent.ui.pricing import compute_cost_usd

    leaf_cost = 0.0
    leaf_priced = 0
    unpriced_calls = 0
    leaf_tokens = 0
    leaf_token_spans = 0
    rollup_tokens: int | None = None
    rollup_cost: float | None = None
    duration_ms: float | None = None

    for sp in spans:
        a = _attrs(sp.get("attributes"))
        tin = _num(a.get("gen_ai.usage.input_tokens"))
        tout = _num(a.get("gen_ai.usage.output_tokens"))
        span_cost = _num(attr(a, "cost.total_usd"))
        if tin is not None or tout is not None:
            leaf_token_spans += 1
            leaf_tokens += int((tin or 0) + (tout or 0))
        if span_cost is not None:
            leaf_cost += span_cost
            leaf_priced += 1
        elif tin is not None or tout is not None:
            estimated = compute_cost_usd(a.get("gen_ai.request.model"), tin or 0, tout or 0)
            if estimated is not None:
                leaf_cost += estimated
                leaf_priced += 1
            elif (tin or 0) + (tout or 0) > 0:
                unpriced_calls += 1
        rt = _num(attr(a, "agent.tokens_used"))
        if rt is not None:
            rollup_tokens = (rollup_tokens or 0) + int(rt)
        rc = _num(attr(a, "agent.cost_usd"))
        if rc is not None:
            rollup_cost = (rollup_cost or 0.0) + rc
        if not sp.get("parent_span_id"):
            try:
                start = datetime.fromisoformat(sp["start_time"])
                end = datetime.fromisoformat(sp["end_time"])
                duration_ms = (end - start).total_seconds() * 1000
            except (KeyError, TypeError, ValueError):
                pass

    if leaf_priced:
        cost: float | None = leaf_cost
        known = unpriced_calls == 0
    elif rollup_cost is not None:
        cost, known = rollup_cost, True
    else:
        cost, known = None, False
    tokens = leaf_tokens if leaf_token_spans else rollup_tokens
    return TraceRollup(
        cost_usd=cost, cost_known=known, tokens=tokens or None, duration_ms=duration_ms
    )


def span_cost(attributes: Any) -> float | None:
    """One span's own cost: reported, else priced from its model and tokens."""
    from fastaiagent.ui.pricing import compute_cost_usd

    a = _attrs(attributes)
    reported = _num(attr(a, "cost.total_usd"))
    if reported is not None:
        return reported
    tin = _num(a.get("gen_ai.usage.input_tokens"))
    tout = _num(a.get("gen_ai.usage.output_tokens"))
    if tin is None and tout is None:
        return None
    return compute_cost_usd(a.get("gen_ai.request.model"), tin or 0, tout or 0)


def costs_by_anchor(
    db: Any, anchors: Iterable[dict[str, Any]], *, project_id: str = ""
) -> dict[str, float]:
    """``{anchor_span_id: cost}``: each priced span's cost goes to its nearest
    ancestor that is an anchor (1.89.0).

    The Agents and Workflows pages priced the ``agent.*`` / workflow span
    itself, which carries no cost (the model calls are its children), so every
    agent showed $0.00. Attributing to the *nearest* anchor gives each agent its
    own calls: a supervisor's worker calls count for the worker, not twice.
    """
    anchor_rows = list(anchors)
    anchor_ids = {a["span_id"] for a in anchor_rows if a.get("span_id")}
    trace_ids = list(dict.fromkeys(a["trace_id"] for a in anchor_rows if a.get("trace_id")))
    out: dict[str, float] = {sid: 0.0 for sid in anchor_ids}
    pid_extra = "AND project_id = ?" if project_id else ""
    pid_params: tuple[Any, ...] = (project_id,) if project_id else ()
    for i in range(0, len(trace_ids), _CHUNK):
        chunk = trace_ids[i : i + _CHUNK]
        rows = db.fetchall(
            f"""SELECT span_id, parent_span_id, attributes FROM spans
                WHERE trace_id IN ({",".join("?" * len(chunk))}) {pid_extra}""",
            (*chunk, *pid_params),
        )
        parent = {r["span_id"]: r["parent_span_id"] for r in rows}
        for r in rows:
            cost = span_cost(r["attributes"])
            if not cost:
                continue
            node: str | None = r["span_id"]
            seen = 0
            while node is not None and node not in anchor_ids and seen < 256:
                node = parent.get(node)
                seen += 1
            if node is not None and node in anchor_ids:
                out[node] += cost
    return out


def trace_rollups(
    db: Any, trace_ids: Iterable[str], *, project_id: str = ""
) -> dict[str, TraceRollup]:
    """``{trace_id: TraceRollup}`` for every trace that has spans; chunked."""
    ids = [t for t in dict.fromkeys(trace_ids) if t]
    out: dict[str, TraceRollup] = {}
    pid_extra = "AND project_id = ?" if project_id else ""
    pid_params: tuple[Any, ...] = (project_id,) if project_id else ()
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i : i + _CHUNK]
        rows = db.fetchall(
            f"""SELECT trace_id, parent_span_id, attributes, start_time, end_time
                FROM spans WHERE trace_id IN ({",".join("?" * len(chunk))}) {pid_extra}""",
            (*chunk, *pid_params),
        )
        by_trace: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_trace.setdefault(r["trace_id"], []).append(dict(r))
        for tid, spans in by_trace.items():
            out[tid] = rollup_spans(spans)
    return out


def sum_rollups(
    rollups: dict[str, TraceRollup], trace_ids: Iterable[str]
) -> tuple[float | None, bool, float | None]:
    """``(cost_usd, cost_known, avg_duration_ms)`` over ``trace_ids``.

    ``cost_known`` is False when any listed trace is missing or unpriced; the
    cost is then a lower bound (still shown, flagged), or ``None`` when nothing
    was priced at all.
    """
    total = 0.0
    any_cost = False
    known = True
    durations: list[float] = []
    ids = [t for t in trace_ids if t]
    if not ids:
        return None, False, None
    for tid in ids:
        r = rollups.get(tid)
        if r is None:
            known = False
            continue
        if r.cost_usd is not None:
            total += r.cost_usd
            any_cost = True
        if not r.cost_known:
            known = False
        if r.duration_ms is not None:
            durations.append(r.duration_ms)
    avg = sum(durations) / len(durations) if durations else None
    return (round(total, 8) if any_cost else None), (known and any_cost), avg
