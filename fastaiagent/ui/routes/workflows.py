"""Workflow directory: chains, swarms, and supervisors.

Derived from root spans named ``chain.<name>`` / ``swarm.<name>`` /
``supervisor.<name>`` that the SDK emits when a Chain, Swarm, or Supervisor
executes. Mirrors the agents route (``routes/agents.py``) but filters on the
workflow runner types instead of on ``agent.*`` spans.

Read-only. Writes (editing a chain, reordering nodes) live in code or on
Platform's visual canvas — not here.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

_RUNNER_TYPES = ("chain", "swarm", "supervisor")


def _runner_type_from_span_name(name: str | None) -> str | None:
    if not name:
        return None
    for rt in _RUNNER_TYPES:
        if name.startswith(f"{rt}."):
            return rt
    return None


def _workflow_name(attrs: dict[str, Any], runner_type: str, span_name: str) -> str:
    """Prefer the explicit attribute, fall back to stripping the ``<type>.`` prefix."""
    from fastaiagent.ui.attrs import attr

    name = attr(attrs, f"{runner_type}.name")
    if name:
        return str(name)
    prefix = f"{runner_type}."
    if span_name and span_name.startswith(prefix):
        return span_name[len(prefix) :]
    return span_name or "unknown"


def _aggregate(
    spans: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Bucket by (runner_type, workflow_name)."""
    from fastaiagent.ui.attrs import attr, trace_cost_usd
    from fastaiagent.ui.pricing import compute_cost_usd

    by_wf: dict[tuple[str, str], dict[str, Any]] = {}
    for span in spans:
        span_name = span.get("name") or ""
        runner_type = _runner_type_from_span_name(span_name)
        if runner_type is None:
            continue
        try:
            attrs = json.loads(span.get("attributes") or "{}")
        except json.JSONDecodeError:
            attrs = {}
        wf_name = _workflow_name(attrs, runner_type, span_name)
        key = (runner_type, wf_name)
        bucket = by_wf.setdefault(
            key,
            {
                "runner_type": runner_type,
                "workflow_name": wf_name,
                "run_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_duration_ms": 0,
                "total_cost_usd": 0.0,
                "last_run": "",
                "node_count": attr(attrs, f"{runner_type}.node_count"),
            },
        )
        bucket["run_count"] += 1
        # OTel convention: UNSET is the default for a span that completed
        # normally without the SDK explicitly marking it OK. Treat UNSET
        # and OK as success; only ERROR counts as a failure.
        if span.get("status") == "ERROR":
            bucket["error_count"] += 1
        else:
            bucket["success_count"] += 1

        start = span.get("start_time") or ""
        end = span.get("end_time") or ""
        try:
            a = datetime.fromisoformat(start)
            b = datetime.fromisoformat(end)
            bucket["total_duration_ms"] += int((b - a).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass

        reported_cost = trace_cost_usd(attrs)
        if reported_cost is not None:
            bucket["total_cost_usd"] += reported_cost
        else:
            est = compute_cost_usd(
                attrs.get("gen_ai.request.model"),
                attrs.get("gen_ai.usage.input_tokens"),
                attrs.get("gen_ai.usage.output_tokens"),
            )
            if est is not None:
                bucket["total_cost_usd"] += est
        if start > bucket["last_run"]:
            bucket["last_run"] = start

        # Latest node_count wins if the attribute was added later.
        latest_nc = attr(attrs, f"{runner_type}.node_count")
        if latest_nc is not None:
            bucket["node_count"] = latest_nc
    return by_wf


def _format(bucket: dict[str, Any]) -> dict[str, Any]:
    runs = bucket["run_count"] or 1
    return {
        "runner_type": bucket["runner_type"],
        "workflow_name": bucket["workflow_name"],
        "run_count": bucket["run_count"],
        "success_rate": bucket["success_count"] / runs,
        "error_count": bucket["error_count"],
        "avg_latency_ms": bucket["total_duration_ms"] / runs,
        "avg_cost_usd": bucket["total_cost_usd"] / runs,
        "last_run": bucket["last_run"],
        "node_count": bucket.get("node_count"),
    }


@router.get("")
def list_workflows(
    request: Request,
    runner_type: str | None = Query(
        None,
        description="Restrict to one of: chain, swarm, supervisor",
    ),
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    if runner_type is not None and runner_type not in _RUNNER_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"runner_type must be one of {list(_RUNNER_TYPES)}",
        )
    ctx = get_context(request)
    db = ctx.db()
    try:
        if runner_type:
            rows = db.fetchall(
                "SELECT * FROM spans WHERE parent_span_id IS NULL "
                "AND name LIKE ? ORDER BY start_time DESC",
                (f"{runner_type}.%",),
            )
        else:
            rows = db.fetchall(
                "SELECT * FROM spans WHERE parent_span_id IS NULL "
                "AND (name LIKE 'chain.%' OR name LIKE 'swarm.%' "
                "OR name LIKE 'supervisor.%') ORDER BY start_time DESC"
            )
        by_wf = _aggregate(rows)
        return {
            "workflows": [_format(b) for b in by_wf.values()],
        }
    finally:
        db.close()


@router.get("/{runner_type}/{name}")
def get_workflow(
    request: Request,
    runner_type: str,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    if runner_type not in _RUNNER_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"runner_type must be one of {list(_RUNNER_TYPES)}",
        )
    ctx = get_context(request)
    db = ctx.db()
    try:
        rows = db.fetchall(
            "SELECT * FROM spans WHERE parent_span_id IS NULL "
            "AND name LIKE ?",
            (f"{runner_type}.%",),
        )
        by_wf = _aggregate(rows)
        key = (runner_type, name)
        if key not in by_wf:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"{runner_type.capitalize()} '{name}' not found",
            )
        return _format(by_wf[key])
    finally:
        db.close()
