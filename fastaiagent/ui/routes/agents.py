"""Agent directory derived from span attributes."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/agents", tags=["agents"])


def _aggregate(
    spans: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bucket root spans by agent name, computing per-agent stats."""
    from fastaiagent.ui.attrs import attr, trace_cost_usd

    by_agent: dict[str, dict[str, Any]] = {}
    for span in spans:
        try:
            attrs = json.loads(span.get("attributes") or "{}")
        except json.JSONDecodeError:
            attrs = {}
        name = attr(attrs, "agent.name")
        if not name:
            continue
        bucket = by_agent.setdefault(
            name,
            {
                "agent_name": name,
                "run_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_duration_ms": 0,
                "total_cost_usd": 0.0,
                "last_run": "",
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
        latency_ms = attr(attrs, "agent.latency_ms")
        if latency_ms is not None:
            try:
                bucket["total_duration_ms"] += int(float(latency_ms))
            except (TypeError, ValueError):
                pass
        else:
            try:
                from datetime import datetime

                a = datetime.fromisoformat(start)
                b = datetime.fromisoformat(end)
                bucket["total_duration_ms"] += int((b - a).total_seconds() * 1000)
            except (ValueError, TypeError):
                pass
        reported_cost = trace_cost_usd(attrs)
        if reported_cost is not None:
            bucket["total_cost_usd"] += reported_cost
        else:
            from fastaiagent.ui.pricing import compute_cost_usd

            est = compute_cost_usd(
                attrs.get("gen_ai.request.model"),
                attrs.get("gen_ai.usage.input_tokens"),
                attrs.get("gen_ai.usage.output_tokens"),
            )
            if est is not None:
                bucket["total_cost_usd"] += est
        if start > bucket["last_run"]:
            bucket["last_run"] = start
    return by_agent


def _format(bucket: dict[str, Any]) -> dict[str, Any]:
    run_count = bucket["run_count"] or 1
    return {
        "agent_name": bucket["agent_name"],
        "run_count": bucket["run_count"],
        "success_rate": bucket["success_count"] / run_count,
        "error_count": bucket["error_count"],
        "avg_latency_ms": bucket["total_duration_ms"] / run_count,
        "avg_cost_usd": bucket["total_cost_usd"] / run_count,
        "last_run": bucket["last_run"],
    }


@router.get("")
def list_agents(request: Request, _user: str = Depends(require_session)) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        # Scan every agent.* span, not just root spans. With workflow
        # wrappers (chain/swarm/supervisor), agents always run as children.
        rows = db.fetchall("SELECT * FROM spans WHERE name LIKE 'agent.%'")
        by_agent = _aggregate(rows)
        return {"agents": [_format(b) for b in by_agent.values()]}
    finally:
        db.close()


@router.get("/{name}")
def get_agent(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        rows = db.fetchall("SELECT * FROM spans WHERE name LIKE 'agent.%'")
        by_agent = _aggregate(rows)
        if name not in by_agent:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Agent '{name}' not found")
        return _format(by_agent[name])
    finally:
        db.close()


@router.get("/{name}/tools")
def get_agent_tools(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Return tools **registered** with this agent and tools **used** at runtime.

    *Registered* is read off the most-recent ``agent.<name>`` root span (the
    SDK emits ``agent.tools`` as JSON on every run — stays None for traces
    emitted before 0.9.4). *Used* scans every ``tool.<name>`` descendant of
    an ``agent.<name>`` span across the whole DB and aggregates call count,
    error count, and avg latency per tool name.

    The UI cross-references the two so it can badge registered-but-never-used
    tools (suggests dead code) and used-but-not-registered names
    (suggests an LLM hallucination).
    """
    from datetime import datetime

    from fastaiagent.ui.attrs import attr

    ctx = get_context(request)
    db = ctx.db()
    try:
        # ── Registered: latest agent.<name> root span with agent.tools JSON ─
        registered: list[dict[str, Any]] = []
        agent_rows = db.fetchall(
            "SELECT attributes FROM spans WHERE name = ? ORDER BY start_time DESC LIMIT 1",
            (f"agent.{name}",),
        )
        if agent_rows:
            try:
                attrs = json.loads(agent_rows[0].get("attributes") or "{}")
            except json.JSONDecodeError:
                attrs = {}
            raw = attr(attrs, "agent.tools")
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = []
                if isinstance(parsed, list):
                    for t in parsed:
                        if isinstance(t, dict) and "name" in t:
                            registered.append(
                                {
                                    "name": t.get("name"),
                                    "origin": t.get("origin") or "unknown",
                                    "description": t.get("description") or "",
                                }
                            )

        # ── Used: aggregate tool.* spans whose parent chain rolls up to agent.<name> ─
        # Pull every trace that touched this agent, then collect tool.* spans
        # in those traces. Simpler than a recursive CTE and correct for the
        # one-level nesting we ship today.
        tool_agg: dict[str, dict[str, Any]] = {}
        trace_rows = db.fetchall(
            "SELECT DISTINCT trace_id FROM spans WHERE name = ?",
            (f"agent.{name}",),
        )
        trace_ids = [r["trace_id"] for r in trace_rows if r.get("trace_id")]
        if trace_ids:
            placeholders = ",".join("?" * len(trace_ids))
            tool_span_rows = db.fetchall(
                f"""SELECT name, status, start_time, end_time, attributes
                    FROM spans
                    WHERE trace_id IN ({placeholders})
                      AND name LIKE 'tool.%'""",
                tuple(trace_ids),
            )
            for span in tool_span_rows:
                try:
                    sattrs = json.loads(span.get("attributes") or "{}")
                except json.JSONDecodeError:
                    sattrs = {}
                tool_name = attr(sattrs, "tool.name") or (
                    span["name"][len("tool.") :] if span["name"] else "?"
                )
                bucket = tool_agg.setdefault(
                    tool_name,
                    {
                        "name": tool_name,
                        "origin": attr(sattrs, "tool.origin") or "unknown",
                        "call_count": 0,
                        "error_count": 0,
                        "total_duration_ms": 0,
                        "last_used": "",
                    },
                )
                bucket["call_count"] += 1
                # tool.status is "ok" / "error" / "unknown"; OTel-level
                # status may also say ERROR.
                tstatus = attr(sattrs, "tool.status")
                if tstatus == "error" or span.get("status") == "ERROR":
                    bucket["error_count"] += 1
                try:
                    a_time = datetime.fromisoformat(span["start_time"])
                    b_time = datetime.fromisoformat(span["end_time"])
                    bucket["total_duration_ms"] += int((b_time - a_time).total_seconds() * 1000)
                except (ValueError, TypeError):
                    pass
                start = span.get("start_time") or ""
                if start > bucket["last_used"]:
                    bucket["last_used"] = start

        used: list[dict[str, Any]] = []
        for bucket in tool_agg.values():
            runs = bucket["call_count"] or 1
            used.append(
                {
                    "name": bucket["name"],
                    "origin": bucket["origin"],
                    "call_count": bucket["call_count"],
                    "error_count": bucket["error_count"],
                    "success_rate": ((bucket["call_count"] - bucket["error_count"]) / runs),
                    "avg_latency_ms": bucket["total_duration_ms"] / runs,
                    "last_used": bucket["last_used"],
                }
            )
        used.sort(key=lambda r: -r["call_count"])

        # ── Cross-reference: mark registered tools that were never called ──
        used_names = {u["name"] for u in used}
        registered_names = {r["name"] for r in registered}
        for row in registered:
            row["used"] = row["name"] in used_names
        for row in used:
            row["registered"] = row["name"] in registered_names

        return {
            "agent_name": name,
            "registered": registered,
            "used": used,
        }
    finally:
        db.close()
