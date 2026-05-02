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
        pid_clause = "AND project_id = ?" if ctx.project_id else ""
        if runner_type:
            sql = (
                "SELECT * FROM spans WHERE parent_span_id IS NULL "
                f"AND name LIKE ? {pid_clause} ORDER BY start_time DESC"
            )
            params: tuple = (
                (f"{runner_type}.%", ctx.project_id)
                if ctx.project_id
                else (f"{runner_type}.%",)
            )
            rows = db.fetchall(sql, params)
        else:
            sql = (
                "SELECT * FROM spans WHERE parent_span_id IS NULL "
                "AND (name LIKE 'chain.%' OR name LIKE 'swarm.%' "
                f"OR name LIKE 'supervisor.%') {pid_clause} "
                "ORDER BY start_time DESC"
            )
            params = (ctx.project_id,) if ctx.project_id else ()
            rows = db.fetchall(sql, params)
        by_wf = _aggregate(rows)
        return {
            "workflows": [_format(b) for b in by_wf.values()],
            "registered": bool(getattr(ctx, "runners", {})),
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
        if ctx.project_id:
            rows = db.fetchall(
                "SELECT * FROM spans WHERE parent_span_id IS NULL "
                "AND name LIKE ? AND project_id = ?",
                (f"{runner_type}.%", ctx.project_id),
            )
        else:
            rows = db.fetchall(
                "SELECT * FROM spans WHERE parent_span_id IS NULL AND name LIKE ?",
                (f"{runner_type}.%",),
            )
        by_wf = _aggregate(rows)
        key = (runner_type, name)
        if key in by_wf:
            return _format(by_wf[key])
        # Fall back to the registered runner — lets a freshly-registered
        # workflow render its topology before its first run lands spans
        # in the DB. The summary stats are zero until then.
        registered = (getattr(ctx, "runners", {}) or {}).get(name)
        if registered is not None and _runner_type_of(registered) == runner_type:
            return {
                "runner_type": runner_type,
                "workflow_name": name,
                "run_count": 0,
                "success_rate": 0.0,
                "error_count": 0,
                "avg_latency_ms": 0.0,
                "avg_cost_usd": 0.0,
                "last_run": "",
                "node_count": (
                    len(getattr(registered, "nodes", []))
                    if hasattr(registered, "nodes")
                    else None
                ),
            }
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"{runner_type.capitalize()} '{name}' not found",
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def _runner_type_of(runner: Any) -> str:
    """Best-effort type tag for a registered runner."""
    cls = type(runner).__name__.lower()
    if "chain" in cls:
        return "chain"
    if "swarm" in cls:
        return "swarm"
    if "supervisor" in cls:
        return "supervisor"
    return cls


def _chain_topology(runner: Any) -> dict[str, Any]:
    """Convert a Chain to the topology shape expected by the frontend."""
    from fastaiagent.chain.executor import _topological_sort

    raw = runner.to_dict()
    nodes_raw = raw.get("nodes", [])
    edges_raw = raw.get("edges", [])

    # Frontend node payload: id + type + label + per-node metadata.
    nodes: list[dict[str, Any]] = []
    tools_out: list[dict[str, Any]] = []
    for n in runner.nodes:
        node_payload: dict[str, Any] = {
            "id": n.id,
            "type": n.type.value,
            "label": n.name or n.id,
        }
        if n.agent is not None:
            agent = n.agent
            node_payload["agent_name"] = getattr(agent, "name", None)
            llm = getattr(agent, "llm", None)
            if llm is not None:
                node_payload["model"] = getattr(llm, "model", "")
                node_payload["provider"] = getattr(llm, "provider", "")
            node_payload["tool_count"] = len(getattr(agent, "tools", []) or [])
            for t in getattr(agent, "tools", []) or []:
                tools_out.append(
                    {
                        "owner": n.id,
                        "name": getattr(t, "name", str(t)),
                        "type": "function",
                    }
                )
        if n.tool is not None:
            node_payload["tool_name"] = getattr(n.tool, "name", None)
        nodes.append(node_payload)

    edges: list[dict[str, Any]] = []
    for e in edges_raw:
        edge_payload: dict[str, Any] = {
            "from": e["source"],
            "to": e["target"],
            "type": "conditional" if e.get("condition") else "sequential",
        }
        if e.get("condition"):
            edge_payload["condition"] = e["condition"]
        if e.get("label"):
            edge_payload["label"] = e["label"]
        if e.get("is_cyclic"):
            edge_payload["is_cyclic"] = True
            edge_payload["cycle_config"] = e.get("cycle_config", {})
        edges.append(edge_payload)

    order = _topological_sort(runner.nodes, runner.edges)
    entrypoint = order[0] if order else (nodes_raw[0]["id"] if nodes_raw else None)

    return {
        "name": runner.name,
        "type": "chain",
        "nodes": nodes,
        "edges": edges,
        "entrypoint": entrypoint,
        "tools": tools_out,
        "knowledge_bases": [],
    }


def _swarm_topology(runner: Any) -> dict[str, Any]:
    """Convert a Swarm into nodes (one per peer) + handoff edges."""
    raw = runner.to_dict()
    nodes: list[dict[str, Any]] = []
    tools_out: list[dict[str, Any]] = []
    for agent_name, agent in runner.agents.items():
        node_payload: dict[str, Any] = {
            "id": agent_name,
            "type": "agent",
            "label": agent_name,
            "agent_name": agent_name,
        }
        llm = getattr(agent, "llm", None)
        if llm is not None:
            node_payload["model"] = getattr(llm, "model", "")
            node_payload["provider"] = getattr(llm, "provider", "")
        node_payload["tool_count"] = len(getattr(agent, "tools", []) or [])
        for t in getattr(agent, "tools", []) or []:
            tools_out.append(
                {
                    "owner": agent_name,
                    "name": getattr(t, "name", str(t)),
                    "type": "function",
                }
            )
        nodes.append(node_payload)

    edges: list[dict[str, Any]] = []
    for source, targets in raw.get("handoffs", {}).items():
        for target in targets:
            edges.append(
                {
                    "from": source,
                    "to": target,
                    "type": "handoff",
                    "label": "handoff",
                }
            )

    return {
        "name": runner.name,
        "type": "swarm",
        "nodes": nodes,
        "edges": edges,
        "entrypoint": raw.get("entrypoint"),
        "tools": tools_out,
        "knowledge_bases": [],
        "max_handoffs": raw.get("max_handoffs"),
    }


def _supervisor_topology(runner: Any) -> dict[str, Any]:
    """Convert a Supervisor into one supervisor node + worker nodes + delegation edges."""
    raw = runner.to_dict()
    sup_id = f"supervisor:{runner.name}"
    sup_llm = raw.get("supervisor_llm", {})
    nodes: list[dict[str, Any]] = [
        {
            "id": sup_id,
            "type": "supervisor",
            "label": runner.name,
            "model": sup_llm.get("model", ""),
            "provider": sup_llm.get("provider", ""),
        }
    ]
    edges: list[dict[str, Any]] = []
    tools_out: list[dict[str, Any]] = []
    for w in raw.get("workers", []):
        worker_id = f"worker:{w['role']}"
        nodes.append(
            {
                "id": worker_id,
                "type": "agent",
                "label": w["role"],
                "agent_name": w.get("agent_name"),
                "model": w.get("model", ""),
                "description": w.get("description", ""),
                "tool_count": len(w.get("tools", [])),
            }
        )
        edges.append(
            {
                "from": sup_id,
                "to": worker_id,
                "type": "delegation",
                "label": "delegate",
            }
        )
        for tool_name in w.get("tools", []):
            tools_out.append(
                {
                    "owner": worker_id,
                    "name": tool_name,
                    "type": "function",
                }
            )

    return {
        "name": runner.name,
        "type": "supervisor",
        "nodes": nodes,
        "edges": edges,
        "entrypoint": sup_id,
        "tools": tools_out,
        "knowledge_bases": [],
        "max_delegation_rounds": raw.get("max_delegation_rounds"),
    }


@router.get("/{runner_type}/{name}/topology")
def get_topology(
    request: Request,
    runner_type: str,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Return the runtime graph for a registered Chain/Swarm/Supervisor.

    Reads from ``app.state.context.runners`` — runners must be passed to
    :func:`fastaiagent.ui.server.build_app` via ``runners=[...]``.
    """
    if runner_type not in _RUNNER_TYPES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"runner_type must be one of {list(_RUNNER_TYPES)}",
        )
    ctx = get_context(request)
    runners = getattr(ctx, "runners", {}) or {}
    runner = runners.get(name)
    if runner is None or _runner_type_of(runner) != runner_type:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"{runner_type} '{name}' is not registered with the local UI server. "
            "Pass it to build_app(runners=[...]) to enable topology.",
        )
    if runner_type == "chain":
        return _chain_topology(runner)
    if runner_type == "swarm":
        return _swarm_topology(runner)
    return _supervisor_topology(runner)
