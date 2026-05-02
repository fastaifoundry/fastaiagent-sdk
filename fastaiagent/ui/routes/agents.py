"""Agent directory derived from span attributes."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from fastaiagent.ui.deps import get_context, project_filter, require_session

logger = logging.getLogger(__name__)

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
                logger.debug("Failed to parse agent.latency_ms: %r", latency_ms, exc_info=True)
        else:
            try:
                from datetime import datetime

                a = datetime.fromisoformat(start)
                b = datetime.fromisoformat(end)
                bucket["total_duration_ms"] += int((b - a).total_seconds() * 1000)
            except (ValueError, TypeError):
                logger.debug("Failed to compute agent span duration from timestamps", exc_info=True)
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


def _registered_agent_names(runners: dict[str, Any]) -> set[str]:
    """Names of every Agent / Supervisor / Swarm-peer found in ``ctx.runners``.

    Used to surface registered-but-not-yet-run agents on the directory
    page so they're discoverable before the first trace lands.
    """
    out: set[str] = set()
    for r in runners.values():
        cls = type(r).__name__.lower()
        if "supervisor" in cls:
            n = getattr(r, "name", None)
            if n:
                out.add(n)
            for w in getattr(r, "workers", []) or []:
                a = getattr(w, "agent", None)
                if a is not None and getattr(a, "name", None):
                    out.add(a.name)
        elif "swarm" in cls:
            agents = getattr(r, "agents", {}) or {}
            out.update(agents.keys())
        elif "chain" in cls:
            for n in getattr(r, "nodes", []) or []:
                a = getattr(n, "agent", None)
                if a is not None and getattr(a, "name", None):
                    out.add(a.name)
        else:
            n = getattr(r, "name", None)
            if n and hasattr(r, "tools"):
                # Looks like an Agent (has .name + .tools).
                out.add(n)
    return out


def _empty_agent_summary(name: str) -> dict[str, Any]:
    """Stub summary for a registered-but-not-run agent.

    Mirrors the shape of ``_format()`` so the frontend can render the
    same cards without a separate code path.
    """
    return {
        "agent_name": name,
        "run_count": 0,
        "success_rate": 0.0,
        "error_count": 0,
        "avg_latency_ms": 0.0,
        "avg_cost_usd": 0.0,
        "last_run": "",
    }


@router.get("")
def list_agents(request: Request, _user: str = Depends(require_session)) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        # Scan every agent.* span, not just root spans. With workflow
        # wrappers (chain/swarm/supervisor), agents always run as children.
        if ctx.project_id:
            rows = db.fetchall(
                "SELECT * FROM spans WHERE name LIKE 'agent.%' AND project_id = ?",
                (ctx.project_id,),
            )
        else:
            rows = db.fetchall("SELECT * FROM spans WHERE name LIKE 'agent.%'")
        by_agent = _aggregate(rows)
        agents = [_format(b) for b in by_agent.values()]

        # Surface registered runners that haven't produced spans yet so the
        # directory page is discoverable before the first run. Span-derived
        # entries take precedence (they have real stats); registered-only
        # rows show zeros.
        seen = {a["agent_name"] for a in agents}
        for name in _registered_agent_names(ctx.runners):
            if name not in seen:
                agents.append(_empty_agent_summary(name))
        return {"agents": agents}
    finally:
        db.close()


def _runner_type_of(runner: Any) -> str | None:
    cls = type(runner).__name__.lower()
    if "chain" in cls:
        return "chain"
    if "swarm" in cls:
        return "swarm"
    if "supervisor" in cls:
        return "supervisor"
    return None


def _registered_workflows_for(ctx: Any, agent_name: str) -> list[dict[str, str]]:
    """List registered workflows whose graph mentions this agent.

    Used by the Agent detail page to surface a topology preview link.
    """
    out: list[dict[str, str]] = []
    runners = getattr(ctx, "runners", {}) or {}
    for r in runners.values():
        rtype = _runner_type_of(r)
        if rtype is None:
            continue
        members: set[str] = set()
        if rtype == "chain":
            for n in getattr(r, "nodes", []) or []:
                a = getattr(n, "agent", None)
                if a is not None:
                    members.add(getattr(a, "name", ""))
        elif rtype == "swarm":
            members.update(getattr(r, "agents", {}).keys())
        elif rtype == "supervisor":
            for w in getattr(r, "workers", []) or []:
                a = getattr(w, "agent", None)
                if a is not None:
                    members.add(getattr(a, "name", ""))
        if agent_name in members:
            out.append({"runner_type": rtype, "name": getattr(r, "name", "")})
    return out


@router.get("/{name}")
def get_agent(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    db = ctx.db()
    try:
        if ctx.project_id:
            rows = db.fetchall(
                "SELECT * FROM spans WHERE name LIKE 'agent.%' AND project_id = ?",
                (ctx.project_id,),
            )
        else:
            rows = db.fetchall("SELECT * FROM spans WHERE name LIKE 'agent.%'")
        by_agent = _aggregate(rows)
        if name not in by_agent:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Agent '{name}' not found")
        payload = _format(by_agent[name])
        payload["workflows"] = _registered_workflows_for(ctx, name)
        return payload
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
    pid_clause, pid_params = project_filter(ctx)
    try:
        # Project-scope guard: 404 if this agent has no spans in this project,
        # so cross-project name probing can't confirm existence.
        if ctx.project_id:
            probe = db.fetchone(
                "SELECT 1 FROM spans WHERE name = ? AND project_id = ? LIMIT 1",
                (f"agent.{name}", ctx.project_id),
            )
            if probe is None:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, f"Agent '{name}' not found"
                )
        # ── Registered: latest agent.<name> root span with agent.tools JSON ─
        registered: list[dict[str, Any]] = []
        agent_rows = db.fetchall(
            f"SELECT attributes FROM spans WHERE name = ? {pid_clause} "
            "ORDER BY start_time DESC LIMIT 1",
            (f"agent.{name}", *pid_params),
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
            f"SELECT DISTINCT trace_id FROM spans WHERE name = ? {pid_clause}",
            (f"agent.{name}", *pid_params),
        )
        trace_ids = [r["trace_id"] for r in trace_rows if r.get("trace_id")]
        if trace_ids:
            placeholders = ",".join("?" * len(trace_ids))
            tool_span_rows = db.fetchall(
                f"""SELECT name, status, start_time, end_time, attributes
                    FROM spans
                    WHERE trace_id IN ({placeholders})
                      AND name LIKE 'tool.%' {pid_clause}""",
                (*trace_ids, *pid_params),
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
                    logger.debug(
                        "Failed to compute tool span duration from timestamps", exc_info=True,
                    )
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


# ---------------------------------------------------------------------------
# Dependency graph: structural "what is this agent made of" view.
#
# Walks the runtime registry (``ctx.runners``) to introspect tools / KBs /
# prompts / guardrails / sub-agents. Falls back to span-derived data with
# ``unresolved=True`` when no runner is registered.
# ---------------------------------------------------------------------------


_KB_TOOL_PREFIX = "search_"


def _detect_runtime_kb(tool: Any, kb_root: str | None = None) -> dict[str, Any] | None:
    """Best-effort resolution of a KB-origin tool back to a real LocalKB.

    The ``LocalKB.as_tool()`` adapter sets ``tool.origin = "kb"`` and names
    the tool ``"search_<kb_name>"`` but doesn't keep a back-reference. We
    parse the kb_name out and ask LocalKB to load its on-disk metadata so we
    can fill in document/chunk counts. Returns ``None`` if the parse fails
    or the KB isn't on disk — the caller falls back to a degraded payload.
    """
    name = getattr(tool, "name", "") or ""
    if not name.startswith(_KB_TOOL_PREFIX):
        return None
    kb_name = name[len(_KB_TOOL_PREFIX):]
    try:
        from fastaiagent.kb.local import LocalKB
    except ImportError:
        return None
    try:
        # Avoid mutating the on-disk index — the read-only loader is fine.
        # Honour the configured kb_root only when set; otherwise let
        # LocalKB use its default path (./.fastaiagent/kb/).
        kb = (
            LocalKB(name=kb_name, path=kb_root)
            if kb_root
            else LocalKB(name=kb_name)
        )
        status = kb.status()
        kb.close()
    except Exception:  # noqa: BLE001 — KB load failures should never 500
        return {
            "name": kb_name,
            "backend": "unknown",
            "documents": None,
            "chunks": None,
            "unresolved": True,
        }
    return {
        "name": kb_name,
        "backend": (status.get("vector_backend") or "local").lower().replace(
            "vectorstore", ""
        ),
        "documents": None,  # LocalKB doesn't expose distinct sources cheaply
        "chunks": status.get("chunk_count"),
    }


def _classify_runner(runner: Any) -> str:
    cls = type(runner).__name__.lower()
    if "supervisor" in cls:
        return "supervisor"
    if "swarm" in cls:
        return "swarm"
    if "chain" in cls:
        return "chain"
    if "agent" in cls:
        return "agent"
    return "agent"


def _find_agent_in_runners(
    runners: dict[str, Any], name: str
) -> tuple[Any, str] | tuple[None, None]:
    """Locate an Agent (or Supervisor / Swarm peer) inside ``ctx.runners``.

    Returns ``(runner, parent_kind)`` where ``parent_kind`` is one of:
      * ``"agent"`` — the runner itself is an Agent named ``name``
      * ``"supervisor"`` — the runner is a Supervisor named ``name``
      * ``"worker"`` — ``name`` matches a worker inside a Supervisor
      * ``"swarm-peer"`` — ``name`` matches an agent inside a Swarm

    The UI uses ``parent_kind`` to decide which sub-graph to render
    (single agent vs. supervisor-with-workers vs. swarm-with-peers).
    """
    for r in runners.values():
        kind = _classify_runner(r)
        if kind == "agent" and getattr(r, "name", None) == name:
            return r, "agent"
        if kind == "supervisor":
            if getattr(r, "name", None) == name:
                return r, "supervisor"
            for w in getattr(r, "workers", []) or []:
                a = getattr(w, "agent", None)
                if a is not None and getattr(a, "name", None) == name:
                    return r, "worker"
        elif kind == "swarm":
            agents = getattr(r, "agents", {}) or {}
            if name in agents:
                return r, "swarm-peer"
    return None, None


def _system_prompt_text(agent: Any) -> str | None:
    """Return the agent's system_prompt as a string, or None if dynamic."""
    p = getattr(agent, "system_prompt", None)
    if isinstance(p, str):
        return p or None
    return None


_VARIABLE_RE = re.compile(r"\{\{(\w+)\}\}")


def _extract_prompts_from_agent(agent: Any) -> list[dict[str, Any]]:
    """Build the prompts list for an agent.

    Sprint 2 spec calls for "prompts the agent references." The most
    discoverable signal is the system prompt's variables — if the system
    prompt has ``{{name}}`` placeholders, surface them so the dependency
    graph mirrors how the prompt would render at run time. We don't (yet)
    reach into the prompt registry to attach a ``version`` because the
    Agent doesn't carry a slug — that's a future enhancement.
    """
    text = _system_prompt_text(agent)
    if text is None:
        return []
    variables = sorted(set(_VARIABLE_RE.findall(text)))
    return [
        {
            "name": "system_prompt",
            "version": None,
            "variables": variables,
            "preview": text[:160],
        }
    ]


def _agent_tools_payload(
    agent: Any, used_by_name: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build the tools list for one agent, merging registered + used stats."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in getattr(agent, "tools", []) or []:
        nm = getattr(t, "name", None) or "?"
        seen.add(nm)
        u = used_by_name.get(nm, {})
        out.append(
            {
                "name": nm,
                "origin": getattr(t, "origin", "unknown"),
                "registered": True,
                "calls": int(u.get("call_count") or 0),
                "success_rate": float(u.get("success_rate") or 0.0),
                "avg_latency_ms": float(u.get("avg_latency_ms") or 0.0),
            }
        )
    # Tools the LLM hallucinated — called by name without being registered.
    for nm, u in used_by_name.items():
        if nm in seen:
            continue
        out.append(
            {
                "name": nm,
                "origin": u.get("origin") or "unknown",
                "registered": False,
                "calls": int(u.get("call_count") or 0),
                "success_rate": float(u.get("success_rate") or 0.0),
                "avg_latency_ms": float(u.get("avg_latency_ms") or 0.0),
            }
        )
    return out


def _agent_kbs_payload(agent: Any, kb_root: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in getattr(agent, "tools", []) or []:
        if getattr(t, "origin", None) == "kb":
            kb = _detect_runtime_kb(t, kb_root)
            if kb is not None:
                out.append(kb)
    return out


def _agent_guardrails_payload(agent: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in getattr(agent, "guardrails", []) or []:
        try:
            d = g.to_dict()
        except Exception:  # noqa: BLE001
            d = {"name": getattr(g, "name", "guardrail")}
        out.append(
            {
                "name": d.get("name"),
                "guardrail_type": d.get("guardrail_type"),
                "position": d.get("position"),
            }
        )
    return out


def _agent_model_payload(agent: Any) -> dict[str, Any]:
    llm = getattr(agent, "llm", None)
    return {
        "provider": getattr(llm, "provider", None),
        "model": getattr(llm, "model", None),
    }


def _used_tools_for_agent(
    db: Any, name: str, project_id: str
) -> dict[str, dict[str, Any]]:
    """Return ``{tool_name: stats}`` of tools called under this agent's spans.

    Mirrors the aggregation in ``get_agent_tools()`` so the registered/used
    semantics match the existing Agent → Tools tab.
    """
    from datetime import datetime

    pid_clause = "AND project_id = ?" if project_id else ""
    pid_params: tuple = (project_id,) if project_id else ()

    trace_rows = db.fetchall(
        f"SELECT DISTINCT trace_id FROM spans WHERE name = ? {pid_clause}",
        (f"agent.{name}", *pid_params),
    )
    trace_ids = [r["trace_id"] for r in trace_rows if r.get("trace_id")]
    out: dict[str, dict[str, Any]] = {}
    if not trace_ids:
        return out
    placeholders = ",".join("?" * len(trace_ids))
    rows = db.fetchall(
        f"""SELECT name, status, start_time, end_time, attributes
            FROM spans
            WHERE trace_id IN ({placeholders})
              AND name LIKE 'tool.%' {pid_clause}""",
        (*trace_ids, *pid_params),
    )
    for span in rows:
        try:
            attrs = json.loads(span.get("attributes") or "{}")
        except json.JSONDecodeError:
            attrs = {}
        nm = (
            attrs.get("tool.name")
            or attrs.get("fastaiagent.tool.name")
            or (span["name"][len("tool.") :] if span["name"] else "?")
        )
        bucket = out.setdefault(
            nm,
            {
                "name": nm,
                "origin": attrs.get("tool.origin")
                or attrs.get("fastaiagent.tool.origin")
                or "unknown",
                "call_count": 0,
                "error_count": 0,
                "total_duration_ms": 0,
            },
        )
        bucket["call_count"] += 1
        if attrs.get("tool.status") == "error" or span.get("status") == "ERROR":
            bucket["error_count"] += 1
        try:
            a_time = datetime.fromisoformat(span["start_time"])
            b_time = datetime.fromisoformat(span["end_time"])
            bucket["total_duration_ms"] += int(
                (b_time - a_time).total_seconds() * 1000
            )
        except (ValueError, TypeError):
            pass
    # Derive success_rate / avg_latency_ms post-hoc.
    for b in out.values():
        runs = b["call_count"] or 1
        b["success_rate"] = (b["call_count"] - b["error_count"]) / runs
        b["avg_latency_ms"] = b["total_duration_ms"] / runs
    return out


def _build_agent_payload(
    agent: Any,
    db: Any,
    project_id: str,
    kb_root: str | None,
) -> dict[str, Any]:
    """Build a single-agent dependency payload (no recursion into sub-agents).

    Used as the building block both for the top-level agent and for each
    sub-agent of a Supervisor or Swarm. Sub-agent recursion is one level
    only — that's what the UI renders cleanly; deeper trees would require
    a different visualisation.
    """
    name = getattr(agent, "name", "")
    used_by_name = _used_tools_for_agent(db, name, project_id)
    return {
        "agent": {
            "name": name,
            "type": "agent",
            "model": getattr(getattr(agent, "llm", None), "model", None),
            "provider": getattr(getattr(agent, "llm", None), "provider", None),
        },
        "tools": _agent_tools_payload(agent, used_by_name),
        "knowledge_bases": _agent_kbs_payload(agent, kb_root),
        "prompts": _extract_prompts_from_agent(agent),
        "guardrails": _agent_guardrails_payload(agent),
        "model": _agent_model_payload(agent),
    }


def _degraded_from_spans(
    db: Any, name: str, project_id: str
) -> dict[str, Any]:
    """Reconstruct a minimal payload from span attributes only.

    Used when no runner is registered for the requested agent — the UI
    still needs *something* to render. We re-use the existing
    ``agent.tools`` / ``agent.llm.*`` span attributes from the most-recent
    agent span.
    """
    from fastaiagent.ui.attrs import attr

    pid_clause = "AND project_id = ?" if project_id else ""
    pid_params: tuple = (project_id,) if project_id else ()
    rows = db.fetchall(
        f"SELECT attributes FROM spans WHERE name = ? {pid_clause} "
        "ORDER BY start_time DESC LIMIT 1",
        (f"agent.{name}", *pid_params),
    )
    attrs: dict[str, Any] = {}
    if rows:
        try:
            attrs = json.loads(rows[0].get("attributes") or "{}")
        except json.JSONDecodeError:
            attrs = {}
    raw = attr(attrs, "agent.tools") or "[]"
    parsed: list[dict[str, Any]] = []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = []
    used = _used_tools_for_agent(db, name, project_id)
    used_by_name = {u["name"]: u for u in used.values()}

    tools = []
    seen: set[str] = set()
    for t in parsed:
        nm = t.get("name") or "?"
        seen.add(nm)
        u = used_by_name.get(nm, {})
        tools.append(
            {
                "name": nm,
                "origin": t.get("origin") or "unknown",
                "registered": True,
                "calls": int(u.get("call_count") or 0),
                "success_rate": float(u.get("success_rate") or 0.0),
                "avg_latency_ms": float(u.get("avg_latency_ms") or 0.0),
            }
        )
    for nm, u in used_by_name.items():
        if nm in seen:
            continue
        tools.append(
            {
                "name": nm,
                "origin": u.get("origin") or "unknown",
                "registered": False,
                "calls": int(u.get("call_count") or 0),
                "success_rate": float(u.get("success_rate") or 0.0),
                "avg_latency_ms": float(u.get("avg_latency_ms") or 0.0),
            }
        )

    raw_g = attr(attrs, "agent.guardrails") or "[]"
    guardrails: list[dict[str, Any]] = []
    if isinstance(raw_g, str):
        try:
            for g in json.loads(raw_g) or []:
                guardrails.append(
                    {
                        "name": g.get("name"),
                        "guardrail_type": g.get("guardrail_type"),
                        "position": g.get("position"),
                    }
                )
        except json.JSONDecodeError:
            pass

    provider = attr(attrs, "agent.llm.provider")
    model = attr(attrs, "agent.llm.model")
    return {
        "agent": {
            "name": name,
            "type": "agent",
            "model": model,
            "provider": provider,
        },
        "tools": tools,
        "knowledge_bases": [],
        "prompts": [],
        "guardrails": guardrails,
        "model": {"provider": provider, "model": model},
        "unresolved": True,
    }


@router.get("/{name}/dependencies")
def get_agent_dependencies(
    request: Request,
    name: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Return the structural dependency graph of one agent.

    Walks ``ctx.runners`` to find the live Agent. For Supervisor parents,
    appends the worker subtree. For Swarm parents, appends peer agents
    with handoff edges. When no runner is registered, falls back to a
    degraded payload reconstructed from span attributes (`unresolved=True`).
    """
    from fastaiagent._internal.config import get_config

    ctx = get_context(request)
    db = ctx.db()
    try:
        # 404 cross-project lookups so probes can't confirm an agent in
        # another project even exists.
        if ctx.project_id:
            probe = db.fetchone(
                "SELECT 1 FROM spans WHERE name = ? AND project_id = ? LIMIT 1",
                (f"agent.{name}", ctx.project_id),
            )
            if probe is None and not _find_agent_in_runners(ctx.runners, name)[0]:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND, f"Agent '{name}' not found"
                )

        kb_root = getattr(get_config(), "kb_dir", None)

        runner, parent_kind = _find_agent_in_runners(ctx.runners, name)
        if runner is None:
            return _degraded_from_spans(db, name, ctx.project_id)

        # Locate the actual Agent object inside the runner.
        if parent_kind == "agent":
            agent = runner
            payload = _build_agent_payload(agent, db, ctx.project_id, kb_root)
            payload["sub_agents"] = []
            return payload

        if parent_kind == "worker":
            # Requested name is one of the supervisor's workers — render
            # that worker's deps. The supervisor itself is a different
            # /agents/{supervisor_name}/dependencies request.
            for w in runner.workers:
                if getattr(w.agent, "name", None) == name:
                    payload = _build_agent_payload(
                        w.agent, db, ctx.project_id, kb_root
                    )
                    payload["sub_agents"] = []
                    payload["parent"] = {
                        "name": getattr(runner, "name", None),
                        "type": "supervisor",
                    }
                    return payload

        if parent_kind == "swarm-peer":
            # Requested name is one of a swarm's peers. Render its own deps,
            # plus a flat list of peer names + handoff allowlist so the UI
            # can render the handoff edges.
            agents = runner.agents
            agent = agents[name]
            payload = _build_agent_payload(agent, db, ctx.project_id, kb_root)
            handoffs = getattr(runner, "handoffs", None) or {
                src: [d for d in agents if d != src] for src in agents
            }
            payload["peers"] = [
                {
                    "name": peer_name,
                    "type": "agent",
                    "model": getattr(getattr(p, "llm", None), "model", None),
                    "provider": getattr(getattr(p, "llm", None), "provider", None),
                }
                for peer_name, p in agents.items()
                if peer_name != name
            ]
            payload["handoffs"] = [
                {"from": src, "to": dst}
                for src, targets in handoffs.items()
                for dst in targets
            ]
            payload["parent"] = {
                "name": getattr(runner, "name", None),
                "type": "swarm",
            }
            payload["sub_agents"] = []
            return payload

        # Supervisor itself — the requested name matches a Supervisor.name.
        # Render the supervisor's own deps plus each worker's subtree.
        if parent_kind == "supervisor":
            # Supervisor.llm + Supervisor.system_prompt give us tool-less
            # deps, but the Supervisor itself doesn't expose .tools — it
            # synthesises tools from workers. Render an empty tools list
            # and surface workers as sub_agents.
            payload = {
                "agent": {
                    "name": name,
                    "type": "supervisor",
                    "model": getattr(getattr(runner, "llm", None), "model", None),
                    "provider": getattr(
                        getattr(runner, "llm", None), "provider", None
                    ),
                },
                "tools": [],
                "knowledge_bases": [],
                "prompts": _extract_prompts_from_agent(runner),
                "guardrails": [],
                "model": _agent_model_payload(runner),
                "sub_agents": [],
            }
            for w in runner.workers:
                sub = _build_agent_payload(w.agent, db, ctx.project_id, kb_root)
                sub["agent"]["type"] = "worker"
                sub["role"] = w.role
                payload["sub_agents"].append(sub)
            return payload

        # Last-resort fallthrough — should be unreachable.
        return _degraded_from_spans(db, name, ctx.project_id)
    finally:
        db.close()
