"""HTTP resume endpoints — the v1.0 ``/approvals`` UI talks to these.

Three endpoints:

- ``GET  /api/executions/{execution_id}`` — full checkpoint history for an
  execution (used by the ``/executions/:id`` detail page).
- ``POST /api/executions/{execution_id}/resume`` — body
  ``{"approved": bool, "metadata": {...}, "reason": "?"}``. Looks up the
  ``chain_name`` from the latest checkpoint, finds the matching runner in
  ``app.state.context.runners``, and calls its ``aresume(...)`` with a
  :class:`Resume` value.
- ``GET  /api/pending-interrupts`` — list of pending rows for the
  ``/approvals`` table.

Per Phase 9's open question (a): the FastAPI app must be built with an
explicit list of runners (``build_app(runners=[chain1, agent2, ...])``)
so the server can re-instantiate the right object on resume. If no runner
is registered for a checkpoint's ``chain_name``, resume returns 503 with
a clear message.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from fastaiagent.checkpointers import SQLiteCheckpointer
from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api", tags=["executions"])


class ResumeBody(BaseModel):
    """Body shape for ``POST /api/executions/{id}/resume``."""

    approved: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None


def _open_store(db_path: str) -> SQLiteCheckpointer:
    cp = SQLiteCheckpointer(db_path=db_path)
    cp.setup()
    return cp


def _replication_state(db: Any, execution_id: str) -> dict[str, dict[str, Any]]:
    """``{checkpoint_id: {"synced", "sync_error"}}`` for one run's checkpoints.

    Read straight from the row rather than through :class:`Checkpoint`, because
    these two are **outbox** columns, not run state — putting them on the model
    would send them over the replication wire and back through
    ``restore_from_plane``, where they mean nothing.

    Worth surfacing at all because of durability audit D2: a checkpoint the
    plane refuses on payload grounds is now quarantined rather than re-sent
    forever, and without this the only trace of that is a log line the operator
    has probably already scrolled past. ``synced=1`` with a ``sync_error`` reads
    as "we gave up on this one, and here is what the plane said".
    """
    try:
        rows = db.fetchall(
            "SELECT checkpoint_id, synced, sync_error FROM checkpoints WHERE execution_id = ?",
            (execution_id,),
        )
    except Exception:
        # An older local.db (pre-v19) has no ``sync_error``. The inspector must
        # still render — replication state is an extra, never a precondition.
        return {}
    return {
        r["checkpoint_id"]: {"synced": bool(r["synced"]), "sync_error": r["sync_error"]}
        for r in rows
        if r["checkpoint_id"]
    }


#: Root-span attributes that link a trace to an execution (decision D7: every
#: resume is its own trace, tied back by id rather than by parentage).
_EXECUTION_ID_KEYS = (
    "agent.execution_id",
    "agent.resumed_execution_id",
    "chain.execution_id",
    "chain.resumed_execution_id",
    "swarm.execution_id",
    "swarm.resumed_execution_id",
    "supervisor.execution_id",
    "supervisor.resumed_execution_id",
)
_RESUMED_KEYS = (
    "agent.resumed_execution_id",
    "chain.resumed_execution_id",
    "swarm.resumed_execution_id",
    "supervisor.resumed_execution_id",
)
_OUTPUT_KEYS = ("agent.output", "chain.output", "swarm.output", "supervisor.output")
#: The first checkpoint is written after the run's root span opened (the span
#: wraps the guardrails and the first model call), so the span scan starts this
#: far before it. The scan is a bound on work, not on correctness: execution
#: ids are matched exactly.
_RUN_SCAN_MARGIN = timedelta(hours=24)
_OUTPUT_MAX = 2000


def _pending_row(db: Any, execution_id: str) -> dict[str, Any] | None:
    """The run's ``pending_interrupts`` row, or ``None`` when nothing waits."""
    row = db.fetchone(
        """SELECT execution_id, chain_name, node_id, reason, context, agent_path, created_at
           FROM pending_interrupts WHERE execution_id = ? LIMIT 1""",
        (execution_id,),
    )
    if row is None:
        return None
    raw = row["context"]
    try:
        context = json.loads(raw or "{}") if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        context = raw
    return {
        "execution_id": row["execution_id"],
        "chain_name": row["chain_name"],
        "node_id": row["node_id"],
        "reason": row["reason"],
        "context": context,
        "agent_path": row["agent_path"],
        "created_at": row["created_at"],
    }


def _interrupt_states(rows: list[Any], pending: dict[str, Any] | None) -> list[str | None]:
    """Per checkpoint: ``"pending"`` / ``"resolved"`` for an ``interrupted`` row.

    An ``interrupted`` row is still waiting only while a pending row exists AND
    nothing ran after it on the same ``agent_path`` — a later step there means
    someone resumed it. Until 1.89.0 the UI showed every ``interrupted`` row as
    "waiting", forever, because it had nothing else to go on.
    """
    from fastaiagent.chain.checkpoint import is_run_end

    states: list[str | None] = []
    for i, cp in enumerate(rows):
        if cp.status != "interrupted":
            states.append(None)
            continue
        ran_after = any(
            (later.agent_path or "") == (cp.agent_path or "") and not is_run_end(later)
            for later in rows[i + 1 :]
        )
        states.append("pending" if pending is not None and not ran_after else "resolved")
    return states


def _scan_start(rows: list[Any]) -> str:
    first = min((cp.created_at for cp in rows if cp.created_at), default="")
    try:
        return (datetime.fromisoformat(first) - _RUN_SCAN_MARGIN).isoformat()
    except ValueError:
        return ""


def _runs(db: Any, execution_id: str, since: str) -> list[dict[str, Any]]:
    """Every trace that ran this execution — the run, each resume, a fork.

    One entry per trace. A Swarm or Supervisor nests agents that carry the same
    id, so the topmost matching span of each trace speaks for it; the resume
    details are read from whichever span in that trace carries them.
    """
    clauses = " OR ".join(f"json_extract(attributes, '$.\"{k}\"') = ?" for k in _EXECUTION_ID_KEYS)
    try:
        rows = db.fetchall(
            f"""SELECT trace_id, span_id, parent_span_id, name, start_time, end_time,
                       status, attributes
                FROM spans WHERE start_time >= ? AND ({clauses})
                ORDER BY start_time ASC""",
            (since, *([execution_id] * len(_EXECUTION_ID_KEYS))),
        )
    except Exception:
        # No spans table yet (tracing never ran here) — the page still renders.
        return []
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        try:
            attrs = json.loads(r["attributes"] or "{}")
        except json.JSONDecodeError:
            attrs = {}
        by_trace.setdefault(r["trace_id"], []).append({**r, "attrs": attrs})

    runs: list[dict[str, Any]] = []
    for trace_id, spans in by_trace.items():
        ids = {s["span_id"] for s in spans}
        top = next((s for s in spans if s["parent_span_id"] not in ids), spans[0])
        attrs = top["attrs"]
        merged: dict[str, Any] = {}
        for s in spans:
            for k, v in s["attrs"].items():
                merged.setdefault(k, v)
        if any(merged.get(k) == execution_id for k in _RESUMED_KEYS):
            role = "resume"
        elif merged.get("agent.forked_execution_id") == execution_id:
            role = "fork"
        else:
            role = "run"
        resume: dict[str, Any] | None = None
        for prefix in ("agent.resume.", "chain.resume."):
            if f"{prefix}approved" in merged or f"{prefix}node_id" in merged:
                resume = {
                    "approved": merged.get(f"{prefix}approved"),
                    "resolver": merged.get(f"{prefix}resolver"),
                    "reason": merged.get(f"{prefix}reason"),
                    "node_id": merged.get(f"{prefix}node_id"),
                    "kind": merged.get(f"{prefix}kind"),
                }
                break
        if (top.get("status") or "").upper() == "ERROR":
            run_status = "failed"
        else:
            run_status = str(attrs.get("agent.status") or attrs.get("chain.status") or "completed")
        output = next((attrs[k] for k in _OUTPUT_KEYS if attrs.get(k) is not None), None)
        if isinstance(output, str) and len(output) > _OUTPUT_MAX:
            output = output[:_OUTPUT_MAX]
        runs.append(
            {
                "trace_id": trace_id,
                "role": role,
                "name": top.get("name"),
                "started_at": top.get("start_time"),
                "ended_at": top.get("end_time"),
                "status": run_status,
                "output": output,
                "resume": resume,
            }
        )
    runs.sort(key=lambda r: r["started_at"] or "")
    return runs


def _hitl_resolution(db: Any, execution_id: str) -> dict[str, Any] | None:
    """The newest ``resolved`` HITL row — recorded only when connected."""
    try:
        row = db.fetchone(
            """SELECT status, resolver, node, occurred_at FROM hitl_events
               WHERE run_id = ? AND event_type = 'resolved'
               ORDER BY occurred_at DESC, created_at DESC LIMIT 1""",
            (execution_id,),
        )
    except Exception:
        return None
    if row is None:
        return None
    return {
        "approved": row["status"] == "approved",
        "resolver": row["resolver"],
        "reason": None,
        "node_id": row["node"],
        "at": row["occurred_at"],
        "trace_id": None,
        "source": "hitl_events",
    }


def _checkpoint_row_to_dict(cp: Any, replication: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pydantic ``Checkpoint`` → JSON-safe dict for the UI."""
    repl = replication.get(cp.checkpoint_id) or {}
    return {
        "synced": repl.get("synced"),
        "sync_error": repl.get("sync_error"),
        "checkpoint_id": cp.checkpoint_id,
        "parent_checkpoint_id": cp.parent_checkpoint_id,
        "chain_name": cp.chain_name,
        "execution_id": cp.execution_id,
        "node_id": cp.node_id,
        "node_index": cp.node_index,
        "status": cp.status,
        "state_snapshot": cp.state_snapshot,
        "node_input": cp.node_input,
        "node_output": cp.node_output,
        "iteration": cp.iteration,
        "iteration_counters": cp.iteration_counters,
        "interrupt_reason": cp.interrupt_reason,
        "interrupt_context": cp.interrupt_context,
        "agent_path": cp.agent_path,
        "created_at": cp.created_at,
    }


@router.get("/executions/{execution_id}")
def get_execution(
    execution_id: str,
    request: Request,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Return the full checkpoint history for one execution."""
    ctx = get_context(request)
    # Project-scope guard: 404 cross-project executions before opening the store.
    if ctx.project_id:
        probe = ctx.db()
        try:
            row = probe.fetchone(
                "SELECT 1 FROM checkpoints WHERE execution_id = ? AND project_id = ? LIMIT 1",
                (execution_id, ctx.project_id),
            )
        finally:
            probe.close()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    store = _open_store(ctx.db_path)
    try:
        rows = store.list(execution_id, limit=500)
        latest = store.get_last(execution_id)
        if latest is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
        repl_db = ctx.db()
        try:
            replication = _replication_state(repl_db, execution_id)
            pending = _pending_row(repl_db, execution_id)
            runs = _runs(repl_db, execution_id, _scan_start(rows))
            hitl = _hitl_resolution(repl_db, execution_id)
        finally:
            repl_db.close()
        quarantined = sum(1 for v in replication.values() if v.get("sync_error"))
        # The newest run is the one still waiting when a pause outlived it.
        if (
            runs
            and pending is not None
            and (pending["created_at"] or "") >= (runs[-1]["started_at"] or "")
        ):
            if runs[-1]["status"] == "completed":
                runs[-1]["status"] = "paused"
        resolution: dict[str, Any] | None = None
        for run in reversed(runs):
            if run["resume"] is not None and run["resume"].get("approved") is not None:
                resolution = {
                    **run["resume"],
                    "at": run["started_at"],
                    "trace_id": run["trace_id"],
                    "source": "trace",
                }
                break
        if resolution is None:
            resolution = hitl
        final = next((r for r in reversed(runs) if r["status"] == "completed"), None)
        states = _interrupt_states(rows, pending)
        checkpoints = []
        for cp, state in zip(rows, states):
            row = _checkpoint_row_to_dict(cp, replication)
            row["interrupt_state"] = state
            checkpoints.append(row)
        return {
            "execution_id": execution_id,
            "chain_name": latest.chain_name,
            "status": latest.status,
            "agent_path": latest.agent_path,
            "checkpoint_count": len(rows),
            "latest_checkpoint_id": latest.checkpoint_id,
            "latest_state_snapshot": latest.state_snapshot,
            # >0 means the plane's copy of this run is incomplete (audit D2).
            "quarantined_count": quarantined,
            "checkpoints": checkpoints,
            # What the page needs to say what happened, not only what is stored
            # (1.89.0): the pause still waiting (or null), every trace that ran
            # this execution, who resolved the pause and how, and the answer.
            "pending_interrupt": pending,
            "runs": runs,
            "resolution": resolution,
            "final_output": final["output"] if final is not None else None,
        }
    finally:
        store.close()


@router.get("/executions/{execution_id}/idempotency-cache")
def list_idempotency_cache(
    execution_id: str,
    request: Request,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Cached ``@idempotent`` results for one execution.

    Helps developers verify their ``@idempotent`` decorator works — these
    are the calls that would be skipped on resume.
    """
    ctx = get_context(request)
    db = ctx.db()
    try:
        if ctx.project_id:
            rows = db.fetchall(
                "SELECT function_key, result, created_at "
                "FROM idempotency_cache "
                "WHERE execution_id = ? AND project_id = ? "
                "ORDER BY created_at ASC",
                (execution_id, ctx.project_id),
            )
        else:
            rows = db.fetchall(
                "SELECT function_key, result, created_at "
                "FROM idempotency_cache WHERE execution_id = ? "
                "ORDER BY created_at ASC",
                (execution_id,),
            )
        from fastaiagent.chain.idempotent import _unwrap

        items = []
        for r in rows:
            try:
                # The value inside the 1.92.0 envelope, or a bare older row.
                result = _unwrap(json.loads(r.get("result") or "null"))[1]
            except json.JSONDecodeError:
                result = r.get("result")
            items.append(
                {
                    "function_key": r.get("function_key"),
                    "result": result,
                    "created_at": r.get("created_at"),
                }
            )
        return {"execution_id": execution_id, "count": len(items), "items": items}
    finally:
        db.close()


@router.get("/pending-interrupts")
def list_pending_interrupts(
    request: Request,
    _user: str = Depends(require_session),
    limit: int = 100,
) -> dict[str, Any]:
    """Rows that ``/approvals`` renders (one per suspended workflow)."""
    ctx = get_context(request)
    # Direct SQL so we can scope by project_id; the checkpointer's
    # ``list_pending_interrupts`` doesn't take a filter.
    db = ctx.db()
    try:
        if ctx.project_id:
            rows = db.fetchall(
                """SELECT execution_id, chain_name, node_id, reason,
                          context, agent_path, created_at
                   FROM pending_interrupts
                   WHERE project_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (ctx.project_id, limit),
            )
        else:
            rows = db.fetchall(
                """SELECT execution_id, chain_name, node_id, reason,
                          context, agent_path, created_at
                   FROM pending_interrupts
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            )
        items = [
            {
                "execution_id": r["execution_id"],
                "chain_name": r["chain_name"],
                "node_id": r["node_id"],
                "reason": r["reason"],
                "context": json.loads(r["context"] or "{}")
                if isinstance(r["context"], str)
                else r["context"],
                "agent_path": r["agent_path"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return {"count": len(items), "items": items}
    finally:
        db.close()


@router.post("/executions/{execution_id}/resume")
async def resume_execution(
    execution_id: str,
    body: ResumeBody,
    request: Request,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    """Resume a suspended workflow with an approval decision.

    Body shape::

        {"approved": true, "metadata": {"approver": "alice"}, "reason": "..."}

    Looks up the runner by the checkpoint's ``chain_name`` in the registry
    the server was built with. Calls its ``aresume(...)``. Returns the
    ``ChainResult`` / ``AgentResult`` dict plus the original ``execution_id``,
    the resumed run's ``trace_id``, and the ``reason`` / ``resolver`` it ran with.
    A top-level ``reason`` is folded into ``metadata["reason"]``; with auth on and
    no ``metadata["resolver"]``, the signed-in user is the resolver.
    """
    ctx = get_context(request)
    # Project-scope guard: 404 cross-project resume attempts before they
    # bind to a runner. The latest checkpoint for this execution must
    # belong to the active project.
    if ctx.project_id:
        probe = ctx.db()
        try:
            row = probe.fetchone(
                "SELECT 1 FROM checkpoints WHERE execution_id = ? AND project_id = ? LIMIT 1",
                (execution_id, ctx.project_id),
            )
        finally:
            probe.close()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
    store = _open_store(ctx.db_path)
    try:
        latest = store.get_last(execution_id)
        if latest is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
        chain_name = latest.chain_name
    finally:
        store.close()

    runner = ctx.runners.get(chain_name)
    if runner is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            (
                f"No runner registered for {chain_name!r}. Pass "
                "`runners=[chain_or_agent_or_swarm_or_supervisor, ...]` to "
                "build_app() so the server can resume it."
            ),
        )

    from fastaiagent import Resume

    # The reason travels in the metadata, which is where the runner reads it (and
    # stamps it on the resumed run's span). Until 1.89.0 a top-level ``reason``
    # was echoed back in the response and dropped. And the resolver is the person
    # signed in to this UI (decision D8) — no one else could have clicked it —
    # unless the caller named one. With ``--no-auth`` there is no one to name.
    metadata = dict(body.metadata or {})
    if body.reason and not metadata.get("reason"):
        metadata["reason"] = body.reason
    if not ctx.no_auth and not metadata.get("resolver"):
        metadata["resolver"] = _user
    resume_value = Resume(approved=body.approved, metadata=metadata)
    aresume = getattr(runner, "aresume", None)
    if aresume is None:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Runner {chain_name!r} does not expose aresume()",
        )

    try:
        result = await aresume(execution_id, resume_value=resume_value)
    except Exception as exc:  # surface AlreadyResumed et al. as a clean 409
        from fastaiagent import AlreadyResumed

        if isinstance(exc, AlreadyResumed):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Execution {execution_id!r} was already resumed.",
            ) from exc
        raise

    # Coerce ChainResult / AgentResult into JSON cleanly. Both are pydantic
    # models, so model_dump produces a dict.
    if hasattr(result, "model_dump"):
        result_dict = result.model_dump(mode="json")
    else:
        # Fall back to a minimal projection for non-pydantic returns.
        result_dict = json.loads(json.dumps(result, default=str))

    return {
        "execution_id": execution_id,
        "chain_name": chain_name,
        "result": result_dict,
        "reason": metadata.get("reason"),
        "resolver": metadata.get("resolver"),
        # The resumed run's own trace (a new one per resume — decision D7).
        "trace_id": getattr(result, "trace_id", None),
    }
