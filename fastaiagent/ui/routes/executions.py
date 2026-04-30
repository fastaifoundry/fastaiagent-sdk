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


def _checkpoint_row_to_dict(cp: Any) -> dict[str, Any]:
    """Pydantic ``Checkpoint`` → JSON-safe dict for the UI."""
    return {
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
    store = _open_store(ctx.db_path)
    try:
        rows = store.list(execution_id, limit=500)
        latest = store.get_last(execution_id)
        if latest is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Execution not found")
        return {
            "execution_id": execution_id,
            "chain_name": latest.chain_name,
            "status": latest.status,
            "agent_path": latest.agent_path,
            "checkpoint_count": len(rows),
            "latest_checkpoint_id": latest.checkpoint_id,
            "latest_state_snapshot": latest.state_snapshot,
            "checkpoints": [_checkpoint_row_to_dict(cp) for cp in rows],
        }
    finally:
        store.close()


@router.get("/pending-interrupts")
def list_pending_interrupts(
    request: Request,
    _user: str = Depends(require_session),
    limit: int = 100,
) -> dict[str, Any]:
    """Rows that ``/approvals`` renders (one per suspended workflow)."""
    ctx = get_context(request)
    store = _open_store(ctx.db_path)
    try:
        rows = store.list_pending_interrupts(limit=limit)
        return {
            "count": len(rows),
            "items": [
                {
                    "execution_id": r.execution_id,
                    "chain_name": r.chain_name,
                    "node_id": r.node_id,
                    "reason": r.reason,
                    "context": r.context,
                    "agent_path": r.agent_path,
                    "created_at": r.created_at,
                }
                for r in rows
            ],
        }
    finally:
        store.close()


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
    ``ChainResult`` / ``AgentResult`` dict plus the original ``execution_id``.
    """
    ctx = get_context(request)
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

    resume_value = Resume(approved=body.approved, metadata=body.metadata)
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
        "reason": body.reason,
    }
