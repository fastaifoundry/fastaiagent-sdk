"""Agent Replay endpoints — wraps Replay / ForkedReplay with server-side fork state."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from fastaiagent._internal.errors import ReplayError
from fastaiagent.multimodal.format import resolve_wire_markers as _resolve_modify_input
from fastaiagent.trace.replay import Replay
from fastaiagent.trace.storage import TraceStore
from fastaiagent.ui.deps import get_context, require_session

router = APIRouter(prefix="/api/replay", tags=["replay"])

# Server-side cache of fork state, keyed by fork_id. One UI is one process, so
# in-memory is fine.
_forks: dict[str, Any] = {}
_forks_lock = threading.Lock()


def _register_fork(forked: Any) -> str:
    fork_id = uuid.uuid4().hex
    with _forks_lock:
        _forks[fork_id] = forked
    return fork_id


def _get_fork(fork_id: str) -> Any:
    with _forks_lock:
        if fork_id not in _forks:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Fork '{fork_id}' not found")
        return _forks[fork_id]


def _replay_for(db_path: str, trace_id: str) -> Replay:
    store = TraceStore(db_path=db_path)
    try:
        return Replay.load(trace_id, store=store)
    except Exception as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Trace '{trace_id}' not found: {e}") from e


@router.get("/{trace_id}")
def load_replay(
    request: Request,
    trace_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    replay = _replay_for(ctx.db_path, trace_id)
    return {"steps": [s.model_dump() for s in replay.steps()]}


class ForkRequest(BaseModel):
    step: int


@router.post("/{trace_id}/fork")
def fork(
    request: Request,
    trace_id: str,
    body: ForkRequest,
    _user: str = Depends(require_session),
) -> dict[str, str]:
    ctx = get_context(request)
    replay = _replay_for(ctx.db_path, trace_id)
    try:
        forked = replay.fork_at(body.step)
    except ReplayError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    fork_id = _register_fork(forked)
    return {"fork_id": fork_id}


class ModifyRequest(BaseModel):
    """Modifications to apply to a forked replay before rerun.

    ``input`` accepts the same shapes as :py:meth:`Agent.run`:

    * ``str`` — text-only replacement
    * ``dict`` — legacy ``{"input": "..."}`` form
    * ``list[dict]`` — multimodal: each part is one of
      ``{"type": "text", "text": "..."}``,
      ``{"type": "image", "data_base64": "...", "media_type": "image/jpeg"}``,
      ``{"type": "pdf", "data_base64": "..."}``

    ``tool_overrides`` (v1.14.1+) maps tool name → canned response. Each
    entry installs a stub :class:`FunctionTool` via
    :meth:`ForkedReplay.with_tool_override` so the next rerun's LLM
    receives the canned value in place of calling the original tool.

    ``tool_response`` is the deprecated v1.13 field. It silently
    dropped its payload for agent reruns (it was routed to
    ``modify_state`` which the agent path never reads). For backwards
    compatibility we accept it but log a warning and route nothing —
    callers should migrate to ``tool_overrides``.
    """

    prompt: str | None = None
    input: dict[str, Any] | list[dict[str, Any]] | str | None = None
    tool_overrides: dict[str, Any] | None = None
    tool_response: dict[str, Any] | None = None  # deprecated, see docstring
    config: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


def _build_stub_tool(name: str, response: Any) -> Any:
    """Build a :class:`FunctionTool` that ignores its args and returns
    ``response`` verbatim. Used by the UI route to wire the
    "Tool response override" UX through to
    :meth:`ForkedReplay.with_tool_override`.
    """
    from fastaiagent import FunctionTool

    def _stub(**_kwargs: Any) -> Any:
        return response

    _stub.__name__ = f"stubbed_{name}"
    _stub.__doc__ = f"Stubbed override for tool {name!r}. Returns the canned response."
    return FunctionTool(name=name, fn=_stub)


@router.patch("/forks/{fork_id}")
def modify_fork(
    fork_id: str,
    body: ModifyRequest,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    import logging

    log = logging.getLogger(__name__)

    forked = _get_fork(fork_id)
    deprecation_warnings: list[str] = []

    if body.prompt is not None:
        forked.modify_prompt(body.prompt)
    if body.input is not None:
        forked.modify_input(_resolve_modify_input(body.input))
    if body.tool_overrides:
        for tool_name, response_value in body.tool_overrides.items():
            forked.with_tool_override(tool_name, _build_stub_tool(tool_name, response_value))
    if body.tool_response is not None:
        # Deprecated: pre-v1.14.1 the UI sent this and the route silently
        # dropped it (routed to modify_state which agent rerun ignores).
        # Surface the breakage so callers migrate to ``tool_overrides``.
        msg = (
            "ModifyRequest.tool_response is deprecated and was a no-op for agent "
            "reruns. Use tool_overrides={'<tool_name>': <canned_response>} instead."
        )
        log.warning(msg)
        deprecation_warnings.append(msg)
    if body.config is not None:
        forked.modify_config(**body.config)
    if body.state is not None:
        forked.modify_state(body.state)

    response: dict[str, Any] = {"fork_id": fork_id, "status": "modified"}
    if deprecation_warnings:
        response["deprecation_warnings"] = deprecation_warnings
    return response


@router.post("/forks/{fork_id}/rerun")
async def rerun_fork(
    fork_id: str,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    forked = _get_fork(fork_id)
    try:
        result = await forked.arerun()
    except ReplayError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    return {
        "fork_id": fork_id,
        "new_trace_id": result.trace_id,
        "new_output": result.new_output,
        "original_output": result.original_output,
        "steps_executed": result.steps_executed,
    }


@router.get("/forks/{fork_id}/compare")
def compare_fork(
    fork_id: str,
    against: str,
    request: Request,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    ctx = get_context(request)
    forked = _get_fork(fork_id)
    store = TraceStore(db_path=ctx.db_path)
    try:
        new_trace = store.get_trace(against)
    except Exception as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Trace '{against}' not found: {e}") from e
    # Build a ReplayResult-like object so compare() can reload the new trace.
    from fastaiagent.trace.replay import ReplayResult

    res = ReplayResult(
        original_output=None,
        new_output=None,
        trace_id=new_trace.trace_id,
        steps_executed=len(new_trace.spans),
    )
    comparison = forked.compare(res)
    dumped: dict[str, Any] = comparison.model_dump()
    return dumped


class SaveTestRequest(BaseModel):
    dataset_path: str | None = None
    input: str
    expected_output: str
    # v1.14.1+: the rerun's trace_id (the *fixed* run). The original
    # failure's id is read from the fork's source trace automatically.
    # Kept named ``trace_id`` for backwards compat with v1.13/v1.14.0
    # callers.
    trace_id: str | None = None
    fork_step: int | None = None
    modifications: dict[str, Any] | None = None


@router.post("/forks/{fork_id}/save-as-test")
def save_as_test(
    fork_id: str,
    body: SaveTestRequest,
    request: Request,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    forked = _get_fork(fork_id)
    ctx = get_context(request)
    db_dir = Path(ctx.db_path).parent
    path = Path(body.dataset_path) if body.dataset_path else (db_dir / "regression_tests.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    # v1.14.1 schema: trace_id = the rerun's id (the fixed run), and
    # source_trace_id = the failure that motivated this regression case.
    # Pre-v1.14.1 the route wrote source_trace_id INTO the trace_id field,
    # losing the link to the actual rerun — fixed here.
    source_trace_id = getattr(getattr(forked, "_trace", None), "trace_id", None)
    fixed_trace_id = body.trace_id  # the rerun's id, supplied by the UI
    fork_step = (
        body.fork_step if body.fork_step is not None else getattr(forked, "_fork_point", None)
    )
    record: dict[str, Any] = {
        "input": body.input,
        "expected_output": body.expected_output,
        "trace_id": fixed_trace_id,
        "source_trace_id": source_trace_id,
        "fixed_trace_id": fixed_trace_id,
        "fork_step": fork_step,
        "modifications": body.modifications or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return {"path": str(path)}


# Test helper — used by the test suite to drain stale forks.
def _clear_forks_for_tests() -> None:
    with _forks_lock:
        _forks.clear()


__all__ = ["router", "_clear_forks_for_tests"]
