"""Agent Replay endpoints — wraps Replay / ForkedReplay with server-side fork state."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from fastaiagent._internal.errors import ReplayError
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
    prompt: str | None = None
    input: dict[str, Any] | None = None
    tool_response: dict[str, Any] | None = None
    config: dict[str, Any] | None = None
    state: dict[str, Any] | None = None


@router.patch("/forks/{fork_id}")
def modify_fork(
    fork_id: str,
    body: ModifyRequest,
    _user: str = Depends(require_session),
) -> dict[str, str]:
    forked = _get_fork(fork_id)
    if body.prompt is not None:
        forked.modify_prompt(body.prompt)
    if body.input is not None:
        forked.modify_input(body.input)
    if body.tool_response is not None:
        forked.modify_state({"tool_response": body.tool_response})
    if body.config is not None:
        forked.modify_config(**body.config)
    if body.state is not None:
        forked.modify_state(body.state)
    return {"fork_id": fork_id, "status": "modified"}


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


@router.post("/forks/{fork_id}/save-as-test")
def save_as_test(
    fork_id: str,
    body: SaveTestRequest,
    request: Request,
    _user: str = Depends(require_session),
) -> dict[str, Any]:
    # Ensure the fork exists (404 otherwise).
    _get_fork(fork_id)
    ctx = get_context(request)
    db_dir = Path(ctx.db_path).parent
    path = Path(body.dataset_path) if body.dataset_path else (db_dir / "regression_tests.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"input": body.input, "expected_output": body.expected_output}) + "\n")
    return {"path": str(path)}


# Test helper — used by the test suite to drain stale forks.
def _clear_forks_for_tests() -> None:
    with _forks_lock:
        _forks.clear()


__all__ = ["router", "_clear_forks_for_tests"]
