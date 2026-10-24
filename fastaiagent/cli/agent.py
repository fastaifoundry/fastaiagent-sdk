"""``fastaiagent agent serve`` — run an Agent or Chain as a FastAPI service
following the uniform deployment contract documented at
``docs/deployment/fastapi.md``.

This saves you from copy-pasting the 80-line starter server into every
project. It's a thin wrapper: if you need custom routes, auth middleware,
or anything else the uniform contract doesn't cover, eject by copying
[examples/33_deploy_fastapi.py] and extending it.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import typer

try:
    # Module level so FastAPI can resolve the ``request: Request`` annotation of
    # a route (annotations are strings in this module); fastapi is an extra.
    from fastapi import Request
except ImportError:  # pragma: no cover - checked when the app is built
    Request = None  # type: ignore[assignment, misc]

try:
    from pydantic import BaseModel

    class RunRequest(BaseModel):
        input: str
        # Names the run (1.93.0), so the caller can ``POST /cancel`` it while it
        # is in progress; one is chosen when absent.
        execution_id: str | None = None

    class CancelRequest(BaseModel):
        execution_id: str

    class ResumeRequest(BaseModel):
        # The ``execution_id`` a paused ``/run`` returned; ``approved`` and
        # ``metadata`` become the ``Resume`` the paused tool receives.
        execution_id: str
        approved: bool
        metadata: dict[str, Any] = {}

    class RunResponse(BaseModel):
        output: str
        latency_ms: int = 0
        tokens_used: int = 0
        trace_id: str | None = None
        # "paused" when the run stopped for a decision — a managed approval
        # policy (the calling application approves, since 1.74.0) or an
        # interrupt(). ``output`` is then empty: the run has not finished, and
        # ``pending_interrupt`` says what it is waiting on.
        status: str = "completed"
        execution_id: str | None = None
        pending_interrupt: dict[str, Any] | None = None
except ImportError:  # pragma: no cover - pydantic is a required fastaiagent dep
    RunRequest = None  # type: ignore[assignment, misc]
    CancelRequest = None  # type: ignore[assignment, misc]
    ResumeRequest = None  # type: ignore[assignment, misc]
    RunResponse = None  # type: ignore[assignment, misc]

agent_app = typer.Typer(
    name="agent",
    help="Run an Agent or Chain as a service.",
    no_args_is_help=True,
)

# ``0.0.0.0`` / ``::`` are wildcard binds — *not* loopback. Kept local (rather
# than imported from ``cli.ui``) for the same reason ``_resolve_target`` is
# duplicated: the two subcommand groups stay import-independent.
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

# Reject request bodies larger than this by default (security_audit_2 N1 —
# ``/run`` had no size cap, so any peer could exhaust the worker's memory).
_DEFAULT_MAX_BODY_BYTES: int = 10 * 1024 * 1024  # 10 MiB


def _is_loopback_host(host: str) -> bool:
    return host.lower() in _LOOPBACK_HOSTS


def _resolve_target(spec: str) -> Any:
    """Resolve ``path/to/file.py:attr`` or ``pkg.module:attr`` into a live object.

    Deliberately duplicated with ``fastaiagent.cli.mcp._resolve_target`` — the
    two commands sit in different subcommand groups and keeping the resolver
    local avoids a cross-module import dance.
    """
    if ":" not in spec:
        raise typer.BadParameter(
            f"Expected 'path/to/file.py:attr' or 'pkg.module:attr', got {spec!r}"
        )
    module_part, attr = spec.rsplit(":", 1)
    path = Path(module_part)
    if path.exists():
        module_name = path.stem
        spec_obj = importlib.util.spec_from_file_location(module_name, str(path))
        if spec_obj is None or spec_obj.loader is None:
            raise typer.BadParameter(f"Cannot load module from {path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.path.insert(0, str(path.parent.resolve()))
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_part)
    if not hasattr(module, attr):
        raise typer.BadParameter(f"Module {module_part!r} has no attribute {attr!r}")
    return getattr(module, attr)


def _build_app(
    target: Any,
    *,
    auth_token: str | None = None,
    max_body_bytes: int = _DEFAULT_MAX_BODY_BYTES,
) -> Any:
    """Build a FastAPI app that exposes ``target`` over the uniform contract.

    ``auth_token`` (opt-in) requires ``Authorization: Bearer <token>`` on the
    execution routes — ``/health`` stays open for liveness probes. When it is
    ``None`` the routes are unauthenticated, exactly as before, so containerised
    deployments behind a gateway are unaffected. ``max_body_bytes`` caps the
    request size (best-effort, via ``Content-Length``).
    """
    try:
        import hmac
        import json

        from fastapi import Depends, FastAPI, Header, HTTPException, status
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "`fastaiagent agent serve` requires fastapi and uvicorn. "
            "Install with: pip install fastapi 'uvicorn[standard]'"
        ) from err

    from fastaiagent._internal.errors import ChainCheckpointError
    from fastaiagent.agent.agent import Agent
    from fastaiagent.chain.chain import Chain
    from fastaiagent.chain.interrupt import AlreadyResumed, Resume
    from fastaiagent.llm.stream import Paused, TextDelta, ToolCallEnd

    if not isinstance(target, (Agent, Chain)):
        raise typer.BadParameter(f"Target must be Agent or Chain, got {type(target).__name__}")

    def _response(result: Any) -> RunResponse:
        """``/run`` and ``/resume`` answer the same way, for an AgentResult
        or a Chain result."""
        if isinstance(target, Agent):
            return RunResponse(
                output=result.output,
                latency_ms=result.latency_ms,
                tokens_used=result.tokens_used,
                trace_id=result.trace_id,
                status=result.status,
                execution_id=result.execution_id,
                pending_interrupt=result.pending_interrupt,
            )
        output = result.output
        return RunResponse(
            output=output if isinstance(output, str) else json.dumps(output, default=str),
            status=result.status,
            execution_id=result.execution_id,
            pending_interrupt=result.pending_interrupt,
        )

    async def require_token(
        authorization: str | None = Header(default=None),
    ) -> None:
        """Constant-time bearer check. No-op when ``auth_token`` is unset."""
        if auth_token is None:
            return
        expected = f"Bearer {auth_token}"
        if not hmac.compare_digest(authorization or "", expected):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    from fastaiagent._internal.fastapi_compat import sdk_app_kwargs

    # FastAPI's own request tracing off: an agent run is traced by the SDK; a
    # second, FastAPI-made span per request would land beside it (FastAPI >= 0.143).
    app = FastAPI(title=f"fastaiagent {target.name}", **sdk_app_kwargs())

    class _LimitBody:
        """Refuse an oversized body by ``Content-Length``. A plain ASGI
        middleware, not ``@app.middleware("http")``: that one wraps ``receive``,
        so a route could never see its client hang up (1.93.0)."""

        def __init__(self, inner: Any) -> None:
            self.inner = inner

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                for name, value in scope.get("headers", []):
                    if name == b"content-length" and value.isdigit():
                        if int(value) > max_body_bytes:
                            refusal = JSONResponse(
                                {"detail": f"Request body exceeds {max_body_bytes} bytes."},
                                status_code=413,
                            )
                            await refusal(scope, receive, send)
                            return
            await self.inner(scope, receive, send)

    app.add_middleware(_LimitBody)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run", response_model=RunResponse, dependencies=[Depends(require_token)])
    async def run(req: RunRequest, request: Request) -> RunResponse:
        if not req.input.strip():
            raise HTTPException(400, "input must be non-empty")
        execution_id = req.execution_id or str(uuid.uuid4())
        if isinstance(target, Agent):
            work = asyncio.ensure_future(target.arun(req.input, execution_id=execution_id))
        else:
            work = asyncio.ensure_future(
                target.aexecute({"input": req.input}, execution_id=execution_id)
            )
        # A client that hangs up stops the run (1.93.0): before, an abandoned
        # ``/run`` kept calling the model and tools to the end.
        while not work.done():
            await asyncio.wait({work}, timeout=0.5)
            if not work.done() and await request.is_disconnected():
                target.cancel(execution_id)
                break
        return _response(await work)

    @app.post("/cancel", dependencies=[Depends(require_token)])
    async def cancel(req: CancelRequest) -> dict[str, Any]:
        """Stop a run in progress by its ``execution_id`` (1.93.0). The run
        answers its own request with ``status="cancelled"``."""
        if not target.cancel(req.execution_id):
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"No run {req.execution_id!r} is in progress on this server.",
            )
        return {"execution_id": req.execution_id, "cancelled": True}

    @app.post("/resume", response_model=RunResponse, dependencies=[Depends(require_token)])
    async def resume(req: ResumeRequest) -> RunResponse:
        # The run's own ``aresume`` claims the pause atomically and emits the
        # resolution event, so a second resume of the same run is a 409 here
        # and nothing is reported twice.
        value = Resume(approved=req.approved, metadata=req.metadata)
        try:
            result = await target.aresume(req.execution_id, resume_value=value)
        except AlreadyResumed as err:
            raise HTTPException(status.HTTP_409_CONFLICT, str(err)) from err
        except ChainCheckpointError as err:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(err)) from err
        return _response(result)

    # ``response_model=None`` and no return annotation: under ``from __future__
    # import annotations`` FastAPI could not resolve ``StreamingResponse`` (a
    # local import), and ``GET /openapi.json`` answered 500 (fixed 1.93.0).
    @app.post("/run/stream", response_model=None, dependencies=[Depends(require_token)])
    async def run_stream(req: RunRequest):  # type: ignore[no-untyped-def]
        if not req.input.strip():
            raise HTTPException(400, "input must be non-empty")
        if not isinstance(target, Agent):
            raise HTTPException(400, "Streaming is only supported for Agent targets")

        async def events() -> Any:
            async for event in target.astream(req.input, execution_id=req.execution_id):
                if isinstance(event, TextDelta):
                    yield f"data: {json.dumps({'type': 'delta', 'text': event.text})}\n\n"
                elif isinstance(event, ToolCallEnd):
                    payload = {
                        "type": "tool_call",
                        "name": event.tool_name,
                        "arguments": event.arguments,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                elif isinstance(event, Paused):
                    # The run stopped for a decision; the caller resumes
                    # ``execution_id`` with ``POST /resume``.
                    paused = {
                        "type": "paused",
                        "reason": event.reason,
                        "execution_id": event.execution_id,
                        "context": event.context,
                    }
                    yield f"data: {json.dumps(paused, default=str)}\n\n"
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _warn_on_exposure(host: str, auth_token: str | None) -> None:
    """Warn (once, at startup) when the service is reachable off-box.

    ``serve`` is a deployment primitive — unlike the single-user local UI it
    intentionally keeps the ``0.0.0.0`` default so container port-mapping works.
    The trade-off is that a wildcard bind with no auth exposes agent execution
    to the whole network, so we make that state loud rather than silent
    (security_audit_2 N1). Loopback binds stay quiet.
    """
    if _is_loopback_host(host):
        return
    if auth_token is None:
        typer.secho(
            f"\n⚠️  SECURITY: serving on a non-loopback address ({host!r}) with "
            "NO authentication.\n"
            "   POST /run and /run/stream execute this agent — its tools and LLM "
            "calls — for anyone\n   who can reach the port. Set --auth-token (or "
            "FASTAIAGENT_SERVE_TOKEN) and terminate TLS\n   at a reverse proxy, or "
            "bind --host 127.0.0.1 if remote access isn't needed.\n",
            fg=typer.colors.RED,
            bold=True,
            err=True,
        )
    else:
        typer.secho(
            f"\n⚠️  Serving on {host!r}. Bearer auth is enabled — make sure TLS is "
            "terminated by a reverse\n   proxy so the token isn't sent in plaintext.\n",
            fg=typer.colors.YELLOW,
            err=True,
        )


@agent_app.command("serve")
def serve(
    target: str = typer.Argument(
        ...,
        help="path/to/file.py:my_agent or pkg.module:my_agent (must resolve to Agent or Chain).",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Interface to bind. Defaults to 0.0.0.0 so the service is reachable "
        "from outside a container; pair a non-loopback bind with --auth-token.",
    ),
    port: int = typer.Option(8000, "--port"),
    auth_token: str | None = typer.Option(
        None,
        "--auth-token",
        envvar="FASTAIAGENT_SERVE_TOKEN",
        help="Require 'Authorization: Bearer <token>' on /run, /run/stream and "
        "/resume. Strongly recommended for any non-loopback bind.",
    ),
    max_body_bytes: int = typer.Option(
        _DEFAULT_MAX_BODY_BYTES,
        "--max-body-bytes",
        help="Reject request bodies larger than this many bytes.",
    ),
) -> None:
    """Run an Agent or Chain as a FastAPI service on the uniform contract:

    - ``GET  /health``
    - ``POST /run``         body: {"input": "..."}
    - ``POST /run/stream``  body: {"input": "..."}  returns SSE
    - ``POST /resume``      body: {"execution_id": "...", "approved": true, "metadata": {}}

    The execution routes are unauthenticated unless ``--auth-token`` is set.
    """
    try:
        import uvicorn
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "`fastaiagent agent serve` requires uvicorn. "
            "Install with: pip install 'uvicorn[standard]'"
        ) from err

    resolved = _resolve_target(target)
    app = _build_app(resolved, auth_token=auth_token, max_body_bytes=max_body_bytes)
    _warn_on_exposure(host, auth_token)
    uvicorn.run(app, host=host, port=port)
