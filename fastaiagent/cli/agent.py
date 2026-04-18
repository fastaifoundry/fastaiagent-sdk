"""``fastaiagent agent serve`` — run an Agent or Chain as a FastAPI service
following the uniform deployment contract documented at
``docs/deployment/fastapi.md``.

This saves you from copy-pasting the 80-line starter server into every
project. It's a thin wrapper: if you need custom routes, auth middleware,
or anything else the uniform contract doesn't cover, eject by copying
[examples/33_deploy_fastapi.py] and extending it.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import typer

try:
    from pydantic import BaseModel

    class RunRequest(BaseModel):
        input: str

    class RunResponse(BaseModel):
        output: str
        latency_ms: int = 0
        tokens_used: int = 0
        trace_id: str | None = None
except ImportError:  # pragma: no cover - pydantic is a required fastaiagent dep
    RunRequest = None  # type: ignore[assignment, misc]
    RunResponse = None  # type: ignore[assignment, misc]

agent_app = typer.Typer(
    name="agent",
    help="Run an Agent or Chain as a service.",
    no_args_is_help=True,
)


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


def _build_app(target: Any) -> Any:
    """Build a FastAPI app that exposes ``target`` over the uniform contract."""
    try:
        import json

        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "`fastaiagent agent serve` requires fastapi and uvicorn. "
            "Install with: pip install fastapi 'uvicorn[standard]'"
        ) from err

    from fastaiagent.agent.agent import Agent
    from fastaiagent.chain.chain import Chain
    from fastaiagent.llm.stream import TextDelta, ToolCallEnd

    if not isinstance(target, (Agent, Chain)):
        raise typer.BadParameter(
            f"Target must be Agent or Chain, got {type(target).__name__}"
        )

    app = FastAPI(title=f"fastaiagent {target.name}")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/run", response_model=RunResponse)
    async def run(req: RunRequest) -> RunResponse:
        if not req.input.strip():
            raise HTTPException(400, "input must be non-empty")
        if isinstance(target, Agent):
            agent_result = await target.arun(req.input)
            return RunResponse(
                output=agent_result.output,
                latency_ms=agent_result.latency_ms,
                tokens_used=agent_result.tokens_used,
                trace_id=agent_result.trace_id,
            )
        # Chain
        chain_result = await target.aexecute({"input": req.input})
        chain_output = (
            chain_result.output
            if isinstance(chain_result.output, str)
            else json.dumps(chain_result.output, default=str)
        )
        return RunResponse(output=chain_output)

    @app.post("/run/stream")
    async def run_stream(req: RunRequest) -> StreamingResponse:
        if not req.input.strip():
            raise HTTPException(400, "input must be non-empty")
        if not isinstance(target, Agent):
            raise HTTPException(400, "Streaming is only supported for Agent targets")

        async def events() -> Any:
            async for event in target.astream(req.input):
                if isinstance(event, TextDelta):
                    yield f"data: {json.dumps({'type': 'delta', 'text': event.text})}\n\n"
                elif isinstance(event, ToolCallEnd):
                    payload = {
                        "type": "tool_call",
                        "name": event.tool_name,
                        "arguments": event.arguments,
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


@agent_app.command("serve")
def serve(
    target: str = typer.Argument(
        ...,
        help="path/to/file.py:my_agent or pkg.module:my_agent (must resolve to Agent or Chain).",
    ),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
) -> None:
    """Run an Agent or Chain as a FastAPI service on the uniform contract:

    - ``GET  /health``
    - ``POST /run``         body: {"input": "..."}
    - ``POST /run/stream``  body: {"input": "..."}  returns SSE
    """
    try:
        import uvicorn
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "`fastaiagent agent serve` requires uvicorn. "
            "Install with: pip install 'uvicorn[standard]'"
        ) from err

    resolved = _resolve_target(target)
    app = _build_app(resolved)
    uvicorn.run(app, host=host, port=port, reload=reload)
