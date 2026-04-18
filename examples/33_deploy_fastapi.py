"""Example 33: Deploy an Agent as a FastAPI service.

Minimum working HTTP server that follows the uniform deployment contract
documented in ``docs/deployment/index.md``:

    GET  /health         -> {"status": "ok"}
    POST /run            -> {"output": ..., "latency_ms": ..., "tokens_used": ...}
    POST /run/stream     -> Server-Sent Events token stream

Run it locally::

    pip install 'fastaiagent' fastapi uvicorn
    export OPENAI_API_KEY=sk-...
    python examples/33_deploy_fastapi.py
    # or, equivalently:
    # uvicorn examples.33_deploy_fastapi:app --port 8000

Then from another shell::

    curl -X POST http://localhost:8000/run \\
        -H 'Content-Type: application/json' \\
        -d '{"input": "What is the capital of France?"}'

    # Streaming
    curl -N -X POST http://localhost:8000/run/stream \\
        -H 'Content-Type: application/json' \\
        -d '{"input": "Tell me a short fact about octopuses."}'

The same ``app`` object in this file can be wrapped in a Dockerfile and
shipped to Cloud Run, Fly, Render, Railway, or ECS. See
``docs/deployment/docker.md``.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fastaiagent import Agent, LLMClient
from fastaiagent.llm.stream import TextDelta, ToolCallEnd

# ---------------------------------------------------------------------------
# Build the agent once at startup so the LLMClient is pooled.
# ---------------------------------------------------------------------------


def build_agent() -> Agent:
    return Agent(
        name="deployed-agent",
        system_prompt="You are a helpful assistant. Be concise.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )


@asynccontextmanager
async def lifespan(app_: FastAPI):  # noqa: ARG001 - FastAPI requires this signature
    app.state.agent = build_agent()
    yield


app = FastAPI(title="fastaiagent deployed-agent", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    input: str


class RunResponse(BaseModel):
    output: str
    latency_ms: int
    tokens_used: int
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")
    result = await app.state.agent.arun(req.input)
    return RunResponse(
        output=result.output,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
        trace_id=result.trace_id,
    )


@app.post("/run/stream")
async def run_stream(req: RunRequest) -> StreamingResponse:
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")

    async def event_source():
        async for event in app.state.agent.astream(req.input):
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

    return StreamingResponse(event_source(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
