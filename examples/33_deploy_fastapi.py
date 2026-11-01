"""Example 33: Deploy an Agent as a FastAPI service.

Minimum working HTTP server that follows the uniform deployment contract
documented in ``docs/deployment/index.md``:

    GET  /health         -> {"status": "ok"}
    POST /run            -> {"output": ..., "latency_ms": ..., "tokens_used": ...,
                             "messages": [...]}
    POST /run/stream     -> Server-Sent Events token stream

A request continues a conversation by sending back the previous response's
``messages``, and names its user in an ``X-User-Id`` header (a stand-in for
your auth). See "Conversations and users" in ``docs/deployment/fastapi.md``.

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
from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from fastaiagent import Agent, LLMClient, RunContext
from fastaiagent.llm.message import Message
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
    # The conversation so far: the ``messages`` the previous response returned.
    messages: list[Message] = Field(default_factory=list)


class RunResponse(BaseModel):
    output: str
    latency_ms: int
    tokens_used: int
    trace_id: str | None = None
    # Send this back as ``messages`` on the next request to continue.
    messages: list[Message] = Field(default_factory=list)


@dataclass
class Caller:
    user_id: str | None


def caller(user_id: str | None) -> RunContext[Caller]:
    # Stand-in for your auth: take the user from a verified token in
    # production, never from the request body.
    return RunContext(state=Caller(user_id=user_id))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest, x_user_id: str | None = Header(default=None)) -> RunResponse:
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")
    result = await app.state.agent.arun(req.input, context=caller(x_user_id), messages=req.messages)
    return RunResponse(
        output=result.output,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
        trace_id=result.trace_id,
        messages=result.messages,
    )


@app.post("/run/stream")
async def run_stream(
    req: RunRequest, x_user_id: str | None = Header(default=None)
) -> StreamingResponse:
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")

    async def event_source():
        stream = app.state.agent.astream(
            req.input, context=caller(x_user_id), messages=req.messages
        )
        async for event in stream:
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
