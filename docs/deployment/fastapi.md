# Deploy with FastAPI + Uvicorn

The baseline. Works on a laptop, a VM, a container, a Kubernetes pod, Render, Railway, Fly, Cloud Run — anywhere Python runs.

## Minimum viable server

A complete production-shaped server in ~80 lines:

```python
# server.py
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fastaiagent import Agent, LLMClient
from fastaiagent.llm.stream import TextDelta, ToolCallEnd


# ---- Build the agent once at startup (keeps LLM client pooled) -----------


def build_agent() -> Agent:
    return Agent(
        name="deployed-agent",
        system_prompt="You are a helpful assistant. Be concise.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    yield  # shutdown hooks go below if you need them


app = FastAPI(lifespan=lifespan)


# ---- Contract ------------------------------------------------------------


class RunRequest(BaseModel):
    input: str


class RunResponse(BaseModel):
    output: str
    latency_ms: int
    tokens_used: int
    trace_id: str | None = None


@app.get("/health")
async def health() -> dict:
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
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "tool_call",
                            "name": event.tool_name,
                            "arguments": event.arguments,
                        }
                    )
                    + "\n\n"
                )
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
```

## Run it

```bash
pip install fastaiagent fastapi uvicorn
export OPENAI_API_KEY=sk-...
uvicorn server:app --host 0.0.0.0 --port 8000
```

Hit it:

```bash
curl -X POST http://localhost:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"input": "What is the capital of France?"}'
```

Streaming (SSE):

```bash
curl -N -X POST http://localhost:8000/run/stream \
  -H 'Content-Type: application/json' \
  -d '{"input": "Tell me a short fact about octopuses."}'
```

## Production notes

- **Concurrency.** Uvicorn defaults to a single worker. For real traffic, use `uvicorn server:app --workers 4` or put it behind gunicorn: `gunicorn server:app -k uvicorn.workers.UvicornWorker -w 4`. Each worker holds its own agent instance — that's fine because LLMClient is pooled per worker.
- **Memory across requests.** The agent above is stateless per request. If you want conversation memory scoped per user, attach a `ComposableMemory` keyed by a user id in the request body — don't share one `AgentMemory` across users (threading nightmare).
- **Graceful shutdown.** Uvicorn handles `SIGTERM` cleanly out of the box; Kubernetes will wait for in-flight requests to finish.
- **Timeouts.** An agent that loops on tool calls can run for minutes. Set a per-route timeout on your ingress (nginx/ALB/Cloud Run) to something generous (`300s`) and handle the truncation server-side via a [middleware-enforced `ToolBudget`](../agents/middleware.md#toolbudget).
- **Observability.** Wire `fa.connect("https://app.fastaiagent.net")` inside `build_agent()` (or at module import time) and production traces land in the same dashboard you use for dev.

## Deploying the container

Same server, four platforms:

| Platform | Command | Notes |
|---|---|---|
| **Render** | Push to GitHub, pick "Web Service", set `uvicorn server:app --host 0.0.0.0 --port $PORT` as the start command | Easy; ~$7/mo Hobby tier |
| **Railway** | `railway up` | Same experience as Render |
| **Fly.io** | `fly launch` → `fly deploy` | Metered per-second; good for bursty traffic |
| **Cloud Run** | See [Docker recipe](docker.md) | Scales to zero |

## When not to use plain FastAPI

- You want **zero cold starts** under bursty load — FastAPI behind a VM is great; FastAPI on Cloud Run has cold starts unless you set `--min-instances=1` (which costs money).
- You want **no Docker**. Skip to [Modal](modal.md) for Python-first serverless with no container work.

---

## Next

- [Docker + Cloud Run / Fly / Render / Railway](docker.md) — wrap this same server in a container
- [Streaming](../streaming/index.md) — more on the underlying stream events
- [Agent Memory](../agents/memory.md) — per-user session memory patterns
