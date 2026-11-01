# Deploy with FastAPI + Uvicorn

The baseline. Works on a laptop, a VM, a container, a Kubernetes pod, Render, Railway, Fly, Cloud Run — anywhere Python runs.

## Minimum viable server

A complete production-shaped server in ~115 lines:

```python
# server.py
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
    # The conversation so far: the `messages` the previous response returned.
    messages: list[Message] = Field(default_factory=list)


class RunResponse(BaseModel):
    output: str
    latency_ms: int
    tokens_used: int
    trace_id: str | None = None
    # Send this back as `messages` on the next request to continue.
    messages: list[Message] = Field(default_factory=list)


@dataclass
class Caller:
    user_id: str | None


def caller(user_id: str | None) -> RunContext[Caller]:
    # Stand-in for your auth: take the user from a verified token in
    # production, never from the request body.
    return RunContext(state=Caller(user_id=user_id))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest, x_user_id: str | None = Header(default=None)) -> RunResponse:
    if not req.input.strip():
        raise HTTPException(400, "input must be non-empty")
    result = await app.state.agent.arun(
        req.input, context=caller(x_user_id), messages=req.messages
    )
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

A second turn sends the first response's `messages` back:

```bash
first=$(curl -s -X POST http://localhost:8000/run -H 'Content-Type: application/json' \
  -d '{"input": "My order number is 4417."}')
jq -n --argjson h "$(echo "$first" | jq .messages)" \
  '{input: "What was my order number?", messages: $h}' |
  curl -s -X POST http://localhost:8000/run -H 'Content-Type: application/json' -d @- |
  jq -r .output
```

Streaming (SSE):

```bash
curl -N -X POST http://localhost:8000/run/stream \
  -H 'Content-Type: application/json' \
  -d '{"input": "Tell me a short fact about octopuses."}'
```

## Conversations and users

The server keeps no state between requests, so any number of workers can answer
any request. Two things make a request part of a conversation:

- **`messages`**: the conversation so far. `/run` returns it, and the client sends
  it back with the next input. It holds the user's messages, the tool turns and the
  answers, never the system prompt. The stream has no final history, so a client of
  `/run/stream` appends its own message and the streamed answer.
- **The user**: whoever the run is for. It reaches the agent as `context=`, where
  tools and [`Memory`](../agents/memory.md) read it. The example takes it from an
  `X-User-Id` header as a stand-in for your auth. In production, read it from a
  verified token: a client that can name any user can read that user's memory.

History from the client is the client's to edit: it can add an answer the agent never
gave, or a tool result that never ran. When clients are not trusted, keep
`result.messages` in your own store keyed by a conversation id, and let the client
send only that id.

Long conversations grow every request. To keep each model call bounded, trim them
with [`TrimLongMessages`](../agents/middleware.md).

## Let the server remember users (`Memory`)

To have the agent remember people across conversations, attach a
[`Memory`](../agents/memory.md) that reads the user from the run's context:

```python
from fastaiagent import Memory


def build_agent() -> Agent:
    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    return Agent(
        name="deployed-agent",
        system_prompt="You are a helpful assistant. Be concise.",
        llm=llm,
        memory=Memory(
            location="postgres://user:pw@host:5432/db",  # facts every worker shares
            user_id=lambda ctx: ctx.state.user_id,
            learn=llm,  # keep durable facts from what each user says
        ),
    )
```

One `Memory` serves every user, each in their own compartment (see
[Memory boundaries](../agents/memory-boundaries.md)). A request without a user gets
no personal memory and writes none.

With `Memory` attached, the server keeps the conversation, so:

- **Send `messages` or keep a window, not both.** `Memory` adds the recent turns to
  the prompt, so a client that also sends them gets every turn twice. A client that
  keeps the conversation itself pairs with `Memory(window=0, …)`: no window, still
  the user's facts. With `window=0` the server keeps no conversation state, so the
  next point does not apply.
- **The recent turns live in the worker's process.** Facts are in `location` and
  every worker reads them, but a user's next request on another worker starts a
  fresh conversation. Run one worker, route each user to the same worker (sticky
  sessions), or [save and load each user's window](../agents/memory.md#multi-user-safety-important).
- **Use Postgres or Redis for `location`** when there is more than one worker or
  machine. The default, `"sqlite"`, is a local file.

## Production notes

- **Concurrency.** Uvicorn defaults to a single worker. For real traffic, use `uvicorn server:app --workers 4` or put it behind gunicorn: `gunicorn server:app -k uvicorn.workers.UvicornWorker -w 4`. Each worker holds its own agent instance — that's fine because LLMClient is pooled per worker.
- **Workers and memory.** The server above keeps no state, so `--workers` is safe. With `Memory` attached, each worker holds its own recent turns; see [Let the server remember users](#let-the-server-remember-users-memory).
- **Graceful shutdown.** Uvicorn handles `SIGTERM` cleanly out of the box; Kubernetes will wait for in-flight requests to finish.
- **Timeouts.** An agent that loops on tool calls can run for minutes. Set a per-route timeout on your ingress (nginx/ALB/Cloud Run) to something generous (`300s`) and handle the truncation server-side via a [middleware-enforced `ToolBudget`](../agents/middleware.md#toolbudgetmax_calls10-message).
- **Observability.** Call `fa.connect(api_key=os.environ["FASTAIAGENT_API_KEY"])` inside `build_agent()` (or at module import time) and production traces land in the same dashboard you use for dev. Pass `target=` only for a plane other than `https://app.fastaiagent.net`.

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
