"""
Production-shape HITL deployment — FastAPI wrapping the support agent.

The CLI in ``agent.py`` resumes via Python's ``input()`` — fine for a demo,
useless for a web app. This server demonstrates the same `interrupt()` /
`aresume()` flow over HTTP, which is what you'd actually deploy.

Endpoints:

  POST /chat
    Body: {"query": "...", "user_email": "alice@acme.com"}
    Response (completed):
      {"status": "completed", "output": "...", "execution_id": "...", "trace_id": "..."}
    Response (paused for approval):
      {"status": "paused", "execution_id": "...", "pending": {reason, context, ...}}

  POST /resume/{execution_id}
    Body: {"approved": true, "approver": "alice@acme.com", "user_email": "alice@acme.com"}
    Response:
      Same shape as /chat — either "completed" with output, or "paused" again
      if the resumed run hits another interrupt().

  GET /health
    {"status": "ok", "agent": "customer-support"}

Run:
    pip install -r requirements.txt fastapi uvicorn
    python server.py                        # serves http://127.0.0.1:8080
    # or:
    uvicorn server:app --host 0.0.0.0 --port 8080

Try it:
    curl -X POST http://127.0.0.1:8080/chat \\
         -H "Content-Type: application/json" \\
         -d '{"query": "Please file an URGENT billing dispute ticket"}'
    # returns status=paused with execution_id

    curl -X POST http://127.0.0.1:8080/resume/<execution_id> \\
         -H "Content-Type: application/json" \\
         -d '{"approved": true, "approver": "alice@acme.com"}'
    # returns status=completed with the agent's response

The Agent's ``SQLiteCheckpointer`` makes this durable: the resume can come
from a different process, hours later — the suspended state lives in
``.fastaiagent/local.db`` and ``aresume()`` claims it atomically.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import agent  # the same Agent instance the CLI uses
from context import create_deps


class ChatRequest(BaseModel):
    query: str
    user_email: str = "anonymous@example.com"


class ResumeRequest(BaseModel):
    approved: bool
    approver: str = "anonymous"
    user_email: str = "anonymous@example.com"


def _serialize_result(result: fa.AgentResult) -> dict[str, Any]:
    """Shape the AgentResult for JSON. Always returns the execution_id so
    the client can call /resume on the next leg if status==paused."""
    base = {
        "status": result.status,
        "execution_id": result.execution_id,
        "trace_id": result.trace_id,
        "tokens_used": result.tokens_used,
        "latency_ms": result.latency_ms,
    }
    if result.status == "paused":
        base["pending"] = result.pending_interrupt or {}
        base["output"] = ""
    else:
        base["output"] = result.output
    return base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Hook for opening real DB pools / verifying upstream dependencies.
    # Today the agent's SQLiteCheckpointer.setup() is invoked lazily.
    yield


app = FastAPI(
    title="Customer Support Agent — HITL HTTP shell",
    description=__doc__,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": agent.name}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict[str, Any]:
    """Single-turn entry point. May return status=paused if a tool called
    interrupt(); the client then POSTs to /resume/<execution_id> with the
    approval decision."""
    deps = await create_deps(user_email=req.user_email)
    ctx = fa.RunContext(state=deps)
    try:
        result = await agent.arun(req.query, context=ctx)
    except Exception as e:
        # Surface tool / LLM errors as 500s rather than letting FastAPI's
        # default handler emit a stack trace.
        raise HTTPException(status_code=500, detail=str(e))
    return _serialize_result(result)


@app.post("/resume/{execution_id}")
async def resume(execution_id: str, req: ResumeRequest) -> dict[str, Any]:
    """Resume a paused agent run with the human's decision."""
    deps = await create_deps(user_email=req.user_email)
    ctx = fa.RunContext(state=deps)
    try:
        result = await agent.aresume(
            execution_id,
            resume_value=fa.Resume(
                approved=req.approved,
                metadata={"approver": req.approver},
            ),
            context=ctx,
        )
    except fa.AlreadyResumed:
        # Atomic-claim contract: a duplicate resume (e.g. two clicks of
        # "Approve" from the UI) is a 409 conflict, not a 500.
        raise HTTPException(
            status_code=409,
            detail=f"Execution {execution_id!r} was already resumed.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _serialize_result(result)


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
