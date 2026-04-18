# Deploying FastAIAgent

FastAIAgent is a library — `agent.run("...")` is a plain function call. Once you have an agent, you deploy it like any other Python service: wrap it in a web framework, ship it to a platform, and point traffic at it. No special runtime needed.

This section shows four battle-tested paths. Pick the one that matches your infrastructure and budget.

## Decision matrix

| Platform | When to use | Cold start | Pricing shape | Scales to zero | K8s needed |
|---|---|---|---|---|---|
| **[FastAPI + Uvicorn](fastapi.md)** | Baseline. Run anywhere Python runs. | None (always warm) | Whatever VM / container host you use | Depends on host | No |
| **[Docker → Cloud Run / Fly / Render / Railway](docker.md)** | You already ship containers, want HTTPS + autoscaling without ops | ~1–3s | Per-request (Cloud Run), per-second (Fly) | Yes | No |
| **[Modal](modal.md)** | You want serverless Python with zero container work | ~0.5–2s | Per-second compute | Yes | No |
| **[Replicate (Cog)](replicate.md)** | Public-facing inference endpoint; the rest of your stack is on Replicate | ~2–5s | Per-second compute | Yes | No |

If you're just starting out, **begin with FastAPI** locally, then wrap the same app in Docker and deploy to Cloud Run. That's the 95% path.

## The uniform API contract

Every recipe in this section exposes the **same HTTP surface**, so you can switch platforms without rewriting callers:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness probe — returns `{"status": "ok"}` |
| `POST` | `/run` | Synchronous run. Body: `{"input": "..."}`. Response: `{"output": "...", "latency_ms": ..., "tokens_used": ..., "trace_id": ...}` |
| `POST` | `/run/stream` | Server-Sent Events stream of tokens. Body: `{"input": "..."}`. Each event is a JSON object with `{"type": "delta"\|"tool_call"\|"done", ...}` |

Use this contract as a template; extend it freely for your needs.

## What about observability?

Every recipe keeps the existing fastaiagent trace/observability story. Inside your handler, `agent.run(input)` emits OTel spans the same way it does locally. Connect to the [fastaiagent Platform](../platform/index.md) via `fa.connect(...)` at startup and all production traces flow to the same dashboard you use for dev:

```python
import fastaiagent as fa
fa.connect("https://app.fastaiagent.net")
# ...define agent, start server...
```

No runtime changes required.

## What about scale?

All four platforms handle horizontal scale out of the box (Cloud Run / Fly / Modal / Replicate all autoscale on request volume). For very heavy workloads, the patterns that still work:

- **Queue-offload long runs.** Return `202 Accepted` + a `run_id`; a worker picks up the job from Redis / SQS / Pub/Sub and POSTs the result to a webhook. This is exactly how LangGraph Platform works internally — you can implement the same pattern with plain FastAPI + arq/Celery in a weekend.
- **Horizontal workers.** Run N copies of the server container behind a load balancer. Stateless agents parallelize trivially. Agents with memory need either sticky sessions or shared persistent memory (see [ComposableMemory.save/load](../agents/memory.md#persistence)).
- **Long-running interactive sessions.** Use the streaming endpoint + WebSockets, or keep session state in `ComposableMemory` with a `persist_path` per user id.

## What about MCP?

If the clients of your agent are other AI runtimes (Claude Desktop, Cursor, Continue, Zed), you likely don't want HTTP at all — you want **MCP**. See [Expose an Agent as an MCP Server](../tools/mcp-server.md). HTTP deployment and MCP are complementary — an agent can be both.

---

## Recipes

- [FastAPI + Uvicorn](fastapi.md) — the baseline
- [Docker → Cloud Run / Fly / Render / Railway](docker.md) — generic container deploy
- [Modal](modal.md) — serverless Python, no container work
- [Replicate (Cog)](replicate.md) — public inference endpoint
