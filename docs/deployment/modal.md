# Deploy on Modal

[Modal](https://modal.com) is serverless Python. You write a Python file, decorate functions, run `modal deploy`, and get an HTTPS endpoint. No Dockerfile, no YAML, no Kubernetes. Scales to zero and per-second-billed.

A fastaiagent agent maps one-to-one onto a Modal web endpoint.

## Minimum viable deployment

```python
# modal_agent.py
import modal

# 1. Define the image Modal will build. Point at your extras.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libstdc++6")  # faiss
    .pip_install("fastaiagent[kb]>=0.6.0", "fastapi>=0.115")
)

# 2. Declare the app and the secrets it needs.
app = modal.App(
    name="fastaiagent-demo",
    image=image,
    secrets=[modal.Secret.from_name("openai-secret")],  # holds OPENAI_API_KEY
)


# 3. A class with @enter=startup, @method=handler. Modal keeps one warm
#    container per replica, so the agent is built once.
@app.cls(min_containers=0, scaledown_window=300)
class AgentService:
    @modal.enter()
    def startup(self):
        from fastaiagent import Agent, LLMClient
        self.agent = Agent(
            name="modal-agent",
            system_prompt="You are a helpful assistant. Be concise.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        )

    @modal.fastapi_endpoint(method="POST")
    def run(self, body: dict) -> dict:
        user_input = (body.get("input") or "").strip()
        if not user_input:
            return {"error": "input must be non-empty"}
        result = self.agent.run(user_input)
        return {
            "output": result.output,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "trace_id": result.trace_id,
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        return {"status": "ok"}
```

## Set the secret

Store your OpenAI key as a Modal secret one time:

```bash
modal secret create openai-secret OPENAI_API_KEY=sk-...
```

## Deploy

```bash
pip install modal
modal token new             # first time only
modal deploy modal_agent.py
```

Modal prints the public URLs:

```
✓ Created web endpoint for AgentService.run  => https://<workspace>--fastaiagent-demo-agentservice-run.modal.run
✓ Created web endpoint for AgentService.health => https://<workspace>--fastaiagent-demo-agentservice-health.modal.run
```

Hit it:

```bash
curl -X POST https://<workspace>--fastaiagent-demo-agentservice-run.modal.run \
  -H 'Content-Type: application/json' \
  -d '{"input": "What is the capital of France?"}'
```

## Streaming

Modal's `fastapi_endpoint` returns any FastAPI-compatible response, so SSE works the same way it does on plain Uvicorn:

```python
from fastapi.responses import StreamingResponse
import json
from fastaiagent.llm.stream import TextDelta

@modal.fastapi_endpoint(method="POST")
def run_stream(self, body: dict):
    async def events():
        async for event in self.agent.astream(body["input"]):
            if isinstance(event, TextDelta):
                yield f"data: {json.dumps({'type': 'delta', 'text': event.text})}\n\n"
        yield "data: " + json.dumps({"type": "done"}) + "\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
```

## Tuning

| Knob | Default | When to change |
|---|---|---|
| `min_containers=0` | Scale to zero | Set to `1` to eliminate cold starts (~$0.20–0.50/hour per warm container) |
| `scaledown_window=300` | Keep warm 5min after last request | Shorten for lower cost on bursty workloads; lengthen for smoother latency |
| `max_containers` | 100 | Cap cost explosion if a caller loops |
| `timeout` | 600s | Match your expected worst-case agent run |
| `cpu` / `memory` | 1 vCPU / 128 MiB | Raise if agents load a large KB in memory |

## Cost shape

Modal charges per second of container wall-time (plus egress). A mostly-LLM-bound agent costs ~$0.000033/s = ~$0.12/hour of actual agent runtime. Idle time (between requests while a container is still warm) is billed at the same rate unless you've scaled to zero.

For a low-volume agent (~100 requests/day averaging 3s each), expect $1–5/month.

## Observability

Wire up the fastaiagent Platform inside `@modal.enter()`:

```python
@modal.enter()
def startup(self):
    import os
    import fastaiagent as fa
    if os.environ.get("FASTAIAGENT_PLATFORM_URL"):
        fa.connect(
            os.environ["FASTAIAGENT_PLATFORM_URL"],
            api_key=os.environ["FASTAIAGENT_API_KEY"],
        )
    # build agent...
```

Add those vars to a second Modal secret and attach it to the app.

## When not to use Modal

- You already have a container platform. Modal is simplest when you don't.
- Your agent requires a persistent volume shared across replicas (Modal has `Volume`s but they add complexity; Postgres-backed memory is cleaner).
- You need to run in a specific VPC / region Modal doesn't support.

---

## Next

- [FastAPI](fastapi.md) — what the underlying server looks like as plain FastAPI
- [Docker → Cloud Run](docker.md) — the container-based alternative
- [Replicate (Cog)](replicate.md) — if Replicate fits the rest of your stack
