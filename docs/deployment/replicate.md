# Deploy on Replicate (Cog)

[Replicate](https://replicate.com) is a public inference platform. You package a model (or an agent) with [Cog](https://cog.run), push it, and get a public HTTPS API. Good fit when:

- You want a public URL your users can call directly.
- The rest of your stack is already on Replicate.
- You're OK with slightly higher cold-start latency (~2–5s) in exchange for zero ops.

## The three files

Cog needs a `cog.yaml` (environment), a `predict.py` (the handler), and optionally your own Python modules.

### `cog.yaml`

```yaml
build:
  gpu: false
  python_version: "3.12"
  python_packages:
    - "fastaiagent[kb]>=0.6.0"
  system_packages:
    - libstdc++6   # faiss

predict: "predict.py:Predictor"
```

### `predict.py`

```python
from cog import BasePredictor, Input

from fastaiagent import Agent, LLMClient


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Runs once when the container starts. Build the agent here so
        it's reused across every predict() call."""
        import os
        # Replicate injects secrets as env vars via the web UI.
        assert os.environ.get("OPENAI_API_KEY"), "set OPENAI_API_KEY in the Replicate UI"

        self.agent = Agent(
            name="replicate-agent",
            system_prompt="You are a helpful assistant. Be concise.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        )

    def predict(
        self,
        input: str = Input(description="User's question or request."),
    ) -> dict:
        """Runs on every request. Each call is its own fresh agent run."""
        result = self.agent.run(input)
        return {
            "output": result.output,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "trace_id": result.trace_id,
        }
```

## Build and run locally

```bash
# Install Cog
brew install cog

# Test locally
cog run python -c "from predict import Predictor; p = Predictor(); p.setup(); print(p.predict(input='hi'))"

# Start the dev HTTP server
cog run -p 5000 python -m cog.server.http
curl -X POST http://localhost:5000/predictions -H 'Content-Type: application/json' \
  -d '{"input": {"input": "What is 2+2?"}}'
```

## Push to Replicate

```bash
# One-time: create a model under your Replicate account at replicate.com/create
# then push:
cog login
cog push r8.im/<username>/my-agent
```

In the Replicate UI, set `OPENAI_API_KEY` (and optionally `FASTAIAGENT_PLATFORM_URL` / `FASTAIAGENT_API_KEY`) as secrets on the model's settings page.

## Call the public endpoint

Replicate gives every model a public REST + Python + JS API. The Python client:

```python
import replicate

output = replicate.run(
    "<username>/my-agent:<version-hash>",
    input={"input": "What is the capital of France?"},
)
print(output)
```

Or raw HTTP:

```bash
curl -X POST https://api.replicate.com/v1/predictions \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "version": "<version-hash>",
    "input": {"input": "What is the capital of France?"}
  }'
```

## Streaming

Cog supports streaming via an iterator return type. Swap `predict` for:

```python
from typing import Iterator
from fastaiagent.llm.stream import TextDelta

def predict(self, input: str) -> Iterator[str]:
    import asyncio

    async def _run():
        async for event in self.agent.astream(input):
            if isinstance(event, TextDelta):
                yield event.text

    # Cog wraps the iterator in an SSE stream automatically.
    loop = asyncio.new_event_loop()
    try:
        gen = _run()
        while True:
            try:
                yield loop.run_until_complete(anext(gen))
            except StopAsyncIteration:
                break
    finally:
        loop.close()
```

Replicate surfaces this as `stream=True` on the client:

```python
for token in replicate.run("<username>/my-agent:<hash>", input={"input": "..."}, stream=True):
    print(token, end="", flush=True)
```

## Cost shape

Replicate bills per second of container time, similar to Modal. A CPU-only agent (what most fastaiagent deployments need) runs at the lowest tier (~$0.00006/second). A typical text-only agent costs fractions of a cent per request.

First request after a long idle period has a **2–5s cold start** while Replicate pulls your image. Paying for Replicate's "always-warm" instances removes it.

## Observability

Same pattern — wire up the fastaiagent Platform in `setup()`:

```python
def setup(self) -> None:
    import os
    import fastaiagent as fa
    if os.environ.get("FASTAIAGENT_PLATFORM_URL"):
        fa.connect(
            os.environ["FASTAIAGENT_PLATFORM_URL"],
            api_key=os.environ["FASTAIAGENT_API_KEY"],
        )
    # ... build agent ...
```

## When not to use Replicate

- You want a **private** endpoint. Replicate is public by default; private models require a higher tier.
- You need **very low-latency** responses. Cog's request/response cycle adds a bit of overhead vs plain FastAPI.
- You want **custom routes / middleware / multiple endpoints** on the same deployment. Cog's one-function-per-model model is stricter than FastAPI.

---

## Next

- [FastAPI](fastapi.md) — the lower-level alternative
- [Docker → Cloud Run](docker.md) — container-based path with private deployment
- [Modal](modal.md) — serverless Python without Cog's constraints
