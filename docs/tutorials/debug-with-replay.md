# Debug a Production Failure with Agent Replay

Your agent failed in production. Here's how to find and fix the bug in
60 seconds with fork-and-rerun debugging.

This tutorial uses the local SQLite trace store. The same flow works
with traces pulled from the FastAIAgent Platform — see
[docs/platform/](../platform/index.md).

## Prereqs

```bash
pip install fastaiagent
export OPENAI_API_KEY=sk-...
```

A runnable end-to-end version of this tutorial lives at
[examples/04_agent_replay.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/04_agent_replay.py).

## 1. Run an agent so we have a trace to debug

```python
from fastaiagent import Agent, FunctionTool, LLMClient

def lookup_order(order_id: str) -> str:
    orders = {"ORD-001": "MacBook Pro, delivered 2026-04-03"}
    return orders.get(order_id, f"Order {order_id} not found.")

agent = Agent(
    name="support-bot",
    system_prompt="You are a support agent. Use lookup_order to check status.",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
)

result = agent.run("What's the status of order ORD-001?")
print(result.trace_id)  # the handle for everything below
```

Every agent run is traced. ``result.trace_id`` is the only thing you
need to keep.

## 2. Load the trace from local storage

```python
from fastaiagent.trace import Replay

replay = Replay.load(result.trace_id)
print(replay.summary())
```

In production, you'd load the trace by the ID surfaced from your alert
or error log, e.g. ``Replay.load("trace_abc123")``.

## 3. Step through to find the failing span

```python
for step in replay.step_through():
    print(f"[{step.step}] {step.span_name}")
```

Each ``ReplayStep`` carries the span name, input, output, and
attributes — enough to spot which step misbehaved.

## 4. Fork at the failing step and modify the prompt

```python
forked = replay.fork_at(step=2)
forked.modify_prompt(
    "You are a support agent. Use lookup_order. "
    "Reply in exactly one sentence. Never use bullet points."
)
```

``fork_at`` returns a ``ForkedReplay`` you can chain modifications on:
``modify_prompt``, ``modify_input``, ``modify_config``, ``modify_state``.

## 5. Rerun and compare

```python
rerun = forked.rerun()
print("Original:", rerun.original_output)
print("New:     ", rerun.new_output)

diff = forked.compare(rerun)
print("Diverged at step:", diff.diverged_at)
```

The rerun uses the modified prompt; ``compare`` shows where the original
and rerun diverged.

## 6. Multimodal forks

When the original input was multimodal, ``modify_input`` accepts the
same shapes ``Agent.run`` does — strings, ``Image``, ``PDF``, or a list:

```python
from fastaiagent import Image

forked.modify_input([
    "Try with a clearer image",
    Image.from_file("clearer_receipt.jpg"),
])
result = forked.rerun()
```

See [docs/multimodal/](../multimodal/index.md) for more on multimodal
inputs.

## CLI shortcuts

```bash
# List recent traces
fastaiagent traces list

# Pull a specific trace as JSON
fastaiagent traces show <trace-id>
```

For interactive replay, use the [Local UI](../ui/index.md) — it ships
inside the wheel and gives you a fork dialog, span inspector, and
side-by-side comparison view.

**That's fork-and-rerun debugging. No other SDK has this.**

## Next steps

- [Agent Replay reference](../replay/index.md) for the complete API
- [Tracing guide](../tracing/index.md) for setting up tracing
- [Evaluation guide](../evaluation/index.md) to prevent regressions with eval datasets
- Runnable end-to-end script:
  [examples/04_agent_replay.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/04_agent_replay.py)
