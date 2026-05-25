# Regression from Trace

The canonical pattern for turning a production failure into a passing
regression test. Pairs with the [Agent Replay](../replay/index.md)
API and the new v1.14 fidelity affordances
([guarantees](../replay/guarantees.md)).

The full source lives at
[`examples/regression-from-trace/`](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/examples/regression-from-trace).

## Before / after — same trace shape, fixed output

The buggy `lookup_order` silently returns ORD-001's record for any
unknown ID (with the requested ID stamped on), so the agent confidently
ships wrong details. After `fix.py` swaps in the fixed tool and reruns
live, the same prompt and LLM produce the correct "not found" reply.

| Failing trace (buggy tool) | Fixed trace (after `with_tool_override`) |
|---|---|
| ![Agent reports a delivery for ORD-999 even though it doesn't exist](../ui/screenshots/0_3-failing-trace.png) | ![Agent correctly reports "The order ORD-999 was not found."](../ui/screenshots/0_3-fixed-trace.png) |

## The loop

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  capture.py  │ ── │  analyze.py  │ ── │   fix.py     │
│  buggy run   │    │  inspect     │    │  fork + fix  │
└──────────────┘    └──────────────┘    └──────────────┘
                                                │
                                                ▼
                          ┌──────────────┐    ┌──────────────┐
                          │  verify.py   │ ── │ save_test.py │
                          │  evaluate()  │    │ append JSONL │
                          └──────────────┘    └──────────────┘
```

Five small scripts, one shared trace ID stashed in
`.fastaiagent-demo/regression-from-trace/last_trace_id.txt`. Run them
in order or jump in at any step — each can be invoked standalone
once its input file exists.

## The deliberate bug

The template ships with a broken `lookup_order` tool. It silently
falls back to ORD-001's record when asked about an unknown order ID,
**stamping the requested ID onto the fallback data** so the LLM has
nothing to cross-check:

```python
def _lookup_order_buggy(order_id: str) -> dict[str, str]:
    found = KNOWN_ORDERS.get(order_id)
    if found is not None:
        return found
    # Silent fallback — overwrite the id so the response looks coherent.
    fallback = dict(KNOWN_ORDERS["ORD-001"])
    fallback["id"] = order_id
    return fallback
```

A customer asks "What's the status of order ORD-999?" The agent
confidently replies "Your order ORD-999 for the MacBook Pro 16-inch
has been delivered on 2026-04-03." Nothing crashed, no test would
catch it — until a customer complaint surfaces the failure.

This is the silent-failure class. Fail-loud bugs (exceptions,
structured errors) get caught by CI; silent ones need the trace →
replay loop to find and fix.

## The five steps

### 1. `capture.py` — reproduce in a trace

```python
agent = build_buggy_agent()
result = agent.run("What's the status of order ORD-999?")
TRACE_ID_FILE.write_text(result.trace_id)
```

Stashes the trace ID for the rest of the loop to consume.

### 2. `analyze.py` — find the smoking gun

```python
replay = Replay.load(trace_id)
for step in replay.step_through():
    print(f"[{step.step}] {step.span_name}")
```

Walks every span. The `tool.lookup_order` span shows the fallback
record reaching the LLM.

### 3. `fix.py` — fork, override, rerun live

```python
forked = (
    Replay.load(trace_id)
    .fork_at(step=0)
    .with_tool_override("lookup_order", fixed_lookup_order_tool())
)
rerun = forked.rerun()    # live mode — LLM re-ingests corrected tool output
```

`with_tool_override` is new in v1.14. It substitutes a single tool by
name while keeping every other tool, prompt, and LLM config from the
original capture. Live rerun mode means the LLM sees the new tool
output and re-generates its reply.

### 4. `save_test.py` — append to the regression dataset

```python
result.save_as_test(
    "regression_dataset.jsonl",
    input="What's the status of order ORD-999?",
    expected_output=str(rerun.new_output),
    source_trace_id=original_trace_id,
)
```

JSONL fields match what `fastaiagent.eval.evaluate(...)` reads
natively.

### 5. `verify.py` — `evaluate()` against the fixed agent

```python
results = evaluate(
    agent_fn=lambda text: build_fixed_agent().run(text).output,
    dataset="regression_dataset.jsonl",
    scorers=[LLMJudge(criteria="correctness")],
)
```

`LLMJudge` (not `exact_match`) because LLM outputs are paraphrase-stable
but not byte-stable. Every captured failure should pass forever once
the fix is in place.

## Why a live rerun, not `determinism="recorded"`

`with_determinism("recorded")` skips the LLM HTTP call and replays
the captured response — useful when **the prompt** is what you're
fixing. For a **tool** fix, the LLM has to re-ingest the new tool
output and re-generate, so `fix.py` uses the default `"live"` mode.
See [Fidelity Guarantees](../replay/guarantees.md) for the per-mode
matrix.

## What gets caught forever

Every row in `regression_dataset.jsonl` is one production-failure
class the agent must keep handling correctly. The dataset grows
append-only — when a new customer complaint surfaces, run
`capture.py` with the new input, `fix.py` with the new fix,
`save_test.py` to commit the case. Future `verify.py` runs catch any
regression of the same failure mode, automatically.
