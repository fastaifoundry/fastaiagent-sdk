# Replay Fidelity Guarantees

This page documents what Agent Replay captures, what it can faithfully
reproduce, and where the known fidelity gaps are. Each guarantee is
backed by a test in
[`tests/test_replay_determinism.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/tests/test_replay_determinism.py).

## Determinism modes

`ForkedReplay.with_determinism(mode)` controls how the LLM is invoked
during a rerun:

| Mode | LLM HTTP call | Output |
|---|---|---|
| `"live"` *(default)* | Yes — uses the captured temperature/seed. | Nondeterministic for non-zero temperature. Matches the original behavior. |
| `"recorded"` | **No** — replays the captured response. | Byte-identical to the original. Use for regression tests. |
| `"deterministic"` | Yes — but with `temperature=0` and `seed=42` forced. | Semantically stable across runs where the provider supports `seed`. |

```python
from fastaiagent.trace.replay import Replay

replay = Replay.load(trace_id)
forked = (
    replay
    .fork_at(step=0)
    .modify_prompt("Be more concise.")
    .with_determinism("recorded")  # ← skips the LLM call entirely
)
result = forked.rerun()
```

## What `"recorded"` mode captures

The recorded `LLMResponse` is reconstructed from these GenAI
semantic-convention attributes on the captured `llm.*` span:

* `gen_ai.response.content` — main text output
* `gen_ai.response.finish_reason` — `stop`, `length`, etc.
* `gen_ai.response.tool_calls` — tool invocations the original LLM
  requested (JSON-decoded into `ToolCall` instances)

If `FASTAIAGENT_TRACE_PAYLOADS=0` was set on the original run, none
of these attributes are present and `recorded` mode raises
`ReplayError`. The fix is to enable payloads on the runs you intend
to replay later.

## Known fidelity gaps

These are documented limitations to set expectations. Each is tracked
in the top-3-recommendation as a deferred follow-up.

### Streaming chunks aren't recorded granularly

Streaming responses (`agent.astream(...)`) yield individual
`StreamEvent` chunks, but only the **final accumulated content**
lands in `gen_ai.response.content`. `recorded` mode reconstructs the
LLMResponse from that final text — streaming cadence is lost.

For most regression-test scenarios this is fine: the final answer is
what you're testing. If you need per-chunk fidelity, file an issue —
granular streaming capture is on the deferred-follow-up list.

### Tool execution is rerun, not recorded

`recorded` mode skips the LLM call but **does not** skip tool calls.
If the LLM's recorded response includes a tool invocation, that tool
will execute live during rerun. Two ways to handle this:

* **`with_tool_override(name, stub)`** — replace a single tool with
  a deterministic stand-in. Other tools keep their reconstructed
  implementations. Multiple calls compose.
* **`with_tools([...])`** — full replacement. All tools come from
  the list you pass; nothing is reconstructed from the trace.

```python
forked.with_tool_override("search_kb", deterministic_stub)
forked.with_tool_override("create_ticket", another_stub)
```

### Provider-specific determinism support

| Provider | `temperature=0` | Seed support |
|---|---|---|
| OpenAI / Azure OpenAI | ✅ | ✅ via `seed=` |
| Anthropic | ✅ | ✅ via `seed=` |
| Gemini (native wire) | ✅ | ✅ via `seed=` |
| Bedrock | ✅ | ❌ — Bedrock has no `seed` field |
| Ollama / vLLM / local | ✅ | ✅ via `seed=` |
| Provider presets (groq, openrouter, mistral, …) | ✅ | Varies — most pass through `seed=` |

`deterministic` mode degrades gracefully on Bedrock — it forces
`temperature=0` and skips the seed. Output is still much more
reproducible than `live` but not byte-identical across runs.

## Divergence detection

`forked.compare(result)` now computes the first divergence by walking
the original and rerun step lists in parallel. Span names and span
output attributes are compared step-for-step; the first mismatch wins.

This replaces the v1.13 behavior of hardcoding `diverged_at` to the
fork point — which was misleading because the actual divergence often
happens later than where the user asked to fork.

```python
comp = forked.compare(rerun_result)
assert comp.compare_status == "ok"
if comp.diverged_at is None:
    print("rerun matched the original step-for-step")
else:
    print(f"diverged at step {comp.diverged_at}: "
          f"{comp.original_steps[comp.diverged_at].span_name}")
```

When the rerun trace can't be loaded (e.g., `Agent.arun` failed),
`compare_status` is set to `"rerun_failed"` and `diverged_at` stays
`None` — so UIs can distinguish "matched everything" from "couldn't
tell".

## Saving a rerun as a regression test

`rerun.save_as_test(...)` appends a JSONL line in the format
`evaluate()` reads natively:

```python
rerun.save_as_test(
    "regression_tests.jsonl",
    input="What is our refund policy?",
    expected_output=str(rerun.new_output),
    source_trace_id=original_trace_id,
)
```

For the full failure → fix → regression-test loop, see
[`examples/62_replay_to_regression.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/62_replay_to_regression.py).
