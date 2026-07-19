# Concepts & Mental Model

This page explains **why** testing an agent is different from testing normal
code, **what** the SDK gives you to make it tractable, and **the concept of how**
those pieces fit into one layered strategy. For the API see the
[Testing reference](index.md).

## Why agent testing is hard

Every testing tool ever built rests on one assumption: `f(x)` returns the same
`y` every time, so you can assert equality. An agent breaks that assumption at
the root — its `f` contains a *sampled* language model. The same input yields
different wording, different tool choices, sometimes a different number of
loops. Assert on equality and your suite fails randomly; assert on nothing and
you have no suite.

So the real question is never "how do I test an agent?" It's **"what do I hold
fixed so that something becomes assertable?"** Every tool below is a different
answer to that question.

## The layers, ordered by how much nondeterminism you remove

| Layer | What you hold fixed | What it actually tests |
|-------|---------------------|------------------------|
| **Deterministic unit tests** (`TestModel` / `FunctionModel`) | *All* of it — the model is replaced | Your harness: does the right tool get called with the right args, does middleware fire in order, does the guardrail trip, does the loop terminate |
| **Evals** (`@case`, `pytest_dataset`, scorers) | Nothing — you measure instead of compare | Model behavior in aggregate, as a score over a dataset |
| **Simulation** (`Scenario`, `SimulatedUser`) | The counterparty | Behavior across a multi-turn conversation |
| **Replay regressions** | The past — a real recorded run | That a specific bug, once fixed, stays fixed |

The progression is the mental model: **remove all nondeterminism to test your
code; keep it and score it to test the model; record it to test history.**

## The concept of how

### The fakes are real clients, not mocks

`TestModel` and `FunctionModel` are **subclasses of `LLMClient`**, not
`MagicMock`s. They implement the same surface (`complete`, `acomplete`,
`stream`, `astream`) and simply never make an HTTP call.

That single decision is what makes everything else compose. Because they *are*
clients, swapping one in exercises the entire real code path — the agent loop,
middleware, guardrails, structured output, the tool-calling protocol. And
because they emit **real OTel spans** (tagged `gen_ai.system="test"`), a fake
run is indistinguishable to tracing, the Local UI, evals, and Replay. A mock
would have forced every test to know the client's internals and would have
produced no trace at all.

### Scripting the two fakes

- **`TestModel`** takes canned responses and plays them as a **round-robin tape
  with a sticky last turn** — past the end of the script it repeats the final
  turn forever. That's deliberate: an agent loop terminates instead of hanging.
  The trade-off is that an over-running agent won't fail loudly. It also records
  every call in `.calls`, so you can assert on *what prompt your agent actually
  sent* — often more valuable than asserting on the reply.
- **`FunctionModel`** takes a responder function that receives the **full
  conversation** and computes the next response, so it can branch on history.
  This is how you script a state machine.

!!! info "A real limitation worth knowing"
    `TestModel` attaches tool calls only to a single string response — you
    **cannot** script "call a tool on turn 1, then answer on turn 2" with
    `TestModel` alone. Use `FunctionModel` with a closure over turn state for
    multi-turn tool flows.

### How a test becomes an eval run

The pytest plugin is **auto-registered** via an entry point — installing the SDK
activates it, no `conftest.py` wiring — and it's opt-in: tests that don't import
its helpers are unaffected.

- **`@case(input=…, expected=…)`** stamps the case onto the test function.
- **`@pytest_dataset("cases.jsonl")`** loads the dataset and desugars into
  ordinary `pytest.mark.parametrize`, so **one dataset row becomes one test**
  with its own ID in the normal pytest report. (The file is read at import time,
  not at test time.)
- **`evaluate_one`** is a fixture that returns a *callable*, so your test body
  keeps control of timing and error handling. It resolves input/expected with
  precedence **explicit args → `@case` → dataset row**, runs the agent, unwraps
  `.output` and `.trace_id` (so the eval case links back to its trace), scores,
  and asserts.

Results persist to `eval_runs` / `eval_cases` in `local.db` under a run name of
`pytest::<nodeid>`, so CI runs show up in the Local UI alongside everything else.

### Closing the loop from production

The layers connect end to end:

```
production failure ─▶ trace ─▶ Replay.fork_at + fix ─▶ save_as_test
                                                            │
        CI ◀─ @pytest_dataset ◀─ JSONL regression dataset ◀──┘
```

`Replay.save_as_test` writes the rerun using the same field names `evaluate()`
reads (`input`, `expected_output`), plus provenance (`source_trace_id`,
`fork_step`, the modifications applied) — so a fixed production failure becomes
a permanent regression row without any format conversion.

## Honest boundaries

- **A deterministic test validates your plumbing, not your prompt.** `TestModel`
  proves the tool was called correctly; it says nothing about whether a real
  model *would* have called it. That's the handoff to
  [evaluation](../evaluation/concepts.md) — don't mistake a green unit suite for
  quality.
- **The public testing surface is small**: two fake clients, the `@case` /
  `pytest_dataset` decorators, and the `evaluate_one` fixture. Things you'll find
  by grepping the repo — `MockLLMClient`, `isolated_local_db`, `CaptureServer` —
  are the SDK's *own* test scaffolding, live in `tests/`, and are not importable
  from the installed package. Don't build on them.
- **Two real gaps**: there's no public span-capture assertion helper (assert via
  the trace store instead), and no public DB-isolation fixture — so
  `evaluate_one`'s auto-persist writes to your real `./.fastaiagent/local.db`
  unless you pass `persist=False` or point `FASTAIAGENT_LOCAL_DB` somewhere
  temporary.

## Next steps

- [Testing reference](index.md) — `TestModel` / `FunctionModel` API and the pytest plugin
- [Evaluation](../evaluation/concepts.md) — scoring when you can't assert equality
- [Simulation](../simulation/concepts.md) — multi-turn behavior
- [Replay](../replay/concepts.md) — turning a production failure into a test
- Examples: `examples/60_test_model.py`, `examples/61_eval_pytest.py`, `examples/62_replay_to_regression.py`
