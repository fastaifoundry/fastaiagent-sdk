# Concepts & Mental Model

This page is the mental model for Agent Replay — *why* it exists, *how* it
reconstructs a past run from its trace, *what* it holds fixed versus re-runs,
and where its boundaries are. Read it first, then use the
[Replay reference](index.md) and [Fidelity Guarantees](guarantees.md) for depth.

## Why replay exists

When an agent misbehaves in production — hallucinates, calls the wrong tool,
gives a bad answer — you need to understand *why* and test a fix *without
burning new production traffic and without hoping you can reproduce it.* The
run already happened; you have its trace. Replay turns that trace back into a
runnable agent.

The loop it enables:

1. **Load** the exact trace of the failure.
2. **Step through** it to find the step that went wrong.
3. **Fork** at that point.
4. **Change** the prompt, input, config, or tools.
5. **Rerun** the agent end-to-end with your change applied.
6. **Compare** original vs. fixed — and save the pair as a regression test.

## How it reconstructs a run

This is the core idea, and it's what makes replay more than a recording: an
agent trace is not just a log, it's a **blueprint**. When an agent runs, its
root `agent.<name>` span captures everything needed to rebuild it — the system
prompt, the LLM provider/model/config, the tool definitions, the guardrails,
the config — and each `llm.*` span captures the model's response
(`gen_ai.response.*`).

So `Replay.load(trace_id)` reads that blueprint and can literally call
`Agent.from_dict(...)` to reconstruct the same agent, then run it again. You
didn't have to save the agent object or keep any session alive — the trace is
enough.

```
trace  ──▶  root agent.<name> span  ──▶  Agent.from_dict(...)  ──▶  re-run
             (prompt, llm cfg,
              tools, guardrails)
```

## What's held fixed vs. re-run

A replay is a *real execution*, not a playback. By default (`"live"` mode) it
reconstructs the agent from the trace and runs it again — re-issuing LLM calls
and re-executing tools. What you override with `modify_*` / `with_tools`
changes; everything else is taken from the trace.

The **determinism mode** controls how faithful the reproduction is:

| Mode | LLM calls | Result | Use it for |
|------|-----------|--------|------------|
| `"live"` *(default)* | Re-issued with captured settings | Realistic, may differ run-to-run | Verifying a fix works under real conditions |
| `"recorded"` | **Skipped** — captured responses replayed in order | **Byte-identical** to the original | Regression tests, exact reproduction |
| `"deterministic"` | Re-issued at `temp=0`, `seed=42` | Semantically stable | Reducing noise without skipping the call |

!!! info "Verified against a live run"
    Running `examples/04_agent_replay.py`: `Replay.load(trace_id)` rebuilt a
    4-span trace (agent → llm → tool → llm), `fork_at(2).modify_prompt(...).rerun()`
    produced a changed answer, `compare()` computed `diverged_at`, and
    `fork_at(0).with_determinism("recorded").rerun()` returned the original
    output **byte-identically with no HTTP call to the provider** — the response
    came from the trace's `gen_ai.response.content`.

!!! warning "Tools re-execute; the LLM is what gets recorded"
    `"recorded"` skips the *LLM* call, not tool calls — a tool the recorded
    response invokes still runs live during a rerun. For tools with side
    effects, override them for the rerun (`with_tool_override("charge_card", stub)`)
    or mark them with a `replay_class` that the central Replay engine honors.
    See [Fidelity Guarantees](guarantees.md).

## Our approach, in context

Most tools in this space offer one of two things. Observability platforms let
you re-run a **single LLM call** in a playground — edit one prompt, see one new
completion; that's prompt iteration, not agent reproduction. Others offer
**visual session playback** — re-watching a recorded timeline, with no
re-execution at all. The one framework with state rewind requires you to author
your workflow in its graph/state abstraction first, and still re-fires calls
non-deterministically.

Replay's approach is different in three ways, and each is deliberate:

- **Whole-agent, from a plain trace** — it reconstructs and re-runs the entire
  agent (the full tool-calling loop), not one call — and it needs nothing but
  the recorded trace. You don't adopt a special graph or keep a session alive.
- **Faithful reproduction is a first-class mode** — `"recorded"` gives
  byte-identical output by substituting the captured model responses, so a past
  run becomes a deterministic regression test.
- **Honest boundaries** — where reproduction can't be guaranteed (side-effecting
  tools, provider nondeterminism in `"live"` mode), the docs say so rather than
  implying magic.

## Boundaries — and the checkpoint cousins

Replay is **trace-based** and works on a **standalone agent** run: it finds the
root `agent.<name>` span and rebuilds from it. Two things it deliberately does
*not* do, with the right tool for each:

- **Chains don't replay.** A chain trace's root is `chain.<name>` and its span
  attributes describe *what happened*, not the graph you defined — the structure
  isn't recoverable from the trace. To re-run a chain from a saved point, use
  the **checkpoint** primitives `Chain.aresume(...)` / `Chain.afork(..., modified_state=...)`.
- **Mid-run state counterfactuals.** `ForkedReplay.modify_state()` intentionally
  raises `NotImplementedError` — trace/replay is a read-only inspect-and-rerun
  surface. To fork from a saved step with changed state, use `Agent.afork(...)`
  / `Chain.afork(...)`.

The distinction in one line: **Replay reconstructs from a trace (no state store
needed); `afork`/`aresume` continue from a checkpoint (a saved state snapshot).**
Use Replay to reproduce and debug a past agent run; use the checkpoint
primitives to branch or resume a durable run — including chains.

## A guided path

1. [`examples/04_agent_replay.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/04_agent_replay.py) — the full loop: run → load → step → fork → modify → rerun → compare → `"recorded"`.
2. [`examples/62_replay_to_regression.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/62_replay_to_regression.py) — turn a production failure into a saved regression test.
3. [`examples/70_tool_replay_class.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/70_tool_replay_class.py) — mark tools `read_only` / `idempotent` / `side_effecting` for safe reruns.

## Next steps

- [Agent Replay reference](index.md) — the full API: `load`, `fork_at`, `modify_*`, `with_determinism`, `with_tool_override`, `compare`
- [Fidelity Guarantees](guarantees.md) — what's captured, what reproduces exactly, and the known gaps
- [Agents — the run loop](../agents/concepts.md#the-run-loop) — what a replay reconstructs and re-executes
- [Chains — durability](../chains/concepts.md) — `aresume` / `afork` for chain runs
