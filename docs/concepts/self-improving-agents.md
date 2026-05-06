# Self-improving agents

Agents that get better at their job over time without retraining.

## The framing

Harrison Chase's "continual learning" framing names three layers an agent can improve along:

| Layer | What changes | Hard part |
|---|---|---|
| **Model** | Weights — SFT, GRPO, RLAIF | Catastrophic forgetting |
| **Harness** | Code, prompts, tools — Meta-Harness rewrites | Replay-eval to prevent drift |
| **Context** | Facts, skills, memory, scoped instructions | Scoping + privacy |

`fastaiagent.learn` ships **the context layer, scoped to memory**. That's a deliberate choice — it's the smallest piece with real leverage that doesn't require a replay-eval system as a prerequisite.

## What we ship in v1

The two pieces that compose into a self-improving loop:

1. **A flagship template** — [Deep Research Agent](../flagships/deep-research-agent.md). A long-horizon agent worth improving.
2. **A trace learning loop** — [`fastaiagent.learn`](../learning/index.md). Reads traces, extracts durable facts, re-injects via `PersistentFactBlock`.

Together: a Deep Research Agent that gets sharper at recurring topics on its own. All offline, all local.

## What "self-improving" means here (and what it doesn't)

It **does** mean:

- Facts learned in one run carry into the next.
- Scoping (`agent` / `project` / `user`) keeps signal isolated.
- The audit chain (`source_trace_id`, `superseded_by`) is queryable.
- A/B comparison is one env var (`DEEP_RESEARCH_DISABLE_LEARNED_MEMORY=1`).

It **does not** mean:

- The agent rewrites its own prompts. (Meta-Harness — future work.)
- The model fine-tunes itself. (Out of scope for the SDK.)
- The agent extracts skills it can call back as tools. (Future work — needs replay-eval first.)
- The improvement is automatic in real-time. The loop is **batch + offline** by design — you run `fastaiagent learn` when you want it.

## Try it

```sh
cd examples/self-improving-research
pip install -r requirements.txt
python agent.py --topic "How does Self-RAG differ from vanilla RAG?"
```

The script walks all three phases: seed runs → extract → replay. Inspect the trace from the replay phase in `fastaiagent ui` — you'll see a `Learned facts (agent:deep-research):` block prepended to the scope and writer system messages. That's the loop closing.

## Where this sits in the broader stack

```
                      Model improvement (out of scope)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Harness improvement       — future                          │
│    Meta-Harness loop       (needs replay-eval first)         │
│                                                              │
│  Context improvement       — v1 ships this                   │
│    PersistentFactBlock     (read-only at runtime)            │
│    learned_memory table    (schema v8)                       │
│    fastaiagent learn       (offline CLI)                     │
│                                                              │
│  Substrate                 — already shipped                 │
│    Traces in local.db      (every run, every harness)        │
│    Universal harness       (LangGraph / CrewAI / PydanticAI) │
└─────────────────────────────────────────────────────────────┘
```

The substrate was the prerequisite — without rich, queryable traces, none of this would work. The learning loop is the simplest thing that can show value on top.

## Future work

Tracked in the planning file:

- **Skills** — extract reusable mini-procedures from successful traces; expose as callable tools.
- **Meta-Harness** — coding agent rewrites template prompts based on trace failure patterns.
- **Replay-eval** — A/B compare runs with and without a candidate change. Prerequisite for both above.
- **Online learning** — agents that update their own context mid-run.
