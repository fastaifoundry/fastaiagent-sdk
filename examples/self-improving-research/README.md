# Self-improving research agent — closed loop

Wraps the [`deep-research-agent`](../deep-research-agent) template with the trace learning loop introduced in PR B. End-to-end:

```
deep_research × N  ──→  traces in local.db
                              │
                              ▼
                   fastaiagent learn (offline)
                              │
                              ▼
                       learned_memory rows
                              │
                              ▼
      optimize (Phase 2.5) — tune prompt + which facts to inject
                              │
                              ▼
              PersistentFactBlock (in memory_setup.py)
                              │
                              ▼
                deep_research again — facts now flow
```

## Run

```sh
pip install -r requirements.txt
export TAVILY_API_KEY=...      # optional — falls back to mock corpus
python agent.py --topic "How does Self-RAG differ from vanilla RAG?"
```

This walks all four phases (seed → learn → optimize → replay) in one shot:

```sh
python agent.py --skip-optimize --topic "..."   # seed → learn → replay (cheaper, no optimize)
python agent.py --skip-seed --topic "..."        # reuse seed traces already in local.db
```

## What you'll see

- Phase 1: 3 seed `deep_research` runs.
- Phase 2: a printed list of facts extracted by the learn loop.
- Phase 2.5: an optimization report (`baseline → steps → holdout-guarded winner`) for a fact-bearing research agent — the prompt + memory levers tuned against a small eval set.
- Phase 3: a single follow-up run. Inspect the trace in `fastaiagent ui` — the scope and writer system prompts now carry a `Learned facts (agent:deep-research):` block injected by `PersistentFactBlock`.

## Phase 2.5 — optimize, and its scope limitation

[`fastaiagent.optimize`](../../docs/evaluation/optimization.md) closes the loop on the cold-eval path: it proposes prompt rewrites and learned-fact subsets, scores each on a held-out split, and keeps the best. **It targets a single `Agent`.** The deep-research flagship is a *multi-agent pipeline* (scope + writer), so Phase 2.5 optimizes a **single fact-bearing research agent** sharing the pipeline's memory scope (`agent`/`deep-research`) — it demonstrates the closed loop but does **not** optimize the whole pipeline. Threading a per-sub-agent result back into the full pipeline (or joint/Replay-grounded optimization) is a later-phase extension. The optimizer is **read-only on the fact store**: it selects *which* learned facts to inject, never creating, editing, or deleting them.

## How it works

The `deep-research-agent` template's [`memory_setup.py`](../deep-research-agent/memory_setup.py) returns a `ComposableMemory` wrapping a `PersistentFactBlock` with `scope="agent", scope_id="deep-research"`. PR A shipped this as a placeholder; PR B activates it.

To disable for an A/B comparison:

```sh
DEEP_RESEARCH_DISABLE_LEARNED_MEMORY=1 python agent.py
```
