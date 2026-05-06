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

This walks all three phases in one shot. To re-run only the replay step (e.g. iterating on the follow-up topic):

```sh
python agent.py --skip-seed --topic "..."
```

## What you'll see

- Phase 1: 3 seed `deep_research` runs.
- Phase 2: a printed list of facts extracted by the learn loop.
- Phase 3: a single follow-up run. Inspect the trace in `fastaiagent ui` — the scope and writer system prompts now carry a `Learned facts (agent:deep-research):` block injected by `PersistentFactBlock`.

## How it works

The `deep-research-agent` template's [`memory_setup.py`](../deep-research-agent/memory_setup.py) returns a `ComposableMemory` wrapping a `PersistentFactBlock` with `scope="agent", scope_id="deep-research"`. PR A shipped this as a placeholder; PR B activates it.

To disable for an A/B comparison:

```sh
DEEP_RESEARCH_DISABLE_LEARNED_MEMORY=1 python agent.py
```
