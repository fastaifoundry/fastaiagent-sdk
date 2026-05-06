# Memory loop

How `fastaiagent.learn` turns past traces into facts that future agents pick up automatically.

## End-to-end flow

```
   Agent runs
        │
        ▼
   spans table in local.db        ← every run is already traced
        │
        ▼
   fastaiagent learn               ← offline CLI, runs on demand
        │ uses an LLM to extract durable facts
        ▼
   learned_memory table            ← schema migration v8
        │
        ▼
   PersistentFactBlock             ← read-only at runtime
        │ injects facts as a SystemMessage
        ▼
   Next agent run                  ← prompt now carries learned context
```

## The pieces

### 1. `MemoryStore` — persistence

[`fastaiagent.learn.MemoryStore`](../api-reference/index.md) is a thin wrapper around the new `learned_memory` table. It exposes:

```python
store.add(Fact(scope="agent", scope_id="my-agent", fact="…"))
store.list_active(scope="agent", scope_id="my-agent")
store.supersede(old_id=12, new_id=34)
```

Inserts are idempotent: re-adding the same `(scope, scope_id, fact, project_id)` returns the existing row id. To replace a fact rather than duplicate it, call `supersede` — the old row is preserved with `superseded_by` pointing at the new row, so the audit chain stays intact.

### 2. `extract_facts_from_trace` — the LLM call

```python
from fastaiagent.learn import extract_facts_from_trace
from fastaiagent.trace.storage import TraceStore

trace = TraceStore().get_trace("…trace_id…")
facts = extract_facts_from_trace(
    trace,
    llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    scope="agent",
    scope_id="my-agent",
)
```

Returns a list of candidate `Fact` objects (not yet stored). The extractor is best-effort — bad LLM output yields `[]` rather than raising. PII guidance is baked into the prompt; the CLI also gates `user` / `project` scopes behind `--allow-personal`.

### 3. `run_extraction` — the windowed batch

For the common "process my last N hours of traces" path:

```python
from fastaiagent.learn import MemoryStore, run_extraction
import fastaiagent as fa

results = run_extraction(
    llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
    store=MemoryStore(),
    scope="agent",
    scope_id="my-agent",
    last_hours=24,
)
```

This is what the `fastaiagent learn` CLI calls under the hood.

### 4. `PersistentFactBlock` — the re-injection

```python
from fastaiagent.agent.memory_blocks import PersistentFactBlock

memory = fa.ComposableMemory(
    primary=fa.AgentMemory(),
    blocks=[
        PersistentFactBlock(scope="agent", scope_id="my-agent", max_facts=30),
    ],
)
agent = fa.Agent(name="my-agent", system_prompt="…", llm=llm, memory=memory)
```

Every `agent.arun(...)` now sees a `Learned facts (agent:my-agent):` block prepended to the system prompt with the active facts (newest first, capped at `max_facts`). The block is read-only at runtime — facts only update via `fastaiagent learn`.

## Conflict resolution

Same fact text → idempotent insert (the UNIQUE constraint deduplicates).

Semantically conflicting facts (e.g. "prefers terse" vs "prefers verbose") are not auto-detected by v1. Two paths:

1. **Manual via CLI.** Inspect `fastaiagent learn list`, then `fastaiagent learn supersede <old_id> <new_id>`.
2. **Automatic via re-extraction.** Re-running the loop produces newer rows; downstream consumers can prefer recency by ordering on `created_at DESC` (which `list_active` does).

A fully automatic semantic-dedup pass (LLM judge per scope_id batch) is tracked in the future-work backlog.

## Privacy

`fastaiagent learn` extracts only `agent`-scoped facts by default. Both `--scope user` and `--scope project` require `--allow-personal` to be set explicitly:

```sh
# Allowed:
fastaiagent learn --scope agent --scope-id deep-research

# Refused without opt-in:
fastaiagent learn --scope user --scope-id user-42
# → "Refusing to extract user/project-scoped facts without --allow-personal."
```

The extraction prompt also instructs the model to skip names, emails, phone numbers, and addresses. This is best-effort, not a guarantee — review extracted facts before re-injecting them in production.

## Inspecting the table

Via the local UI:

```
GET /api/learned_memory                        # active facts
GET /api/learned_memory?include_superseded=true
GET /api/learned_memory?scope=agent&scope_id=my-agent
GET /api/learned_memory/scopes                 # distinct (scope, scope_id) pairs
```

Or directly:

```sh
sqlite3 .fastaiagent/local.db \
  "SELECT id, scope, scope_id, fact, superseded_by FROM learned_memory ORDER BY created_at DESC LIMIT 20;"
```

## What's not in v1

- **Skill extraction** — reusable mini-procedures. Needs a different storage model + replay-eval to verify a skill before re-injecting.
- **Prompt/harness mutation** — Harrison's Meta-Harness pattern. Requires an automated quality gate before a coding agent's prompt rewrites can ship.
- **Online learning** — agents updating their own context mid-run. Out of scope; the SDK's UI is refresh-based by design.
- **Cron triggers** — `fastaiagent learn` runs on demand. Schedule it however you want.
