# Memory observability

See *what your agent remembered* — in the trace and in the Local UI.

Until now, agent memory was a black box at runtime: reads and writes emitted no
spans, so you couldn't tell which block recalled what, or why. This example
shows the new memory observability: `memory.read` / `memory.write` spans with a
**child span per block**, including VectorBlock similarity **scores** and the
bounded **snippets** it recalled — plus a Memory page that browses the durable
`learned_memory` facts `PersistentFactBlock` reads back across runs.

## Run

```sh
# 1. Seed a trace whose spans show what the agent remembered (needs OPENAI_API_KEY)
zsh -lc 'python companion.py'

# 2. Capture the Local UI screenshots (needs Playwright)
pip install playwright && python -m playwright install chromium
zsh -lc 'python snapshot.py'
```

`companion.py` runs a real agent with a `ComposableMemory`
(`StaticBlock` + `VectorBlock` + `SummaryBlock` + `FactExtractionBlock`) over a
few turns. It demonstrates all the memory-observability features:

- **`FactExtractionBlock(persist=True)`** — extracted facts are written to the
  durable `learned_memory` table *during the run*, each stamped with the run's
  trace id (so the Memory page shows a clickable `trace` source).
- **`VectorBlock(dedupe_against_upstream=True)`** — recall skips anything the
  `StaticBlock` already pinned (see `deduped_count` on the `memory.read.vector`
  span).
- A **manual** `MemoryStore.add` fact (source `manual`) plus a **supersede**
  chain, so the Memory page's "Show superseded" history toggle has something to
  show.

`snapshot.py` boots `fastaiagent ui` and saves PNGs to `screenshots/`.

> **Where the Memory page rows come from:** most are **auto-persisted during the
> run** by `FactExtractionBlock(persist=True)` (source = `trace`, confidence
> 0.6); one is added manually via `MemoryStore.add` (source = `manual`,
> confidence 1.0). In production you can also populate the table offline with
> `fastaiagent learn`. Rows are never auto-detected without one of these
> producers.

## What to look for

- **`memory.read`** (one per turn) → child spans per block. Open the
  `memory.read.vector` child: `memory.scores` is the per-item similarity in rank
  order, `memory.snippets` is what was injected. This is the difference between
  "memory recalled something" and "memory recalled the *wrong* thing, score 0.71".
- **`memory.write`** → child spans with an `action` (`embedded`, `summarized`,
  `extracted_facts`, `stored`, `noop`) and a `detail` count.
- The **Memory** page (sidebar → Knowledge → Memory) lists learned facts with a
  scope filter and a "Mask secrets" toggle.

Snippets and fact text honor `FASTAIAGENT_TRACE_PAYLOADS=0` and any installed
`RedactionPolicy`, exactly like the rest of the trace UI.
