# Memory, end to end: a developer's guide

This guide takes you from "my agent forgets everything" to a **multi-user,
personalized, observable** memory setup — using one object, `Memory`.

By the end you'll know how to: keep conversation context, learn and recall
durable facts per user, run the full fact lifecycle (create / read / update /
forget), scale to an external backend, retrieve by meaning, and *see* it all in
the trace + Memory UI.

> `Memory` is the recommended front door. It's built on composable blocks
> (`ComposableMemory` + `StaticBlock`/`VectorBlock`/…), which remain available
> for custom behaviours — see [Reference: Memory](../agents/memory.md#advanced-composable-blocks).

## The mental model: three tiers

| Tier | True for | Lifetime |
|---|---|---|
| **global** | everyone using the agent (shared truth) | durable |
| **user** | one user (personalization) | durable, cross-session |
| **session** | one conversation (the working window) | ephemeral |

`project_id` is an orthogonal tenant partition applied across tiers.

## Step 1 — Remember the conversation

```python
from fastaiagent import Agent, LLMClient, Memory

llm = LLMClient(provider="openai", model="gpt-4.1")
agent = Agent(name="assistant", llm=llm, memory=Memory())

agent.run("My name is Alice.")
agent.run("What's my name?")          # → "Alice" (the session window remembers)
```

`Memory()` with nothing else is just a sliding window (`window=20` by default).

## Step 2 — Personalize per user (multi-user safe)

Pass a `user_id` **resolver** — one agent definition serves every user, resolved
per run from `RunContext`. Add `learn=llm` to extract and persist durable facts.

```python
from dataclasses import dataclass
from fastaiagent import RunContext

@dataclass
class Session:
    user_id: str

agent = Agent(name="support", llm=llm, memory=Memory(
    user_id=lambda ctx: ctx.state.user_id,
    learn=llm,
))

agent.run("I have a dog named Rex.", context=RunContext(state=Session("alice")))
agent.run("I have a cat named Mia.", context=RunContext(state=Session("bob")))

# Each user is fully isolated — durable facts AND the live window:
agent.run("What's my pet?", context=RunContext(state=Session("alice")))  # → "Rex"
```

Under the hood `Memory` routes to a **per-user working memory**, so Bob's turn
never leaks into Alice's context. A missing/unresolved id yields **no** personal
facts (safe-by-default).

## Step 3 — The fact lifecycle (CRUD)

Use `Memory` as a direct store too:

```python
mem = Memory(location="sqlite")

# CREATE
fid = mem.persist("Alice prefers email", tier="user", id="alice")

# READ
mem.retrieve(tier="user", id="alice")            # → [Fact("Alice prefers email", ...)]

# UPDATE (supersede — keeps history, never overwrites)
mem.update("Alice prefers Slack", old="Alice prefers email", tier="user", id="alice")

# DELETE
mem.forget(tier="user", id="alice")              # removes the subject's facts
```

`update` marks the old fact superseded (kept in the audit history) and activates
the new one. `forget` hard-deletes, including superseded history for that subject.

## Step 4 — Global vs user facts

```python
mem.persist("Support replies within 24 hours.", tier="global")   # everyone sees it
mem.persist("Alice is on the Pro plan.", tier="user", id="alice")# only Alice
```

Attach a `Memory(agent_id="support", user_id=..., learn=llm)` and both tiers are
injected each turn — global always, user only for the resolved user.

## Step 5 — Retrieve by meaning (semantic)

```python
mem = Memory(location="sqlite", semantic="auto")   # in-process FAISS + default embedder
mem.persist("The user is allergic to peanuts", tier="user", id="alice")
mem.retrieve("what foods should we avoid?", tier="user", id="alice")   # → the peanut fact
```

`semantic="auto"` builds a vector index sized to the embedder; pass a
`VectorStore` for a shared/production index. Facts written by `learn=` are
indexed automatically. Results stay scope-isolated and skip superseded facts.

## Step 6 — Scale to an external backend

Same API, different `location` — no agent-code change:

```python
Memory(location="postgres://user:pw@host:5432/db")   # fastaiagent[postgres]
Memory(location="redis://host:6379/0")               # fastaiagent[redis]
Memory(location=my_store)                            # any FactStore
```

All backends share one conformance-tested contract (idempotent add, safe
scoping, supersede, delete). Runnable demo: `examples/memory_backends/`.

## Step 7 — See it: traces + the Memory UI

Every read/write and direct op is a trace span. Run `fastaiagent ui`:

- **Trace detail** — `memory.read` / `memory.write` with a child span per block
  (rendered counts, `VectorBlock` similarity scores, snippets), plus
  `memory.persist` / `memory.retrieve` / `memory.update` for direct ops.
- **Memory page** (Knowledge → Memory) — browse durable facts by tier, with a
  **Source** column (`trace` link back to the run that learned it, or `manual`),
  a **scope filter**, and a **Show superseded** toggle for the audit history.

Content in spans (snippets, queries) is payload-gated (`FASTAIAGENT_TRACE_PAYLOADS=0`)
and honors the "Mask secrets" redaction path.

> The Memory page browses the local SQLite store; with an external backend,
> memory operations are observed via the trace spans above.

## Safe-by-default scoping

At `user`/`project` scope, an **empty id returns nothing** — one user's facts
can never leak into another's context. Use `scope_id="*"` (low-level store) to
deliberately read across all subjects. The `agent`/global tier stays permissive.

## When to drop to blocks

Reach for `ComposableMemory` + blocks only for custom behaviours — writing your
own `MemoryBlock`, precise block ordering, or upstream dedupe. 95% of apps never
need to. See [Reference: composable blocks](../agents/memory.md#advanced-composable-blocks).

## Full example

`examples/memory_simple/` runs one agent for two users end to end (learn +
recall + isolation) and captures the trace + Memory page;
`examples/memory_backends/` runs the full lifecycle across SQLite / Postgres /
Redis.
