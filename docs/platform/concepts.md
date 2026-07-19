# Concepts & Mental Model

This page explains **what** connecting to the platform does, **why** the SDK is
local-first, and **the concept of how** data actually moves — the outbox
pattern, the push/pull boundary, and the guarantees that keep the agent hot path
safe. For setup see the [Platform Connection reference](index.md).

## What it is

The SDK works fully standalone. `fa.connect(api_key=..., project=...)` is an
optional upgrade that lights up observability, governance, and shared resources:

```python
import fastaiagent as fa
fa.connect(api_key="fa-...", project="my-project")
```

Your agent code doesn't change. What changes is where telemetry *also* goes, and
which platform-owned resources become readable.

## Why local-first

Because an agent shouldn't stop working when a network does. The design rule is
that every feature has a complete local implementation, and connecting **adds a
second sink** rather than swapping the first one out. Local SQLite stays the
source of truth; the platform is a replica fed from it.

!!! info "A nuance worth correcting"
    It's tempting to read "connect replaces local backends with platform
    services." That's not what the code does for telemetry — traces,
    checkpoints, and HITL events are written locally *and* drained upward. The
    one place connecting genuinely substitutes a backend is prompt resolution
    (`PromptRegistry.get(source="auto")` prefers the platform, silently falling
    back to local).

## The concept of how: the outbox pattern

This is the spine of the whole integration, and it's the same shape for traces,
checkpoints, and HITL events:

```
write locally (synced=0)  ──▶  background drain  ──▶  POST  ──▶  mark synced=1
        │                                                  (only on confirmed 2xx)
   durable source of truth
```

Three consequences fall out of it:

1. **SQLite is the queue; the batch processor is just a doorbell.** The trace
   exporter *ignores* the batch of spans handed to it and instead drains
   unsynced rows from the store. A platform outage can't lose data — rows simply
   stay unsynced until the next drain.
2. **Delivery is at-least-once, and that's safe.** Rows are marked synced only
   after a 2xx, so a crash mid-flight means a re-send. The server is idempotent
   by `span_id`, so a re-sent span is counted once.
3. **Retry lives in the durable buffer, not the processor.** Export always
   reports success upward; retries (transport errors and 5xx, with backoff) are
   the buffer's job. A `4xx` is terminal and is not retried.

### Lossy vs. non-lossy buffers — a deliberate asymmetry

Trace backlog is **bounded** (by count and age): if the platform is unreachable
long enough, old spans are dropped *from the re-send queue* — but **kept
locally**. Checkpoint backlog is **never** abandoned: an active or paused run's
durability must not be sacrificed to a bound. Telemetry is expendable; execution
state is not.

## The push/pull boundary

The clean formulation: **telemetry and definitions flow up; platform-owned
resources are read down; nothing is ever synced down onto your code.**

**Pushed up** — traces, HITL events, checkpoints, governance enrollment (all
background), plus explicitly-called publishes: prompts, eval results, eval
datasets. Agents are pushed too, but only by code you write — there's no
`Agent.push()`, because *agents are code* and shouldn't be silently synced.

**Read down** — these fetch resources the platform owns, which is different from
syncing your local objects: `PromptRegistry.get(...)`, `Dataset.from_platform`,
`Replay.from_platform`, `PlatformKB.search`, plane-backed memory blocks, and the
cached governance policy.

!!! warning "Three honest exceptions to 'push-only'"
    - **Checkpoint restore** genuinely pulls state down — the platform serves a
      saved checkpoint back and the **SDK resumes locally**. State can descend;
      *execution* never does.
    - **The runner channel** long-polls for commands. This is the one place the
      platform initiates work in the SDK.
    - **HITL approval polling** blocks a run on platform state.

    So the accurate claim is "no sync-down of agent/tool/chain *definitions*,"
    not "no pull at all."

### Loud vs. quiet failure

A useful rule when something silently doesn't happen:

- **Setup-time APIs are loud.** `Dataset.from_platform`, `Replay.from_platform`,
  and friends raise `PlatformNotConnectedError`.
- **Run-time paths are quiet.** Prompt lookups fall back to local, plane-backed
  memory degrades, background exports swallow and log. This is intentional — see
  below.

## Never block the agent hot path

Every platform interaction is designed so a slow or dead plane cannot break a
run:

- All network work happens on background threads; retry backoff never sleeps in
  your agent's path.
- Export always returns success; the buffer owns retries.
- Background paths wrap failures and log at debug rather than raising.
- Even `connect()` itself doesn't block on the plane — an unreachable target is
  swallowed, and traces simply queue locally.
- A governance *denial* is returned to the model as a refusal string, so the
  agent continues reasoning instead of crashing.

## Governance: fail-open gate, fail-closed decision

Managed governance has an asymmetry that's easy to misread:

- **The gate is fail-open.** With no `agent_id`, no connection, or no policy
  pattern matching the tool name, there is no gating and **no network call at
  all**. Unmanaged runs are untouched.
- **The decision is fail-closed.** Once a policy *does* match, an unreachable
  plane means refuse — not allow.
- `governance_fail_mode="closed"` closes the remaining gap, refusing when the
  policy cache itself couldn't be loaded.

A matched call resolves to `allow`, `deny` (a refusal string handed back to the
model), or `require_approval` — which raises through the same `interrupt()`
machinery as [durability](../durability/concepts.md), checkpointing and pausing
the run until a human decides.

## Tenancy comes from the key

`connect()` verifies the API key and the **server** returns the domain, project,
and scopes. The `project` argument is only an override for payload labeling —
you can't widen your own access by passing a different string.

## Next steps

- [Platform Connection](index.md) — setup, the per-service table, offline behavior
- [Connected governance](connected-governance.md) · [Connected HITL](connected-hitl.md)
- [Connected checkpoints](../durability/connected-checkpoints.md) — replication and restore
- [Platform KB](../knowledge-base/platform-kb.md) — hosted retrieval
- [Tracing](../tracing/concepts.md) — the local store the drain reads from
