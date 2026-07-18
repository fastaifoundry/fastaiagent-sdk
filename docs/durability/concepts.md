# Concepts & Mental Model

This page explains how durability actually works *under the hood* — the
checkpoint record, exactly when it's written, how resume decides where to
re-enter, and the atomic operations that make interrupt/resume and idempotency
safe. It's an internals explainer, not a quickstart; for setup see the
[Durability reference](index.md) and [Quickstart](quickstart.md).

## The problem

An agent or chain is a sequence of steps, some of which are expensive (LLM
calls), side-effecting (charge a card), or slow (wait days for a human). If the
process crashes at step 4 of 6, you don't want to redo steps 1–3. If a step
pauses for human approval, you don't want to keep a process alive for three
days. And when execution *does* re-enter a step, you must not fire its side
effects twice.

Durability solves all three by writing the run's progress to a **checkpoint
store** as it goes, so any process with access to that store can resume exactly
where the last one left off.

## The checkpoint record

Everything rests on one row. A `Checkpoint` (`fastaiagent/chain/checkpoint.py`)
captures the state of a run *at one boundary*:

| Field | What it holds |
|-------|---------------|
| `execution_id` | The run identifier — **immutable across resume**; the key everything is looked up by |
| `node_id` | *Where* this checkpoint is — the re-entry address (see the scheme below) |
| `node_index` | Position in topological/turn order |
| `status` | `"completed"`, `"interrupted"`, or `"failed"` — **this drives resume** |
| `state_snapshot` | The full run state at this point, JSON-frozen (chain state, or serialized messages + turn for an agent) |
| `node_input` / `node_output` | The step's inputs (e.g. saved tool args) and result |
| `iteration_counters` | Cycle counts, so a resumed chain loop picks up mid-cycle |
| `interrupt_reason` / `interrupt_context` | For a paused run — the frozen snapshot the human approves |
| `agent_path` | Hierarchical location in multi-agent topologies (see composition) |

The `node_id` is the **re-entry address**, and its scheme tells you what kind of
boundary it is:

- **Chain node** — the node's own id (e.g. `"review"`).
- **Agent turn** — `"turn:N"` (checkpoint written *before* the Nth LLM call).
- **Agent tool** — `"turn:N/tool:<name>"` (written *before* the tool runs, with
  the tool's args in `node_input`).

## When checkpoints are written

Checkpoints are written at boundaries chosen so resume never loses or repeats
committed work:

- **Chain**: after each node completes → a `status="completed"` checkpoint with
  the post-node `state_snapshot` (`chain/executor.py`).
- **Agent, per turn**: `_put_turn_checkpoint` runs *before each LLM call*
  (`turn:N`) — the resume point for a crash mid-inference.
- **Agent, per tool**: `_put_tool_checkpoint` runs *before each tool dispatch*
  (`turn:N/tool:X`), saving the exact args — so resume re-invokes the tool with
  the same input.

`execution_id` is minted once at the start of a run (or supplied by you) and
placed in a `ContextVar` so every node, tool, and `@idempotent` function in that
run reads the same id.

!!! info "Verified against a live run"
    A chain node that called `interrupt()` wrote exactly one checkpoint —
    `node_id='review' status='interrupted' interrupt_reason='manager_approval'`
    — plus a `pending_interrupts` row, and the result came back
    `status="paused"` with `pending_interrupt={reason, context, node_id,
    agent_path}`. That is the whole suspended-run footprint on disk.

## How resume decides where to re-enter

Resume reads the **latest checkpoint's `status`** and branches. This is the
contract (`Chain.aresume` / `Agent.aresume`):

| Latest status | You pass | What happens |
|---------------|----------|--------------|
| `interrupted` | `resume_value=Resume(...)` | Atomically **claim** the pending row, then **re-run the interrupted node from the top** with the resume value in scope — so `interrupt()` *returns* it instead of raising |
| `interrupted` | *nothing* | Raises `ChainResumeError` — an interrupted run needs a `Resume(...)` |
| `completed` / `failed` | *nothing* | Restart at the **next** node after the last committed one |

The subtle, important part is the interrupted case: the paused node is
**re-executed from the beginning**. Everything before the `interrupt()` call
runs again — which is exactly why side effects need `@idempotent` (below). The
resume value is injected via a `ContextVar` set *only for that first
re-executed node*, then cleared, so downstream nodes run normally.

For an agent, the same idea specializes by `node_id`: a `turn:N` crash re-issues
the LLM call with saved history; a `turn:N/tool:X` crash re-invokes the tool
with saved args (no LLM re-call); a tool `interrupt()` re-invokes the tool so
its `interrupt()` returns the `Resume`.

## Interrupt, suspend, and the atomic claim

`interrupt(reason, context)` (`fastaiagent/chain/interrupt.py`) is the whole
suspend mechanism, and it's tiny:

```python
def interrupt(reason, context):
    v = _resume_value.get()
    if v is not None:
        return v                     # resuming: return the human's decision
    raise InterruptSignal(reason, context)   # first pass: suspend
```

First pass, there's no resume value, so it raises `InterruptSignal`. The
executor catches it and, **in one transaction** (`record_interrupt`), writes
both the `interrupted` checkpoint and a `pending_interrupts` row — so the
approvals UI never sees a half-suspended run — then returns `status="paused"`.
The process can now exit.

**Frozen context**: the `context` dict is JSON-serialized at suspend time and
never recomputed. The human approves a specific snapshot, not whatever the world
looks like at resume time.

**Double-resume safety**: resume first *claims* the pending row by deleting it,
and the claim is atomic. On SQLite it's a `BEGIN; SELECT; DELETE; COMMIT` under
a lock; on Postgres it's a single `DELETE ... RETURNING`. Exactly one caller
gets the row back; everyone else gets `None` and raises `AlreadyResumed`. This
is what makes "resume" safe under concurrent resumers *and* against a
resume-after-completion.

!!! info "Verified against a live run"
    Resuming with `Resume(approved=True)` claimed the row, re-entered `review`,
    and `interrupt()` returned the decision → run completed. A **second**
    `resume` for the same `execution_id` raised `AlreadyResumed`.

## Idempotency — absorbing the re-execution

Because a resumed node runs from the top, any side effect before the resume
point fires **again**. That's the footgun. `@idempotent`
(`fastaiagent/chain/idempotent.py`) absorbs it:

```python
@idempotent
def charge_card(amount, account): ...
```

On first execution within a run it runs the body and caches the (JSON-serialized)
result under `(execution_id, key)`. The default `key` is
`sha256(qualname + args + kwargs)`. On any later call in the **same
execution_id** with the same key, it returns the cached value and **never runs
the body again**. In a *different* execution, or outside any chain run (e.g. a
unit test), it's a cache miss and runs normally.

!!! info "Verified against a live run"
    With `charge_card` wrapped in `@idempotent`, running → suspending →
    resuming a high-value charge fired the side effect **exactly once**
    (`calls == 1`), even though the `review` node re-executed on resume.

## The backends

A checkpointer is anything implementing the `Checkpointer` protocol
(`put`, `get_last`, `list`, `record_interrupt`,
`delete_pending_interrupt_atomic`, `get_idempotent`/`put_idempotent`, `prune`).
Two ship in-box, over three tables (`checkpoints`, `pending_interrupts`,
`idempotency_cache`):

- **`SQLiteCheckpointer`** (default) — single-file, great for local/dev and
  single-process. Atomicity comes from an in-process lock + explicit
  transactions. Cross-process coordination relies on SQLite file locking (not
  for distributed use).
- **`PostgresCheckpointer`** — for multi-process/production. The atomic claim is
  a `DELETE ... RETURNING` and correctness rests on Postgres MVCC, so many
  workers can race to resume and exactly one wins.

`prune(older_than)` deletes old `completed`/`failed` checkpoints and idempotency
rows but **never** `interrupted` ones — pruning a suspended run would orphan a
live human-in-the-loop.

## Composition across agents, chains, swarms

All checkpoints for a run share one `execution_id`; the `agent_path` column says
*which* agent/worker wrote each one. It's a `ContextVar` that's **extended, not
overwritten**, as topologies nest:

| Topology | `agent_path` shape |
|----------|--------------------|
| Chain | (none; `node_id` is the rendezvous point) |
| Agent | `agent:<name>` → `agent:<name>/tool:<tool>` |
| Swarm | `swarm:<s>/agent:<a>/tool:<t>` |
| Supervisor | `supervisor:<s>/worker:<r>/tool:<t>` |

So a Supervisor and its workers checkpoint into the same run, and resume can
scope to one agent's subtree by matching the `agent_path` prefix.

## `afork` vs `aresume`

- **`aresume(execution_id, ...)`** continues the *same* run from its last
  checkpoint (crash recovery, or returning a `Resume`).
- **`afork(execution_id, checkpoint_id=..., ...)`** branches a *new*
  `execution_id` from a saved checkpoint (linked via `parent_checkpoint_id`),
  leaving the original intact — for counterfactuals and what-if replays. (This
  is the checkpoint-based cousin of trace-based [Replay](../replay/concepts.md),
  which chains can't use.)

## Next steps

- [Durability reference](index.md) · [Quickstart](quickstart.md)
- [Side effects & idempotency](side-effects.md) — patterns for `@idempotent`
- [Multi-agent](multi-agent.md) — `agent_path` across Swarm/Supervisor
- [Checkpointers](checkpointers.md) — SQLite vs Postgres, custom backends
- [Chains — HITL](../chains/hitl.md) · [Agents — durability](../agents/durability.md)
