# Checkpointing

Chains automatically checkpoint state after each node. If execution fails, you can resume from the last successful checkpoint instead of restarting from scratch.

Checkpointing is built on a `Checkpointer` protocol with pluggable backends. The default is `SQLiteCheckpointer`, which writes to the unified `local.db`.

> **Durability covers more than crash recovery.** This page documents the
> Chain-level checkpoint API. For suspending HITL via `interrupt()`,
> resume entry points (Python / HTTP / CLI), the `/approvals` UI,
> Postgres for production, and the `@idempotent` decorator that makes
> resumed nodes safe to re-run, see the
> [Durability section](../durability/index.md).

## How It Works

```python
from fastaiagent import Chain

# Enable checkpointing (on by default)
chain = Chain("my-pipeline")
chain.add_node("step1", agent=agent1)
chain.add_node("step2", agent=agent2)
chain.add_node("step3", agent=agent3)
chain.connect("step1", "step2")
chain.connect("step2", "step3")

# First run — step2 might fail
try:
    result = chain.execute({"input": "data"})
except Exception as e:
    print(f"Failed: {e}")
```

After each node completes, the chain state is saved to a checkpoint. The `execution_id` returned in `ChainResult` identifies the run for resume.

## Resuming from a Checkpoint

```python
# Resume from where it failed
result = await chain.resume(
    execution_id="<saved-execution-id>",
    modified_state={"retry_count": 1},  # Optional: patch state before resuming
)
```

The chain loads the last successful checkpoint, optionally merges in your modified state, and continues from the next node.

## Custom Checkpoint Backend

The default `SQLiteCheckpointer` is right for local development and
single-process production deployments. For multi-process or distributed
deployments — e.g. an HTTP worker pool that any of N replicas could
serve a `chain.resume(...)` request from — use `PostgresCheckpointer`.

```python
from fastaiagent import Chain, SQLiteCheckpointer
from fastaiagent.checkpointers.postgres import PostgresCheckpointer

# Local development / single-process production:
chain = Chain("flow", checkpointer=SQLiteCheckpointer(db_path="/path/to/local.db"))

# Multi-process production — any worker can resume any execution:
chain = Chain("flow", checkpointer=PostgresCheckpointer(
    "postgresql://user:pass@host/db",
    schema="fastaiagent",  # default; pass a custom name to share one DB
))
```

`PostgresCheckpointer` ships under the `[postgres]` extra
(`pip install 'fastaiagent[postgres]'`). It uses psycopg3 with `JSONB`
columns, `TIMESTAMPTZ` timestamps, a partial index on the
`failed`/`interrupted` rows, and a single `DELETE … RETURNING *` atomic
claim for `chain.resume(...)` — concurrent resumers see `AlreadyResumed`
deterministically.

> **Migration note (v1.0):** the constructor kwarg was renamed from `checkpoint_store` to `checkpointer`, and `CheckpointStore` was renamed to `SQLiteCheckpointer`. There is no deprecation alias — update call sites directly.

## Inspecting Checkpoints

```python
from fastaiagent import SQLiteCheckpointer

store = SQLiteCheckpointer()
store.setup()

for cp in store.list(execution_id="<id>"):
    print(f"Node: {cp.node_id}, State: {cp.state_snapshot}")

latest = store.get_last(execution_id="<id>")
print(f"Last completed: {latest.node_id}")
```

The `Checkpointer` protocol also exposes `get_by_id`, `delete_execution`, and `prune(older_than=…)` for housekeeping.

## Disabling Checkpointing

For lightweight chains that don't need persistence:

```python
chain = Chain("quick-pipeline", checkpoint_enabled=False)
```

## Error Handling

```python
from fastaiagent._internal.errors import ChainCheckpointError

try:
    result = await chain.resume(execution_id="nonexistent-id")
except ChainCheckpointError as e:
    print(f"Checkpoint error: {e}")
```

---

## Next Steps

- [Chains](index.md) — Core chain documentation
- [Cyclic Workflows](cyclic-workflows.md) — Retry loops and exit conditions
- [Human-in-the-Loop](hitl.md) — Pause chains for human approval
- [Replay](../replay/index.md) — Debug failed executions with trace replay
