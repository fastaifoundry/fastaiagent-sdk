# Checkpointing

Chains automatically checkpoint state after each node. If execution fails, you can resume from the last successful checkpoint instead of restarting from scratch.

## How It Works

```python
from fastaiagent.chain.checkpoint import CheckpointStore

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
    print(f"Failed at: {e}")
    print(f"Execution ID: {result.execution_id}")  # Save this
```

After each node completes, the chain state is saved to a checkpoint store (SQLite by default). The `execution_id` in the result identifies this particular run.

## Resuming from Checkpoint

```python
# Resume from where it failed
result = await chain.resume(
    execution_id="<saved-execution-id>",
    modified_state={"retry_count": 1},  # Optional: modify state before resuming
)
```

The chain loads the last successful checkpoint, optionally merges in your modified state, and continues execution from the next node.

## Custom Checkpoint Store

```python
store = CheckpointStore(db_path="/path/to/checkpoints.db")
chain = Chain("my-pipeline", checkpoint_store=store)
```

## Inspecting Checkpoints

```python
store = CheckpointStore()
checkpoints = store.load(execution_id="<id>")
for cp in checkpoints:
    print(f"Node: {cp.node_id}, State: {cp.state_snapshot}")

latest = store.get_latest(execution_id="<id>")
print(f"Last completed: {latest.node_id}")
```

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
