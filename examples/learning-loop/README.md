# Trace Learning Loop — minimal demo

The smallest possible illustration of the learning loop introduced in `fastaiagent.learn`:

1. Run an `Agent` a few times → traces land in `local.db`.
2. Run the offline extractor (`run_extraction`) → durable facts persist in the new `learned_memory` table.
3. Build a new agent with `PersistentFactBlock` → next run automatically gets those facts injected as a system message.

```sh
pip install -r requirements.txt
python agent.py
```

To inspect the persisted facts via the local UI:

```sh
fastaiagent ui
# open http://127.0.0.1:7843 → /api/learned_memory
```

Or from the CLI:

```sh
fastaiagent learn list --scope agent --scope-id learning-loop-demo
```

## What's not in this demo

This is a single-agent toy to keep the loop visible. For a realistic closed loop where the learned facts measurably improve a flagship template, see [`examples/self-improving-research`](../self-improving-research/).
