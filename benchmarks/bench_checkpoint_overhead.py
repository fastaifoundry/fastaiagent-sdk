"""Benchmark: Checkpoint overhead. Target: <50ms per node."""

import tempfile
import time

from fastaiagent import SQLiteCheckpointer
from fastaiagent.chain.checkpoint import Checkpoint


def bench_checkpoint(num_nodes=100):
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteCheckpointer(db_path=f"{tmp}/bench.db")
        store.setup()
        state = {"key": "value", "data": list(range(100))}

        start = time.monotonic()
        for i in range(num_nodes):
            store.put(
                Checkpoint(
                    chain_name="bench-chain",
                    execution_id="exec-001",
                    node_id=f"node_{i}",
                    node_index=i,
                    state_snapshot=state,
                )
            )
        elapsed = time.monotonic() - start
        per_node_ms = (elapsed / num_nodes) * 1000

        store.close()
        return elapsed, per_node_ms


if __name__ == "__main__":
    n = 100
    elapsed, per_node = bench_checkpoint(n)
    print(f"Checkpoints: {n}")
    print(f"Total: {elapsed:.3f}s")
    print(f"Per node: {per_node:.1f}ms")
    print(f"Target: <50ms — {'PASS' if per_node < 50 else 'FAIL'}")
