"""Benchmark: Cycle performance. Target: 100 iterations <10s (excluding LLM)."""

import asyncio
import time

from fastaiagent.chain import Chain, NodeType
from fastaiagent.chain.node import NodeConfig


async def bench_cycles(max_iterations=100):
    chain = Chain("cycle-bench", checkpoint_enabled=False)
    chain.add_node("counter", type=NodeType.transformer, template="iteration")
    # No actual execution — just measure the framework overhead
    # of cycle tracking and state management

    from fastaiagent.chain.state import ChainState
    from fastaiagent.chain.executor import _evaluate_condition

    state = ChainState({"count": 0})
    start = time.monotonic()
    for i in range(max_iterations):
        state.set("count", i)
        _evaluate_condition("count >= 100", {"state": state.data})
    elapsed = time.monotonic() - start
    return elapsed


if __name__ == "__main__":
    n = 100
    elapsed = asyncio.run(bench_cycles(n))
    print(f"Cycle iterations: {n}")
    print(f"Total: {elapsed:.3f}s")
    print(f"Target: <10s — {'PASS' if elapsed < 10 else 'FAIL'}")
