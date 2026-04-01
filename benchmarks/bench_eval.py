"""Benchmark: Eval pipeline. Target: 100 cases <60s (excluding LLM)."""

import time

from fastaiagent.eval import Dataset, evaluate
from fastaiagent.eval.builtins import Contains, ExactMatch


def bench_eval(num_cases=100):
    # Create dataset
    items = [{"input": f"Question {i}", "expected": f"answer_{i}"} for i in range(num_cases)]
    dataset = Dataset.from_list(items)

    # Simple agent (no LLM calls)
    def mock_agent(input_text: str) -> str:
        num = input_text.split()[-1]
        return f"The answer is answer_{num}"

    start = time.monotonic()
    results = evaluate(
        agent_fn=mock_agent,
        dataset=dataset,
        scorers=[ExactMatch(), Contains()],
    )
    elapsed = time.monotonic() - start

    return elapsed, results


if __name__ == "__main__":
    n = 100
    elapsed, results = bench_eval(n)
    print(f"Cases: {n}")
    print(f"Eval time: {elapsed:.3f}s")
    print(results.summary())
    print(f"Target: <60s — {'PASS' if elapsed < 60 else 'FAIL'}")
