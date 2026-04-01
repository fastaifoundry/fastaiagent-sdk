"""Benchmark: Tracing overhead. Target: <5% of execution time."""

import time

from fastaiagent.trace.otel import get_tracer, reset


def work():
    """Simulate agent work."""
    total = 0
    for i in range(10000):
        total += i * i
    return total


def bench_no_trace(iterations=1000):
    start = time.monotonic()
    for _ in range(iterations):
        work()
    return time.monotonic() - start


def bench_with_trace(iterations=1000):
    tracer = get_tracer()
    start = time.monotonic()
    for _ in range(iterations):
        with tracer.start_as_current_span("bench-work"):
            work()
    return time.monotonic() - start


if __name__ == "__main__":
    reset()
    n = 1000

    baseline = bench_no_trace(n)
    traced = bench_with_trace(n)
    overhead = ((traced - baseline) / baseline) * 100

    print(f"Iterations: {n}")
    print(f"Baseline: {baseline:.3f}s")
    print(f"With tracing: {traced:.3f}s")
    print(f"Overhead: {overhead:.1f}%")
    print(f"Target: <5% — {'PASS' if overhead < 5 else 'FAIL'}")
    reset()
