"""Benchmark: Replay load. Target: 1000-span trace <2s."""

import tempfile
import time

from fastaiagent.trace.replay import Replay
from fastaiagent.trace.storage import SpanData, TraceData


def bench_replay_load(num_spans=1000):
    spans = [
        SpanData(
            span_id=f"span_{i:06d}",
            trace_id="bench_trace",
            name=f"operation_{i}",
            start_time=f"2025-01-01T00:00:{i % 60:02d}Z",
            end_time=f"2025-01-01T00:00:{(i + 1) % 60:02d}Z",
            attributes={"index": i, "data": f"payload_{i}"},
        )
        for i in range(num_spans)
    ]
    trace = TraceData(
        trace_id="bench_trace",
        name="benchmark-trace",
        start_time=spans[0].start_time,
        end_time=spans[-1].end_time,
        spans=spans,
    )

    start = time.monotonic()
    replay = Replay(trace)
    steps = replay.steps()
    _ = replay.summary()
    elapsed = time.monotonic() - start

    return elapsed, len(steps)


if __name__ == "__main__":
    n = 1000
    elapsed, steps = bench_replay_load(n)
    print(f"Spans: {n}")
    print(f"Steps loaded: {steps}")
    print(f"Load time: {elapsed:.3f}s")
    print(f"Target: <2s — {'PASS' if elapsed < 2 else 'FAIL'}")
