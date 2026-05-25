"""Benchmark: trace redaction overhead.

Target (top3recommendation.md §3.9): <5ms p95 per span at typical
attribute sizes. Measures the cost of ``_capture_redact`` running
inside ``LocalStorageProcessor.on_end`` — the chokepoint every span
write goes through when a ``RedactionPolicy(mode="capture")`` is
installed.

Two scenarios:

  * **No-policy baseline** — measures the zero-overhead fast path
    that fires when no policy is installed (the default v1.14
    behavior). Should be effectively a no-op.
  * **Active policy** — installs a realistic 4-pattern policy and
    measures the per-span cost over a representative attribute
    payload (LLM messages + tool I/O + agent in/out).

Both report mean, p50, p95, and per-span absolute time so a regression
can be spotted in CI.

Run::

    python benchmarks/bench_redaction_overhead.py
"""

from __future__ import annotations

import json
import statistics
import time

from fastaiagent.trace.redaction import (
    RedactionPolicy,
    _capture_redact,
    set_redaction_policy,
)


# Representative production-ish span payload — LLM messages, tool I/O,
# agent in/out. Sized around 8–10 KB which is realistic for a tool-using
# agent's root span.
def _sample_attrs() -> dict:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. " * 8},
        {"role": "user", "content": "Look up order ORD-12345 and tell me the status. " * 4},
        {"role": "assistant", "content": "Calling lookup_order. " * 4},
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "id": "ORD-12345",
                    "status": "delivered",
                    "delivered_on": "2026-04-03",
                    "secret_api_key": "sk-DEMO12345678901234567890ABCDEFGH",
                    "items": ["MacBook Pro 16-inch", "USB-C cable"],
                }
            ),
        },
        {"role": "assistant", "content": "Order ORD-12345 was delivered on 2026-04-03. " * 4},
    ]
    return {
        "agent.name": "support-bot",
        "fastaiagent.runner.type": "agent",
        "agent.input": "Look up order ORD-12345",
        "agent.output": (
            "Order ORD-12345 was delivered on 2026-04-03. "
            "Internal note: sk-DEMO12345678901234567890ABCDEFGH"
        ),
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.request.messages": json.dumps(messages),
        "gen_ai.response.content": (
            "Your order ORD-12345 was delivered on April 3, 2026. " * 3
            + "(internal: sk-DEMO12345678901234567890ABCDEFGH)"
        ),
        "tool.name": "lookup_order",
        "tool.input": json.dumps({"order_id": "ORD-12345"}),
        "tool.output": json.dumps(
            {
                "id": "ORD-12345",
                "status": "delivered",
                "delivered_on": "2026-04-03",
            }
        ),
        "fastaiagent.cost.total_usd": 0.0012,
        "gen_ai.usage.input_tokens": 240,
        "gen_ai.usage.output_tokens": 120,
    }


def _report(label: str, samples_ms: list[float]) -> float:
    """Pretty-print mean/p50/p95 and return p95 for the pass-fail gate."""
    mean = statistics.mean(samples_ms)
    p50 = statistics.median(samples_ms)
    samples_sorted = sorted(samples_ms)
    p95 = samples_sorted[int(0.95 * len(samples_sorted))]
    print(f"  {label}:")
    print(f"    mean={mean:.4f}ms  p50={p50:.4f}ms  p95={p95:.4f}ms  n={len(samples_ms)}")
    return p95


def bench_no_policy(iterations: int = 5_000) -> list[float]:
    set_redaction_policy(None)
    attrs = _sample_attrs()
    samples_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _capture_redact(attrs)
        samples_ms.append((time.perf_counter() - t0) * 1_000)
    return samples_ms


def bench_with_policy(iterations: int = 5_000) -> list[float]:
    set_redaction_policy(
        RedactionPolicy(
            patterns=(
                r"sk-[A-Za-z0-9]{20,}",  # API keys
                r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # credit cards
                r"Bearer\s+[A-Za-z0-9\-_\.]+",  # JWTs
                r"\bord-\d+",  # order IDs (mock PII)
            ),
            replacement="[REDACTED]",
            mode="capture",
        )
    )
    attrs = _sample_attrs()
    samples_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _capture_redact(attrs)
        samples_ms.append((time.perf_counter() - t0) * 1_000)
    return samples_ms


P95_TARGET_MS = 5.0


if __name__ == "__main__":
    print(f"Iterations: 5_000  |  attribute payload: ~{len(json.dumps(_sample_attrs()))} bytes")
    print()
    print("Scenario A — no policy installed (zero-overhead fast path):")
    baseline_p95 = _report("baseline", bench_no_policy())
    print()
    print("Scenario B — RedactionPolicy(mode='capture') with 4 regex patterns:")
    active_p95 = _report("active", bench_with_policy())
    print()
    print(f"Target: <{P95_TARGET_MS}ms p95 per span")
    print(
        f"  baseline (no policy): {baseline_p95:.4f}ms — "
        f"{'PASS' if baseline_p95 < P95_TARGET_MS else 'FAIL'}"
    )
    print(
        f"  active redaction:    {active_p95:.4f}ms — "
        f"{'PASS' if active_p95 < P95_TARGET_MS else 'FAIL'}"
    )
    # Reset so subsequent tests in the same process aren't affected.
    set_redaction_policy(None)
    if active_p95 >= P95_TARGET_MS:
        raise SystemExit(1)
