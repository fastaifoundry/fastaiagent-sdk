"""
Replay Demo — read a research run's trace.

Run: python replay_demo.py

Walks through the Replay workflow against a Supervisor trace:
  1. Run the supervisor on a topic, capture the trace.
  2. Load the trace; print every span — including the nested worker spans
     under ``supervisor:research-team/worker:<role>/...``.
  3. Inspect one step.
  4. Try to fork it: Replay reruns standalone agent traces only, so a
     Supervisor trace is refused with a ReplayError (1.95.0). Before, the
     rerun was a default agent that reported success.

This is the multi-agent analogue of ``examples/customer-support-agent/replay_demo.py``
— same primitives, but the trace tree is taller because each worker
delegation adds a sub-tree.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastaiagent._internal.errors import ReplayError

from tools import make_deps
from topology import build_supervisor


async def demo() -> None:
    deps = make_deps()
    ctx = fa.RunContext(state=deps)

    supervisor = build_supervisor()

    print("\n" + "=" * 60)
    print("  Research Agent — Replay Demo")
    print("=" * 60)

    # ── Step 1: Run the supervisor ───────────────────────────────────────────
    topic = "Retrieval-augmented generation"
    print(f"\nStep 1: Running supervisor on topic={topic!r}\n")
    result = await supervisor.arun(topic, context=ctx)
    print(f"   Output (first 200 chars): {result.output[:200]}...")
    print(f"   {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms} ms")

    trace_id = result.trace_id
    if not trace_id:
        print("\n   No trace captured — aborting replay demo.")
        return

    # ── Step 2: Load and inspect ────────────────────────────────────────────
    print(f"\nStep 2: Loading trace {trace_id}\n")
    try:
        replay = fa.Replay.load(trace_id)
    except Exception as e:
        print(f"   Could not load replay: {e}")
        return

    print(replay.summary())
    steps = replay.steps()
    print(f"\n   Total steps: {len(steps)}")

    # ── Step 3: Inspect one step ────────────────────────────────────────────
    if len(steps) > 1:
        step = replay.inspect(1)
        print(f"\nStep 3: Inspecting step [1] {step.span_name}")
        print(f"   attributes: {sorted(step.attributes)[:6]} ...")

    # ── Step 4: Why there is no fork ────────────────────────────────────────
    # A rerun rebuilds a standalone Agent from the trace's root span. This root
    # is the supervisor, which has no agent blueprint, so fork_at() refuses
    # (1.95.0) instead of rerunning a default agent and reporting success.
    print("\nStep 4: Forking a Supervisor trace\n")
    try:
        replay.fork_at(0)
    except ReplayError as e:
        print(f"   Refused, as it should be: {str(e)[:160]}...")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
