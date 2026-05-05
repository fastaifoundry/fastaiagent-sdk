"""
Replay Demo — fork-and-rerun a research run.

Run: python replay_demo.py

Walks through the Replay workflow against a Supervisor trace:
  1. Run the supervisor on a topic, capture the trace.
  2. Load the trace; print every span — including the nested worker spans
     under ``supervisor:research-team/worker:<role>/...``.
  3. Pick a step (default: step 1, just after the supervisor's first LLM
     call) and fork from there with a different topic, then rerun.
  4. Compare the original and forked timelines.

This is the multi-agent analogue of ``examples/customer-support-agent/replay_demo.py``
— same primitives, but the trace tree is taller because each worker
delegation adds a sub-tree.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

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

    # ── Step 3: Fork at a handoff boundary ──────────────────────────────────
    if len(steps) <= 1:
        print("\n   Only one step — skipping fork demo.")
        return

    fork_step = min(1, len(steps) - 1)
    print(f"\nStep 3: Forking from step [{fork_step}] with a swapped topic\n")
    forked = replay.fork_at(fork_step)
    forked.modify_input("Constitutional AI")  # different topic, same workflow

    forked_result = forked.rerun()
    print(f"   Forked: steps_executed={forked_result.steps_executed}")
    print(f"   Original trace: {trace_id}")

    # ── Step 4: Compare ─────────────────────────────────────────────────────
    print("\nStep 4: Comparing original vs forked timeline\n")
    comparison = forked.compare(forked_result)
    print(f"   Original steps: {len(comparison.original_steps)}")
    print(f"   Diverged at:    step {comparison.diverged_at}")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
