"""
Agent Replay Demo — Fork-and-rerun debugging.

This script demonstrates the Agent Replay workflow:
1. Run the agent with a query that triggers a tool call
2. Load the trace and inspect it step by step
3. Fork from a specific step with modified input
4. Re-run and compare outcomes

Run: python replay_demo.py
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from agent import agent
from context import create_deps


async def demo():
    deps = await create_deps(user_email="alice@acme.com")
    ctx = fa.RunContext(state=deps)

    # ─── Step 1: Run the agent ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Agent Replay Demo")
    print("=" * 60)

    print("\nStep 1: Running agent with a support query...\n")
    result = await agent.arun(
        "I was charged twice on my last invoice. My email is alice@acme.com. "
        "Can you look into this and fix it?",
        context=ctx,
    )
    print(f"   Agent: {result.output[:200]}...")
    print(f"   {result.tokens_used} tokens | ${result.cost:.4f} | {result.latency_ms}ms")

    trace_id = result.trace_id
    if not trace_id:
        print("\n   No trace captured. Make sure tracing is enabled.")
        return

    # ─── Step 2: Load and inspect the trace ──────────────────────────────────
    print(f"\nStep 2: Loading trace {trace_id}...\n")
    try:
        replay = fa.Replay.load(trace_id)
        print(replay.summary())

        steps = replay.steps()
        print(f"\n   Total steps: {len(steps)}")

        # ─── Step 3: Fork from a step ────────────────────────────────────────
        if len(steps) > 1:
            fork_step = min(1, len(steps) - 1)
            print(f"\nStep 3: Forking from step [{fork_step}]...\n")

            forked = replay.fork_at(fork_step)
            forked.modify_input({"email": "bob@startup.io"})

            forked_result = forked.rerun()
            print(f"   Forked result: steps_executed={forked_result.steps_executed}")
            print(f"   Original trace: {trace_id}")

            # ─── Step 4: Compare ─────────────────────────────────────────────
            print(f"\nStep 4: Comparison\n")
            comparison = forked.compare(forked_result)
            print(f"   Original steps: {len(comparison.original_steps)}")
            print(f"   Diverged at: step {comparison.diverged_at}")
        else:
            print("\n   Only one step — skipping fork demo.")

    except Exception as e:
        print(f"\n   Could not load replay: {e}")
        print("   (Replay requires traces to be stored in TraceStore)")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
