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
from fastaiagent._internal.errors import GuardrailBlockedError, ReplayError, TraceError

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
    # Picking a query that does not trip the HITL gate in create_ticket so
    # the demo runs unattended. For a HITL flow + Replay combo, use
    # `_drive_until_complete` from agent.py and pass the resulting trace_id.
    result = await agent.arun(
        "Can you look up the account for alice@acme.com and tell me the plan?",
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
    except (ReplayError, TraceError) as e:
        print(f"\n   Could not load replay: {e}")
        print("   (Replay requires traces to be stored in TraceStore)")
        return
    print(replay.summary())

    steps = replay.steps()
    print(f"\n   Total steps: {len(steps)}")
    if len(steps) <= 1:
        print("\n   Only one step — skipping fork demo.")
        return

    # ─── Step 3: Fork from a step ────────────────────────────────────────────
    fork_step = min(1, len(steps) - 1)
    print(f"\nStep 3: Forking from step [{fork_step}]...\n")

    # A rerun blocks every tool unless you allow it (1.89.0). Under "block" the
    # read-only lookups named in ``allow`` run again and create_ticket stays
    # blocked, so the rerun cannot open a real ticket. The tools take a
    # RunContext[Deps], which a trace does not record: pass it (1.95.0).
    forked = (
        replay.fork_at(fork_step)
        .with_tool_policy("block", allow=["search_kb", "lookup_account", "check_order_status"])
        .with_context(ctx)
        .modify_input("Can you look up the account for bob@startup.io and tell me the plan?")
    )
    try:
        forked_result = forked.rerun()
    except GuardrailBlockedError as e:
        # The rerun runs the agent's guardrails too; the PII output filter can
        # block an answer that repeats an email address.
        print(f"   The rerun's answer was blocked by a guardrail: {e}")
        return
    print(f"   Forked result: steps_executed={forked_result.steps_executed}")
    print(f"   Tools: {forked_result.tool_dispositions}")
    print(f"   Original trace: {trace_id}")

    # ─── Step 4: Compare ─────────────────────────────────────────────────────
    print("\nStep 4: Comparison\n")
    comparison = forked.compare(forked_result)
    print(f"   Original steps: {len(comparison.original_steps)}")
    print(f"   Diverged at: step {comparison.diverged_at}")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
