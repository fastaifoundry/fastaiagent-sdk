"""Example 04: Agent Replay — fork-and-rerun debugging.

Shows the complete replay workflow:
1. Run a real agent with a tool (produces a traced execution)
2. Load the trace from local storage
3. Step through the execution
4. Fork at a specific step
5. Modify the prompt
6. Rerun with the modification
7. Compare original vs rerun

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/04_agent_replay.py
"""

import os

from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.trace.replay import Replay


def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    orders = {
        "ORD-001": "MacBook Pro 16-inch, shipped 2026-04-01, delivered 2026-04-03",
        "ORD-002": "AirPods Pro, processing, estimated delivery 2026-04-10",
    }
    return orders.get(order_id, f"Order {order_id} not found.")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY=sk-... && python examples/04_agent_replay.py")
        raise SystemExit(0)

    # ── Step 1: Run a real agent ─────────────────────────────────────────
    print("Step 1: Running agent...")
    agent = Agent(
        name="support-bot",
        system_prompt=(
            "You are a customer support agent. Use the lookup_order tool "
            "to check order status. Be concise."
        ),
        llm=LLMClient(provider="openai", model="gpt-4.1"),
        tools=[FunctionTool(name="lookup_order", fn=lookup_order)],
    )

    result = agent.run("What's the status of order ORD-001?")
    print(f"  Output: {result.output}")
    print(f"  Trace ID: {result.trace_id}")
    print(f"  Tool calls: {len(result.tool_calls)}")
    print()

    # ── Step 2: Load the trace from local storage ────────────────────────
    print("Step 2: Loading trace from local storage...")
    assert result.trace_id, "agent.run() did not produce a trace_id"
    replay = Replay.load(result.trace_id)
    print()

    # ── Step 3: View summary and step through ────────────────────────────
    print("Step 3: Execution summary:")
    print(replay.summary())
    print()
    print("  Steps:")
    for step in replay.step_through():
        attrs_summary = ""
        if step.attributes:
            interesting = {
                k: v
                for k, v in step.attributes.items()
                if k in ("agent.name", "tool.name", "tool.status", "gen_ai.request.model")
            }
            if interesting:
                attrs_summary = f"  {interesting}"
        print(f"    [{step.step}] {step.span_name}{attrs_summary}")
    print()

    # ── Step 4: Fork at step 2 ───────────────────────────────────────────
    fork_point = min(2, len(replay.steps()) - 1)
    print(f"Step 4: Forking at step {fork_point}...")
    forked = replay.fork_at(step=fork_point)
    print()

    # ── Step 5: Modify the prompt ────────────────────────────────────────
    new_prompt = (
        "You are a customer support agent. Use the lookup_order tool. "
        "Reply in exactly one sentence. Never use bullet points."
    )
    print(f"Step 5: Modifying prompt to: {new_prompt!r}")
    forked.modify_prompt(new_prompt)
    print()

    # ── Step 6: Rerun ────────────────────────────────────────────────────
    print("Step 6: Rerunning with modified prompt...")
    rerun_result = forked.rerun()
    print(f"  Original output: {rerun_result.original_output}")
    print(f"  New output:      {rerun_result.new_output}")
    print(f"  Rerun trace ID:  {rerun_result.trace_id}")
    print()

    # ── Step 7: Compare ──────────────────────────────────────────────────
    print("Step 7: Comparing original vs rerun...")
    comparison = forked.compare(rerun_result)
    print(f"  Diverged at step: {comparison.diverged_at}")
    print(f"  Original steps:   {len(comparison.original_steps)}")
    print(f"  Rerun steps:      {len(comparison.new_steps)}")
    print()

    print("Done! You can also replay this trace later:")
    print(f"  replay = Replay.load('{result.trace_id}')")
