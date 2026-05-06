"""
Replay Demo — fork the sales-SDR chain at a node boundary.

Run: python replay_demo.py

The supervisor in ``research-agent`` had a fairly tall trace tree because every
delegation added a worker sub-tree. This Chain is flatter — one chain root,
one tool span per node — but the same Replay primitives apply: load by
trace_id, inspect the steps, fork at a chosen step with modified input,
rerun, compare.

Note: this demo runs the chain to *qualified path* but does not actually
trigger the HITL gate on send (it just inspects + forks the trace before
that point). To exercise resume across HITL, see ``agent.py``'s
``run_one`` which loops ``aresume()`` until the run is no longer paused.
"""

from __future__ import annotations

import asyncio
import uuid

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from tools import make_deps
from workflow import build_chain


async def demo() -> None:
    chain = build_chain()
    deps = make_deps()
    ctx = fa.RunContext(state=deps)
    execution_id = f"replay-{uuid.uuid4().hex[:8]}"

    print("\n" + "=" * 60)
    print("  Sales SDR — Replay Demo")
    print("=" * 60)

    # ─── Step 1: Run the chain (to the HITL pause) ───────────────────────────
    topic_email = "alice@acme-saas.com"
    print(f"\nStep 1: Running chain on {topic_email!r} (qualified path)\n")
    result = await chain.aexecute(
        {"prospect_email": topic_email},
        execution_id=execution_id,
        context=ctx,
    )
    print(f"   Status: {result.status}")
    if result.status == "paused":
        # Decline so we don't actually mock-send during the replay demo;
        # we want a clean trace, not an outbox entry.
        result = await chain.aresume(
            execution_id,
            resume_value=fa.Resume(approved=False, metadata={"approver": "replay-demo"}),
            context=ctx,
        )
        print(f"   After auto-decline: {result.status}")

    # ─── Step 2: Find a trace by browsing local.db ──────────────────────────
    # Chain runs land their root span as ``chain.<name>``. The Replay API
    # accepts the OTel trace_id; we don't have it on ChainResult directly
    # so we pull the latest trace for this chain from the trace store.
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore.default()
    rows = store.list_traces(name_filter="chain.sales-sdr", limit=1)
    if not rows:
        print("\n   No trace found for chain.sales-sdr; aborting.")
        return
    trace_id = rows[0].trace_id

    # ─── Step 3: Load + inspect ─────────────────────────────────────────────
    print(f"\nStep 2: Loading trace {trace_id}\n")
    replay = fa.Replay.load(trace_id)
    print(replay.summary())
    steps = replay.steps()
    print(f"\n   Total steps: {len(steps)}")

    # ─── Step 4: Fork ────────────────────────────────────────────────────────
    if len(steps) <= 1:
        print("   Only one step — skipping fork demo.")
        return
    fork_step = min(1, len(steps) - 1)
    print(f"\nStep 3: Forking at step [{fork_step}] with a different prospect\n")
    forked = replay.fork_at(fork_step)
    forked.modify_input("carol@megacorp.global")  # switch prospect mid-flight

    forked_result = forked.rerun()
    print(f"   Forked result: steps_executed={forked_result.steps_executed}")

    print("\nStep 4: Comparing original vs forked timeline\n")
    comparison = forked.compare(forked_result)
    print(f"   Original steps: {len(comparison.original_steps)}")
    print(f"   Diverged at:    step {comparison.diverged_at}")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
