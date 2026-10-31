"""
Replay Demo — read the sales-SDR chain's trace.

Run: python replay_demo.py

The supervisor in ``research-agent`` had a fairly tall trace tree because every
delegation added a worker sub-tree. This Chain is flatter — one chain root,
one tool span per node — and the same Replay primitives read it: load by
trace_id, list the steps, inspect one. Replay reruns standalone agent traces
only, so forking this chain trace is refused (1.95.0); a chain is forked from
its checkpoint with ``chain.afork``.

Note: this demo runs the chain to *qualified path* but does not actually
trigger the HITL gate on send. To exercise resume across HITL, see
``agent.py``'s ``run_one`` which loops ``aresume()`` until the run is no
longer paused.
"""

from __future__ import annotations

import asyncio
import uuid

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa
from fastaiagent._internal.errors import ReplayError

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

    # ─── Step 4: Inspect one step ───────────────────────────────────────────
    if len(steps) > 1:
        step = replay.inspect(1)
        print(f"\nStep 3: Inspecting step [1] {step.span_name}")
        print(f"   attributes: {sorted(step.attributes)[:6]} ...")

    # ─── Step 5: Why there is no fork ───────────────────────────────────────
    # A rerun rebuilds a standalone Agent from the trace's root span, and a
    # chain root has none, so fork_at() refuses a chain trace (1.95.0) instead
    # of rerunning a default agent and reporting success. A chain is forked
    # from its checkpoint: chain.afork(execution_id, checkpoint_id=..., input=...).
    print("\nStep 4: Forking a Chain trace\n")
    try:
        replay.fork_at(0)
    except ReplayError as e:
        print(f"   Refused, as it should be: {str(e)[:160]}...")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
