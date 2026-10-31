"""
Replay Demo — read the meeting-notes chain's trace.

Run: python replay_demo.py

This demo runs the chain on the sample transcript, loads the trace, lists
every step in it and inspects one. Replay reruns standalone agent traces
only, so forking this chain trace is refused (1.95.0); a chain is forked
from its checkpoint with ``chain.afork``. A chain's trace is for reading.
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

    print("\n" + "=" * 64)
    print("  Meeting Notes — Replay Demo")
    print("=" * 64)

    transcript = "fixtures/sample_transcript.md"
    print(f"\nStep 1: Running chain on {transcript!r}\n")
    result = await chain.aexecute(
        {"path": transcript}, execution_id=execution_id, context=ctx
    )
    print(f"   Status: {result.status}")
    print(
        "   action_items:",
        len((result.final_state.get("output") or {}).get("action_items", [])),
    )

    # Pull the latest chain trace.
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore.default()
    rows = store.list_traces(name_filter="chain.meeting-notes", limit=1)
    if not rows:
        print("\n   No trace found for chain.meeting-notes; aborting.")
        return
    trace_id = rows[0].trace_id

    print(f"\nStep 2: Loading trace {trace_id}\n")
    replay = fa.Replay.load(trace_id)
    print(replay.summary())
    steps = replay.steps()
    print(f"\n   Total steps: {len(steps)}")

    if len(steps) > 1:
        step = replay.inspect(1)
        print(f"\nStep 3: Inspecting step [1] {step.span_name}")
        print(f"   attributes: {sorted(step.attributes)[:6]} ...")

    # A chain is forked from its checkpoint, not its trace: Replay rebuilds a
    # standalone Agent from a trace's root span, and a chain root has none, so
    # fork_at() refuses a chain trace (1.95.0) instead of rerunning a default
    # agent and reporting success. Fork the run itself with
    # chain.afork(execution_id, checkpoint_id=..., input=...).
    print("\nStep 4: Forking a Chain trace\n")
    try:
        replay.fork_at(0)
    except ReplayError as e:
        print(f"   Refused, as it should be: {str(e)[:160]}...")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
