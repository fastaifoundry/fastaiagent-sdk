"""
Replay Demo — fork the chain at a node boundary and rerun.

Run: python replay_demo.py

The interesting fork point here is between ``analyze`` and ``merge`` —
forking just before merge lets you inspect what the three analyzers
returned and rerun merge with hand-edited analyzer payloads. Useful
when debugging "why did merge produce a partial MeetingNotes" without
spending the LLM tokens on a fresh ``analyze`` run.

For a less academic use case, this demo just runs the chain on the
sample transcript, loads the trace, lists every step in the trace tree,
forks at step 1 with a different transcript path, and reruns from there.
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

    if len(steps) <= 1:
        print("   Only one step — skipping fork demo.")
        return

    fork_step = min(1, len(steps) - 1)
    print(f"\nStep 3: Forking at step [{fork_step}]\n")
    forked = replay.fork_at(fork_step)
    forked.modify_input({"path": transcript})

    forked_result = forked.rerun()
    print(f"   Forked result: steps_executed={forked_result.steps_executed}")

    print("\nStep 4: Comparing original vs forked timeline\n")
    comparison = forked.compare(forked_result)
    print(f"   Original steps: {len(comparison.original_steps)}")
    print(f"   Diverged at:    step {comparison.diverged_at}")

    print("\nReplay demo complete.\n")


if __name__ == "__main__":
    asyncio.run(demo())
