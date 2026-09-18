"""
Streaming Demo — chain.aexecute with the chain's root span observed.

Run: python streaming_demo.py --prospect carol@megacorp.global

Unlike Agent / Supervisor, ``Chain`` doesn't expose ``astream()`` directly —
the chain executor walks nodes serially, and each node's agent / tool fires
in one shot. What you CAN observe in real time is the chain's trace tree:
each agent a tool node wraps produces an ``agent.<worker>`` sub-tree with its
own ``llm.<provider>.<model>`` span inside, and those land as they finish.

Note what is *not* traced: the chain executor opens one ``chain.<name>``
root span and no span per node, so ``enrich`` / ``score`` / ``draft`` have no
spans of their own. What you see instead is what the wrapped agents do —
``agent.lead-scorer``, its ``llm.*`` call, and the ``tool.icp_kb_search`` /
``retrieval.*`` pairs from its KB lookups. The ``chain.<name>`` root arrives
last, since a span is written when it ends.

This demo tails the local trace store so you see every span land as it's
written — live visibility into a running chain without needing the Local
UI's HTTP server.
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from tools import make_deps
from workflow import build_chain


async def stream_topic(prospect_email: str) -> None:
    chain = build_chain()
    deps = make_deps()
    ctx = fa.RunContext(state=deps)
    execution_id = f"sdr-stream-{uuid.uuid4().hex[:8]}"

    print(f"\nProspect: {prospect_email}")
    print(f"Execution: {execution_id}")
    print("=" * 60)

    started = time.monotonic()

    # Run the chain in a task so we can poll the trace store concurrently.
    async def _run():
        return await chain.aexecute(
            {"prospect_email": prospect_email},
            execution_id=execution_id,
            context=ctx,
        )

    task = asyncio.create_task(_run())

    # Tail the trace store. ``TraceStore`` is a trace-level query API
    # (``list_traces`` / ``get_trace``) — there is no span-level listing, and no
    # way to filter by execution_id mid-run: ``chain.execution_id`` lives on the
    # ``chain.<name>`` root span, which is only written once it ends. So we
    # snapshot the trace ids already in local.db (it is shared across runs) and
    # treat any new trace as ours. Child spans are written by
    # ``LocalStorageProcessor.on_end`` the instant each one completes.
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore.default()
    known_traces = {t.trace_id for t in store.list_traces()}
    live_traces: set[str] = set()
    seen: set[str] = set()

    def drain() -> None:
        for summary in store.list_traces():
            if summary.trace_id not in known_traces:
                live_traces.add(summary.trace_id)
        for trace_id in sorted(live_traces):
            for span in store.get_trace(trace_id).spans:
                if span.span_id in seen:
                    continue
                seen.add(span.span_id)
                elapsed = int((time.monotonic() - started) * 1000)
                print(f"  [{elapsed:>5} ms]  {span.name}")

    while not task.done():
        await asyncio.sleep(0.25)
        drain()

    result = await task
    # Final pass — the ``chain.<name>`` root ends as ``aexecute`` returns, so
    # the polling loop above never sees it.
    drain()
    print("─" * 60)
    if result.status == "paused":
        print(f"Status: paused — {result.pending_interrupt}")
    else:
        print(f"Status: {result.status}")
    print(f"Total: {int((time.monotonic() - started) * 1000)} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prospect",
        default="carol@megacorp.global",
        help="Prospect email (must exist in tools.py mock corpus)",
    )
    args = parser.parse_args()
    asyncio.run(stream_topic(args.prospect))


if __name__ == "__main__":
    main()
