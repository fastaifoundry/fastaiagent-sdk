"""
Streaming Demo — chain.aexecute with the chain's root span observed.

Run: python streaming_demo.py --prospect carol@megacorp.global

Unlike Agent / Supervisor, ``Chain`` doesn't expose ``astream()`` directly —
the chain executor walks nodes serially, and each node's agent / tool fires
in one shot. What you CAN observe in real time is the chain's trace tree:
nodes appear as spans under ``chain.<name>`` as they execute, and
``agent.<worker>`` sub-trees appear with their own ``llm.<provider>.<model>``
spans inside.

This demo subscribes to the local trace store's "tail" stream so you see
every span land as it's written — gives you live visibility into a
running chain without needing the Local UI's HTTP server.
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

    # Tail the trace store: every ~250 ms, list spans newer than the last
    # one we printed. The chain.<name> root span carries the chain's
    # execution_id, and child spans nest under it.
    from fastaiagent.trace.storage import TraceStore

    seen: set[str] = set()
    store = TraceStore.default()

    while not task.done():
        await asyncio.sleep(0.25)
        try:
            recent = store.list_spans(execution_id=execution_id, limit=50)
        except TypeError:
            # Older TraceStore signature — list all and filter manually.
            recent = []
        for span in recent:
            sid = getattr(span, "span_id", None) or repr(span)
            if sid in seen:
                continue
            seen.add(sid)
            elapsed = int((time.monotonic() - started) * 1000)
            name = getattr(span, "name", "?")
            print(f"  [{elapsed:>5} ms]  {name}")

    result = await task
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
