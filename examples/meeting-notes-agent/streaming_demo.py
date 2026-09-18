"""
Streaming Demo — observe the chain trace landing in real time.

Run: python streaming_demo.py

The most interesting moment in this chain is the ``analyze`` node — three
LLM calls fire concurrently inside one tool. With OTel tracing the spans are
written to ``local.db`` as each call completes; this demo tails the trace
store every 250 ms and prints any new span. You'll see three
``agent.<name>`` / ``llm.openai.gpt-4o`` pairs land close together as each
analyzer returns — that near-simultaneous arrival *is* the parallelism.

What you won't see is a span per chain node: the chain executor traces the
``chain.<name>`` root only, so ``load`` / ``analyze`` / ``merge`` have no
spans of their own. The ``chain.meeting-notes`` root arrives last, because a
root span is written when it ends.
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


async def stream(transcript_path: str) -> None:
    chain = build_chain()
    deps = make_deps()
    ctx = fa.RunContext(state=deps)
    execution_id = f"stream-{uuid.uuid4().hex[:8]}"

    print(f"\nTranscript: {transcript_path}")
    print(f"Execution: {execution_id}")
    print("─" * 64)

    started = time.monotonic()

    async def _run():
        return await chain.aexecute(
            {"path": transcript_path}, execution_id=execution_id, context=ctx
        )

    task = asyncio.create_task(_run())

    from fastaiagent.trace.storage import TraceStore

    # ``TraceStore`` is a trace-level query API (``list_traces`` / ``get_trace``);
    # there is no span-level tail, and no way to filter by execution_id while the
    # run is in flight — ``chain.execution_id`` is set on the ``chain.*`` root
    # span, and a root span is only written when it *ends*. So we tail by trace
    # instead: snapshot the trace ids already in local.db (it is shared across
    # runs), and treat anything new as ours. Child spans are written by
    # ``LocalStorageProcessor.on_end`` the instant each one completes, which is
    # what makes the three concurrent ``llm.*`` spans visible as they land.
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
    # One last pass: the ``chain.*`` root span ends as ``aexecute`` returns, so
    # it is never visible to the polling loop above.
    drain()
    print("─" * 64)
    print(f"Status: {result.status} in {int((time.monotonic() - started) * 1000)} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", default="fixtures/sample_transcript.md")
    args = parser.parse_args()
    asyncio.run(stream(args.transcript))


if __name__ == "__main__":
    main()
