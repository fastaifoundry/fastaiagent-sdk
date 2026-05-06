"""
Streaming Demo — observe the chain trace landing in real time.

Run: python streaming_demo.py

The most interesting moment in this chain is the ``analyze`` node — three
LLM calls fire concurrently inside one tool. With OTel tracing the
spans are written to ``local.db`` as each call completes; this demo
tails the trace store every 250 ms and prints any new span. You'll see
all three ``llm.openai.gpt-4o`` spans appear nearly simultaneously,
followed by ``tool.merge_into_notes`` once they've all returned.
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

    seen: set[str] = set()
    store = TraceStore.default()

    while not task.done():
        await asyncio.sleep(0.25)
        try:
            recent = store.list_spans(execution_id=execution_id, limit=50)
        except TypeError:
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
    print("─" * 64)
    print(f"Status: {result.status} in {int((time.monotonic() - started) * 1000)} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", default="fixtures/sample_transcript.md")
    args = parser.parse_args()
    asyncio.run(stream(args.transcript))


if __name__ == "__main__":
    main()
