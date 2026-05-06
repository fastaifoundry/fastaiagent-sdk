"""
Streaming demo — surface progress events as the pipeline runs.

The deep-research pipeline orchestrates three SDK Agents (scope, N
researchers, writer). Each Agent's ``astream`` emits ``TextDelta`` /
``ToolCallStart`` / ``ToolCallEnd`` / ``Usage`` events as the LLM works.

This demo runs the full pipeline and prints a concise progress trace —
one line per phase boundary plus a token tick for each research branch
as it finishes. It does NOT stream every text delta to stdout (that
would be noisy with N parallel branches) — for that, use a single Agent
directly with ``async for event in agent.astream(...)``.

Usage:
    python streaming_demo.py --topic "Self-RAG vs RAG"
"""

from __future__ import annotations

import argparse
import asyncio
import time

from dotenv import load_dotenv

load_dotenv()

from agent import (
    _run_research_phase,
    _run_scope,
    _run_write,
)
from tools import make_deps

import fastaiagent as fa  # noqa: F401  (import side-effects: load tracing)


async def run(topic: str) -> None:
    deps = make_deps()
    ctx = fa.RunContext(state=deps)

    print(f"[scope] starting on topic: {topic!r}")
    t0 = time.perf_counter()
    brief = await _run_scope(topic, ctx)
    print(f"[scope] done in {time.perf_counter() - t0:.1f}s — {len(brief.subtopics)} subtopics:")
    for st in brief.subtopics:
        print(f"        • {st.title}")

    print(f"[research] dispatching {len(brief.subtopics)} parallel branches…")
    t1 = time.perf_counter()
    all_findings = await _run_research_phase(brief, ctx)
    print(f"[research] all branches done in {time.perf_counter() - t1:.1f}s")
    for f in all_findings:
        print(f"        ✓ {f.subtopic} ({len(f.citations)} sources)")

    print("[write] composing final report…")
    t2 = time.perf_counter()
    report = await _run_write(brief, all_findings, ctx)
    print(f"[write] done in {time.perf_counter() - t2:.1f}s — {len(report)} chars")

    print("\n" + "─" * 60 + "\n")
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming demo for deep-research")
    parser.add_argument("--topic", default="Self-RAG vs vanilla RAG")
    args = parser.parse_args()
    asyncio.run(run(args.topic))


if __name__ == "__main__":
    main()
