"""
Research Agent — FastAIAgent SDK Template (multi-agent / Supervisor)

A research workflow demonstrating Supervisor + Worker delegation with a real
revision loop. The supervisor's LLM coordinates three workers:

    researcher  →  writer  →  verifier
                   ↑           │
                   └─ revise ──┘   (up to 2 rounds; verifier has authority)

Usage:
    python agent.py
    python agent.py --topic "Retrieval-augmented generation"
    python agent.py --connect       # export traces to FastAIAgent Platform

Companion files:
    tools.py          — mock web_search (real-API stubs included)
    topology.py       — Supervisor + 3 worker Agents
    streaming_demo.py — supervisor.astream with handoff events
    replay_demo.py    — fork-and-rerun across handoff boundaries
    eval_suite.py     — Faithfulness + AnswerRelevancy + custom RequiredSources
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from tools import make_deps
from topology import build_supervisor


async def run_topic(topic: str) -> None:
    deps = make_deps()
    ctx = fa.RunContext(state=deps)

    supervisor = build_supervisor()

    print(f"\nTopic: {topic}")
    print("=" * 60)
    result = await supervisor.arun(topic, context=ctx)

    print(result.output)
    print()
    print("─" * 60)
    print(f"  retrieved sources: {len(deps.trail)}")
    print(
        f"  {result.tokens_used} tokens | "
        f"${result.cost:.4f} | "
        f"{result.latency_ms} ms | "
        f"trace_id={result.trace_id}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent Research Agent")
    parser.add_argument(
        "--topic",
        default="Retrieval-augmented generation",
        help="Research topic to investigate (default: %(default)r)",
    )
    parser.add_argument("--connect", action="store_true", help="Connect to FastAIAgent Platform")
    args = parser.parse_args()

    if args.connect:
        api_key = os.getenv("FASTAIAGENT_API_KEY")
        project = os.getenv("FASTAIAGENT_PROJECT", "research-agent")
        target = os.getenv("FASTAIAGENT_TARGET", "https://app.fastaiagent.net")
        if api_key:
            fa.connect(api_key=api_key, target=target, project=project)
            print(f"Connected to FastAIAgent Platform (project: {project})")
        else:
            print("FASTAIAGENT_API_KEY not set — running without platform connection")

    asyncio.run(run_topic(args.topic))


if __name__ == "__main__":
    main()
