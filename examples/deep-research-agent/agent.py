"""
Deep Research Agent — FastAIAgent SDK Template (Scope → parallel Research → Write)

A long-horizon research workflow that mirrors the architecture popularized by
LangChain's Open Deep Research and Anthropic's research agents:

    ScopeAgent  ──→  ResearchBrief
                       │
                       ▼
                 ┌──────┴──────┐
                 ▼      ▼      ▼
             Researcher × N  (parallel via asyncio.gather)
                 │      │      │
                 └──────┬──────┘
                        ▼
                  ResearchFindings × N
                        │
                        ▼
                   WriteAgent  ──→  Markdown report

Why this shape: the empirical lesson from Open Deep Research is to
parallelize *information gathering* and serialize *writing*. Multi-agent
synthesis tends to produce disjoint sections; a single one-shot writer
keeps the report coherent.

Trace shape:

  deep_research.session            ← template.kind="deep-research", topic, plan
    ├── deep_research.scope        ← ResearchBrief (structured)
    ├── deep_research.research × N ← ResearchFindings per subtopic (structured, parallel)
    └── deep_research.write        ← report metadata (chars, citation count)

Plan, brief, and findings are persisted as JSON in span attributes under
the ``fastaiagent.research.*`` namespace. The session span also carries
``fastaiagent.template.kind = "deep-research"`` so the local UI can
identify and filter these runs without parsing span names. The local UI
/ replay tooling can reconstruct everything from the spans.

Usage:
    python agent.py --topic "Current state of MCP server adoption in 2026"
    python agent.py --connect       # also push traces to FastAIAgent Platform

Companion files:
    tools.py          — web_search (Tavily/Brave/Serper/mock) + web_fetch
    topology.py       — Scope + Researcher + Writer agent factories
    spans.py          — structured research-span helpers
    memory_setup.py   — placeholder for the trace-learning loop (see PR B)
    streaming_demo.py — astream variant
    replay_demo.py    — replay from local.db
    eval_suite.py     — golden questions + scorers
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

import spans
from memory_setup import build_memory
from tools import make_deps
from topology import (
    ResearchBrief,
    ResearchFindings,
    build_researcher,
    build_scope_agent,
    build_writer_agent,
)

import fastaiagent as fa
from fastaiagent.trace import trace_context
from fastaiagent.trace.span import set_template_kind


async def _run_scope(
    topic: str, ctx: fa.RunContext, memory: object | None = None
) -> ResearchBrief:
    """Phase 1 — produce a structured research brief from the user's topic."""
    with trace_context("deep_research.scope") as span:
        agent = build_scope_agent()
        if memory is not None:
            agent.memory = memory  # type: ignore[assignment]
        result = await agent.arun(topic, context=ctx)
        brief = result.parsed
        if not isinstance(brief, ResearchBrief):
            raise RuntimeError(
                f"ScopeAgent did not return a ResearchBrief. "
                f"Got parsed={type(brief).__name__!r}, raw output={result.output[:200]!r}"
            )
        spans.set_brief(span, brief)
        return brief


async def _run_one_researcher(
    subtopic_title: str,
    rationale: str,
    ctx: fa.RunContext,
) -> ResearchFindings:
    """Run a single sub-research branch in its own span."""
    with trace_context("deep_research.research") as span:
        spans.set_subtopic(span, subtopic_title)
        agent = build_researcher(subtopic_title, rationale)
        # The user message just kicks the agent off; the system prompt has
        # the actual instructions including the assigned subtopic.
        result = await agent.arun(
            f"Research your assigned subtopic: {subtopic_title}",
            context=ctx,
        )
        findings = result.parsed
        if not isinstance(findings, ResearchFindings):
            # Tolerate: produce empty findings rather than crash the pipeline.
            findings = ResearchFindings(
                subtopic=subtopic_title,
                summary=(
                    f"(researcher returned unstructured output: "
                    f"{(result.output or '')[:200]!r})"
                ),
                citations=[],
            )
        spans.set_findings(span, findings)
        return findings


async def _run_research_phase(
    brief: ResearchBrief,
    ctx: fa.RunContext,
) -> list[ResearchFindings]:
    """Phase 2 — run all sub-researchers in parallel via asyncio.gather."""
    tasks = [
        _run_one_researcher(st.title, st.rationale, ctx)
        for st in brief.subtopics
    ]
    return await asyncio.gather(*tasks)


def _build_writer_input(
    brief: ResearchBrief,
    all_findings: list[ResearchFindings],
) -> str:
    """Assemble a single user message for the writer with brief + findings."""
    parts = [
        f"# Research Brief\n\n**Topic:** {brief.topic}\n\n{brief.summary}\n",
        "## Findings by subtopic\n",
    ]
    for f in all_findings:
        parts.append(f"### {f.subtopic}\n\n{f.summary}\n")
        if f.citations:
            parts.append("**Sources:**\n")
            for c in f.citations:
                parts.append(f"- [{c.title}]({c.url}) — {c.relevance}")
            parts.append("")
    parts.append(
        "Compose the final Markdown report following the format in your "
        "system prompt. Citations must come from the Sources lists above."
    )
    return "\n".join(parts)


async def _run_write(
    brief: ResearchBrief,
    all_findings: list[ResearchFindings],
    ctx: fa.RunContext,
    memory: object | None = None,
) -> str:
    """Phase 3 — single one-shot LLM call that composes the final report."""
    with trace_context("deep_research.write") as span:
        agent = build_writer_agent()
        if memory is not None:
            agent.memory = memory  # type: ignore[assignment]
        user_msg = _build_writer_input(brief, all_findings)
        result = await agent.arun(user_msg, context=ctx)
        report = result.output if isinstance(result.output, str) else str(result.output)
        spans.set_report_metadata(span, report)
        return report


async def run_deep_research(topic: str) -> str:
    """End-to-end pipeline. Returns the final Markdown report."""
    deps = make_deps()
    memory = build_memory()
    # ``RunContext.state`` carries the typed deps (search backend config, trail,
    # etc.) through to every tool call. Memory attaches per-Agent, not per-ctx.
    ctx = fa.RunContext(state=deps)

    with trace_context("deep_research.session") as session_span:
        set_template_kind(session_span, "deep-research")
        spans.set_topic(session_span, topic)

        # Memory is wired into the judgment-heavy phases (scope + write).
        # Researchers stay un-memory'd — they're stateless workers.
        brief = await _run_scope(topic, ctx, memory=memory)
        spans.set_plan(session_span, brief)

        all_findings = await _run_research_phase(brief, ctx)

        report = await _run_write(brief, all_findings, ctx, memory=memory)

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Research Agent")
    parser.add_argument(
        "--topic",
        default="Current state of MCP (Model Context Protocol) adoption",
        help="Research topic to investigate (default: %(default)r)",
    )
    parser.add_argument(
        "--connect",
        action="store_true",
        help="Also push traces to FastAIAgent Platform",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny offline-friendly research query and exit 0 on success.",
    )
    args = parser.parse_args()

    if args.connect:
        api_key = os.getenv("FASTAIAGENT_API_KEY")
        project = os.getenv("FASTAIAGENT_PROJECT", "deep-research-agent")
        target = os.getenv("FASTAIAGENT_TARGET", "https://app.fastaiagent.net")
        if api_key:
            fa.connect(api_key=api_key, target=target, project=project)
            print(f"Connected to FastAIAgent Platform (project: {project})")
        else:
            print("FASTAIAGENT_API_KEY not set — running without platform connection")

    if args.self_test:
        # ``--self-test`` is wired into the test gate. Use a topic the mock
        # backend covers so this works without TAVILY_API_KEY.
        topic = "Retrieval-augmented generation"
    else:
        topic = args.topic

    print(f"\nTopic: {topic}")
    print("=" * 60)
    report = asyncio.run(run_deep_research(topic))
    print(report)


if __name__ == "__main__":
    main()
