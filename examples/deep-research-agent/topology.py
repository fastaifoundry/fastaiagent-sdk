"""
Topology — Scope agent, sub-researcher factory, Write agent.

This template uses a three-phase pipeline (Scope → parallel Research →
Write) orchestrated explicitly in ``agent.py`` via ``asyncio.gather`` over
plain ``Agent`` instances. There is no Supervisor — cross-agent
parallelism is the template's responsibility, not the SDK's.

Why: as of this version, the SDK's executor runs tool calls within a
single agent turn sequentially, so going through a Supervisor wouldn't
parallelize the sub-research branches. Orchestrating between agents with
``asyncio.gather`` parallelizes them naturally because each ``Agent.arun``
is independently async.

Structured outputs:

  * ``ScopeAgent`` returns a typed ``ResearchBrief`` with subtopics
  * Each researcher returns a typed ``ResearchFindings`` with citations
  * ``WriteAgent`` returns plain Markdown (humans + UI render this)

Models default to ``gpt-4o`` for scope and write (judgment-heavy phases)
and ``gpt-4o-mini`` for researchers (volume-heavy, cheaper). Override via
``LLM_MODEL_*`` env vars.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field
from tools import web_fetch, web_search

import fastaiagent as fa
from fastaiagent.agent.middleware import ToolBudget

# ─── Structured output schemas ───────────────────────────────────────────────


class Subtopic(BaseModel):
    """A single parallel research track within the larger topic."""

    title: str = Field(..., description="Short, specific question or angle.")
    rationale: str = Field(
        ..., description="One sentence on why this subtopic matters for the brief."
    )


class ResearchBrief(BaseModel):
    """Output of the scope phase — the plan the researchers execute against."""

    topic: str = Field(..., description="Restated topic, refined for clarity.")
    summary: str = Field(
        ...,
        description="2–3 sentences clarifying what the user is asking and why.",
    )
    subtopics: list[Subtopic] = Field(
        ...,
        min_length=1,
        max_length=5,
        description=(
            "Independent research tracks. Each will be researched in parallel "
            "by a dedicated sub-agent."
        ),
    )


class Citation(BaseModel):
    title: str
    url: str
    relevance: str = Field(..., description="One sentence on why this source matters.")


class ResearchFindings(BaseModel):
    """Output of one parallel research branch."""

    subtopic: str
    summary: str = Field(
        ..., description="3–6 sentences summarizing what was learned."
    )
    citations: list[Citation] = Field(default_factory=list)


# ─── System prompts ──────────────────────────────────────────────────────────


SCOPE_PROMPT = """You scope research questions for a deep-research pipeline.

Given a user's topic, produce a structured research brief:

  1. Restate the topic precisely.
  2. Write a 2–3 sentence summary clarifying what the user wants to know
     and why it matters.
  3. Decompose the topic into 2–5 *independent* subtopics. Each subtopic
     becomes its own parallel research branch — so they MUST be answerable
     without depending on each other's results. If a subtopic naturally
     splits, list both halves.

Return a ResearchBrief.

Heuristics:
  * Prefer FEWER, BETTER subtopics over many shallow ones. 3 is the
    sweet spot for most queries; only go to 5 when the topic is truly
    multi-faceted.
  * Each subtopic title should be a focused question or angle, not a
    keyword.
  * Avoid overlap — overlap means duplicated work and a worse final
    report.
"""


def researcher_prompt(subtopic: str, rationale: str) -> str:
    """Per-branch system prompt — bakes the subtopic into the system message."""
    return f"""You are one of several parallel research agents on a deep-research team.

Your assigned subtopic:
  {subtopic}

Why it matters:
  {rationale}

Your job: gather authoritative sources via the ``web_search`` tool. For any
search hit whose snippet is too thin to support a citation, call
``web_fetch`` on its URL to read the full page. Iterate 1–3 search rounds.

Then return a ResearchFindings with:
  * ``subtopic``  — repeat your assigned subtopic verbatim.
  * ``summary``   — 3–6 sentences on what you learned. Synthesize, don't
    copy. Every factual claim must be defensible from your citations.
  * ``citations`` — a list of (title, url, relevance) for each source you
    actually relied on. Drop sources that didn't pan out.

Rules:
  * Stay in your lane — do NOT research outside your assigned subtopic.
    Other agents are covering the rest.
  * Do NOT invent URLs. If the search returned nothing useful, say so in
    the summary and return an empty citations list.
  * Do NOT draft prose for the final report — that's the writer's job.
"""


WRITER_PROMPT = """You are the final writer for a deep-research pipeline.

You receive (a) the research brief, and (b) a list of ResearchFindings —
one per subtopic — that the parallel researchers gathered. Your job is to
compose ONE coherent, well-structured Markdown report.

Format strictly:

  # <Topic>

  ## Summary
  <2–4 sentences. The reader should be able to stop here and have the
  answer.>

  ## Findings
  ### <Subtopic 1>
  <Paragraph(s) with inline citations like [1], [2].>

  ### <Subtopic 2>
  ...

  ## Sources
  [1] <Title> — <URL>
  [2] <Title> — <URL>

Rules:
  * Every factual claim in the body MUST have a numbered citation that
    points to an entry in the Sources list.
  * The Sources list is the union of citations from ALL ResearchFindings,
    de-duplicated by URL, numbered in order of first appearance.
  * Do NOT add facts that aren't in the findings. If the findings are
    thin, say so — accuracy beats coverage.
  * Do NOT include findings sections that have no useful content. Drop
    empty subtopics rather than padding.
  * Keep the report under 1000 words unless the brief is unusually broad.
"""


# ─── Agent factories ─────────────────────────────────────────────────────────


def _scope_llm() -> fa.LLMClient:
    return fa.LLMClient(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL_SCOPE", "gpt-4o"),
    )


def _researcher_llm() -> fa.LLMClient:
    return fa.LLMClient(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL_RESEARCHER", "gpt-4o-mini"),
    )


def _writer_llm() -> fa.LLMClient:
    return fa.LLMClient(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL_WRITER", "gpt-4o"),
    )


def build_scope_agent() -> fa.Agent:
    return fa.Agent(
        name="scope",
        system_prompt=SCOPE_PROMPT,
        llm=_scope_llm(),
        output_type=ResearchBrief,
    )


def build_researcher(subtopic: str, rationale: str) -> fa.Agent:
    tool_budget = int(os.getenv("RESEARCH_TOOL_BUDGET", "6"))
    return fa.Agent(
        name=f"researcher:{subtopic[:40]}",
        system_prompt=researcher_prompt(subtopic, rationale),
        llm=_researcher_llm(),
        tools=[web_search, web_fetch],
        # Cap each branch at 6 tool calls (3 searches + 3 fetches typical).
        # Protects against a misbehaving model spamming the search backend.
        middleware=[
            ToolBudget(
                max_calls=tool_budget,
                message="Research budget exhausted for this subtopic.",
            )
        ],
        # Give the loop room for one LLM turn per tool call plus the final
        # structured-output turn, so ``ToolBudget`` is the *graceful* limiter
        # rather than a hard ``MaxIterationsError`` when a model spreads its
        # calls one-per-turn (the default max_iterations of 10 could bind first
        # and abort the branch before it emits ``ResearchFindings``).
        config=fa.AgentConfig(max_iterations=tool_budget + 4),
        output_type=ResearchFindings,
    )


def build_writer_agent() -> fa.Agent:
    return fa.Agent(
        name="writer",
        system_prompt=WRITER_PROMPT,
        llm=_writer_llm(),
    )
