"""
Topology — Supervisor + 3 worker Agents.

Roles:

  * **researcher** — calls ``web_search`` (mocked by default) to gather
    sources for the topic. Produces a set of findings + URLs.
  * **writer**    — composes a structured Markdown report with inline
    citations like ``[1]`` linked to the source URLs the researcher found.
  * **verifier**  — re-reads the writer's draft against the researcher's
    findings; if any claim is uncited or unsupported, returns
    ``REVISIONS_REQUESTED`` plus the specific issues. Otherwise returns
    ``APPROVED``.

The Supervisor orchestrates the loop. Its system prompt explicitly tells
it to re-delegate to the writer on a ``REVISIONS_REQUESTED`` response.
``max_delegation_rounds=6`` gives room for: research + write + verify +
(write + verify) × 2 revisions.
"""

from __future__ import annotations

import os

import fastaiagent as fa
from fastaiagent.agent.middleware import ToolBudget

from tools import web_search

# ─── Worker system prompts ───────────────────────────────────────────────────

RESEARCHER_PROMPT = """You are the research worker on a 3-person team.

Your only job: call the `web_search` tool to gather authoritative sources
on the topic the supervisor hands you. Search 1–3 times with focused
queries, then return a Markdown bulleted list of findings:

  - **<source title>** — <url>
    <one-sentence summary of what this source contributes>

Do NOT draft prose, do NOT speculate beyond what the search results
support, and do NOT cite anything you did not actually retrieve.
"""


WRITER_PROMPT = """You are the writer worker on a 3-person team.

You receive (a) a research topic, and (b) a Markdown bulleted list of
sources from the researcher. Compose a concise, well-structured report.

Format strictly as Markdown:

  # <Topic>

  ## Summary
  <2–4 sentences>

  ## Findings
  ### <Subheading>
  <Paragraph with inline citations like [1], [2].>

  ### <Subheading>
  ...

  ## Sources
  [1] <Title> — <URL>
  [2] <Title> — <URL>

Rules:
  * Every factual claim in the body MUST have a numbered citation that
    points to an entry in the Sources list.
  * The Sources list MUST contain only URLs that appear in the
    researcher's findings — do not invent sources.
  * If you receive verifier feedback in the prompt, address it
    point-by-point in your revision.
"""


VERIFIER_PROMPT = """You are the verifier worker on a 3-person team.

You receive (a) the researcher's findings, and (b) the writer's draft
report. Audit the draft against the findings.

Return EXACTLY ONE of:

  * `APPROVED` — every factual claim has a citation, every cited URL
    appears in the researcher's findings, and the report stays on topic.

  * `REVISIONS_REQUESTED:` followed by a numbered list of specific issues:
      1. <Issue>: which claim is uncited / which URL doesn't exist in
         findings / where the report drifts off-topic.
      2. ...
    End with one line: "Send back to writer."

Be strict but constructive. The supervisor will re-delegate to the writer
with your feedback so they can fix it.
"""


# ─── Supervisor system prompt ────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You orchestrate a 3-worker research team.

Workflow:
  1. Delegate to `researcher` to gather sources for the user's topic.
  2. Delegate to `writer` with the topic AND the researcher's findings,
     asking for a draft report.
  3. Delegate to `verifier` with BOTH the researcher's findings AND the
     writer's draft.
  4. If the verifier returns `REVISIONS_REQUESTED:`, delegate back to
     `writer` with the verifier's specific feedback for revision. Then
     re-delegate to `verifier`. Repeat up to 2 revision rounds.
  5. Once the verifier returns `APPROVED`, return the writer's final
     report verbatim as your answer. Do NOT add commentary.

If after 2 revisions the verifier still flags issues, return the latest
draft prefixed with: "[verifier still flagged issues — best-effort draft]".
"""


def build_supervisor() -> fa.Supervisor:
    llm = fa.LLMClient(provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o"))

    researcher = fa.Agent(
        name="researcher",
        system_prompt=RESEARCHER_PROMPT,
        llm=llm,
        tools=[web_search],
        # Cap the researcher at 3 search calls per delegation — protects
        # against a misbehaving model spamming the search backend.
        middleware=[ToolBudget(max_calls=3, message="Research budget exhausted.")],
    )

    writer = fa.Agent(
        name="writer",
        system_prompt=WRITER_PROMPT,
        llm=llm,
    )

    verifier = fa.Agent(
        name="verifier",
        system_prompt=VERIFIER_PROMPT,
        llm=llm,
    )

    return fa.Supervisor(
        name="research-team",
        llm=llm,
        workers=[
            fa.Worker(
                agent=researcher,
                role="researcher",
                description="Gathers sources for a topic via web_search.",
            ),
            fa.Worker(
                agent=writer,
                role="writer",
                description="Drafts a Markdown report with inline citations.",
            ),
            fa.Worker(
                agent=verifier,
                role="verifier",
                description=(
                    "Audits the draft for citation coverage and on-topic-ness; "
                    "returns APPROVED or REVISIONS_REQUESTED with specific issues."
                ),
            ),
        ],
        system_prompt=SUPERVISOR_PROMPT,
        # researcher + writer + verifier + (writer + verifier) × 2 = 7. We
        # leave a little headroom; the supervisor's astream multiplies this
        # by 2 internally to allow per-round LLM iteration.
        max_delegation_rounds=8,
    )
