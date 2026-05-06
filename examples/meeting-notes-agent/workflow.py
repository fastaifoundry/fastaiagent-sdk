"""
Workflow — the Chain DAG that produces structured meeting notes.

```
input: {"path": "fixtures/sample_transcript.md"}
                │
                ▼
        ┌───────────────┐
        │   load        │ tool — reads .md / .txt / .pdf, extracts
        │               │        title + date heuristically
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   analyze     │ tool — fans out to 3 LLM agents IN PARALLEL
        │               │        (asyncio.gather inside one tool):
        │               │          • summarizer
        │               │          • action_extractor
        │               │          • decision_extractor
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │   merge       │ tool — Pydantic-validates each analyzer's JSON
        │               │        and produces a typed MeetingNotes model
        └───────┬───────┘
                ▼
            (end of chain — agent.py optionally fans out to per-attendee
             followup emails via draft_followup_email called outside the chain)
```

Why parallelism lives inside ``analyze`` rather than as ``NodeType.parallel``:
``NodeType.parallel`` passes ``context["input"]`` (the entire chain initial
state, stringified if it's a dict) to every child agent. For our use-case we
want each analyzer to see the raw transcript text — which is cleaner
expressed as ``asyncio.gather`` over three ``agent.arun(transcript)`` calls
inside one tool node. The chain DAG stays linear; the parallelism is
honest and visible in the ``analyze`` tool's source.

If you do want true graph-level parallel branches, ``NodeType.parallel``
sees the same ``RunContext`` thanks to the v1.6.1 fix and works for cases
where each child agent runs against the full initial-state dict.
"""

from __future__ import annotations

import os

import fastaiagent as fa
from fastaiagent.chain.node import NodeType

from tools import (
    MeetingDeps,
    analyze_meeting,
    load_transcript,
    merge_into_notes,
)


# ─── System prompts for the three analyzer agents ──────────────────────────


SUMMARIZER_PROMPT = """You are the summarizer in a meeting-notes pipeline.

You receive the full transcript of a meeting in your user message.

Return ONLY valid JSON of this shape (no Markdown fences, no commentary):

  {"summary": "<2-4 sentences>", "attendees": ["Name", "Name", ...]}

Rules:
  * The summary must capture the meeting's most important outcomes —
    decisions made and major topics. Skip pleasantries and side chatter.
  * ``attendees`` should reflect everyone who actually spoke or was
    addressed by name. Don't invent attendees the transcript didn't mention.
"""


ACTION_PROMPT = """You are the action-item extractor in a meeting-notes pipeline.

You receive the full transcript of a meeting in your user message.

Return ONLY valid JSON of this shape (no Markdown fences, no commentary):

  {"action_items": [
      {"text": "...", "owner": "Single Person", "due": "YYYY-MM-DD or null"},
      ...
  ]}

Rules:
  * Each action_item must be concrete and assigned to a SINGLE NAMED
    PERSON — never "the team" or "we". If the transcript implicitly
    assigns to a team, attribute to whoever ran the meeting.
  * The ``due`` field must echo the date format used in the transcript
    (e.g., "October 1", "Monday April 28") if specified, else null.
  * Do not duplicate action items that paraphrase each other.
  * Do not include past commitments — only forward-looking actions.
"""


DECISION_PROMPT = """You are the decision extractor in a meeting-notes pipeline.

You receive the full transcript of a meeting in your user message.

Return ONLY valid JSON of this shape (no Markdown fences, no commentary):

  {"decisions": [
      {"text": "<one sentence stating what was decided>",
       "rationale": "<one sentence on WHY, or null>"},
      ...
  ]}

Rules:
  * A "decision" is a forward-looking commitment the meeting reached —
    "we will / will not", a priority ranking, a choice between options.
  * Don't conflate decisions with action items: an action is who-does-what,
    a decision is what-the-team-chose. Same fact may produce one of each.
  * If the transcript stated the rationale for the choice, capture it; else
    rationale is null.
"""


FOLLOWUP_PROMPT = """You are the followup-email drafter for a meeting-notes pipeline.

You receive a JSON blob containing one attendee's name and the slice of
the meeting notes relevant to them (their action items + decisions that
affect them + the meeting summary).

Compose a short personalized email (60-100 words). Return ONLY valid
JSON of this shape:

  {"to": "<address>", "subject": "...", "body": "..."}

Rules:
  * Open with one sentence framing what the meeting decided.
  * List the attendee's action items as a clear bulleted list with their
    due dates inline.
  * No emoji, no marketing voice, no "circling back" filler.
  * If the attendee has zero action items, refuse politely — return body
    "No action items required." and subject "FYI: <meeting title>".
"""


# ─── Build the Chain ─────────────────────────────────────────────────────────


def build_chain() -> fa.Chain:
    """Construct the meeting-notes pipeline. Re-runnable; safe in tests."""
    chain = fa.Chain("meeting-notes", checkpoint_enabled=True)

    chain.add_node(
        "load",
        type=NodeType.tool,
        tool=load_transcript,
        input_mapping={"path": "{{state.path}}"},
    )

    chain.add_node(
        "analyze",
        type=NodeType.tool,
        tool=analyze_meeting,
        # The previous tool's return is the dict in state.output. We
        # pull just the transcript text via dotted-template traversal so
        # the analyzer agents receive the raw text, not a stringified
        # envelope.
        input_mapping={"transcript": "{{state.output.text}}"},
    )

    chain.add_node(
        "merge",
        type=NodeType.tool,
        tool=merge_into_notes,
        input_mapping={
            # state.output now holds analyze_meeting's payload; node_results
            # gives us the typed envelope from load that's no longer in
            # state.output (overwritten by analyze).
            "transcript_meta": "{{node_results.load.output}}",
            "analysis": "{{node_results.analyze.output}}",
        },
    )

    chain.connect("load", "analyze")
    chain.connect("analyze", "merge")
    return chain
