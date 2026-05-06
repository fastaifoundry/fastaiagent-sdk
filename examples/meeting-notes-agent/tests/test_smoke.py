"""Smoke tests — no live LLM calls.

Exercise the deterministic parts so a developer iterating on prompts /
schema / scoring rubric gets fast feedback before spending tokens.

Run from the example directory:

    python -m pytest tests/

Coverage:
  * imports of every example module
  * Chain topology shape
  * load_transcript on the sample fixture (no LLM)
  * MeetingNotes Pydantic round-trip
  * for_attendee slicing logic
  * Custom scorer math against a hand-built MeetingNotes instance
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import fastaiagent as fa

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─── Imports ─────────────────────────────────────────────────────────────────


def test_imports() -> None:
    import agent  # noqa: F401
    import eval_suite  # noqa: F401
    import replay_demo  # noqa: F401
    import schema  # noqa: F401
    import streaming_demo  # noqa: F401
    import tools  # noqa: F401
    import workflow  # noqa: F401


# ─── Chain topology ─────────────────────────────────────────────────────────


def test_chain_topology() -> None:
    from workflow import build_chain

    chain = build_chain()
    assert chain.name == "meeting-notes"
    ids = [n.id for n in chain.nodes]
    assert ids == ["load", "analyze", "merge"], "Order is load → analyze → merge"

    edges = {(e.source, e.target) for e in chain.edges}
    assert ("load", "analyze") in edges
    assert ("analyze", "merge") in edges


# ─── Transcript loader (no LLM) ─────────────────────────────────────────────


def test_load_transcript_parses_title_and_date() -> None:
    from tools import MeetingDeps, load_transcript

    ctx = fa.RunContext(state=MeetingDeps())
    fixture = _HERE / "fixtures" / "sample_transcript.md"
    result = asyncio.run(load_transcript.aexecute({"path": str(fixture)}, context=ctx))
    assert result.error is None
    payload = result.output
    assert isinstance(payload["text"], str) and len(payload["text"]) > 200
    # The fixture's first heading: "# Q3 Roadmap Sync — 2026-04-22"
    assert payload["title"] == "Q3 Roadmap Sync"
    assert payload["date"] == "2026-04-22"


def test_load_transcript_missing_file_returns_error() -> None:
    from tools import MeetingDeps, load_transcript

    ctx = fa.RunContext(state=MeetingDeps())
    result = asyncio.run(load_transcript.aexecute({"path": "/nope/missing.md"}, context=ctx))
    assert result.error is None
    assert "File not found" in result.output["error"]


# ─── Schema round-trip ──────────────────────────────────────────────────────


def test_meeting_notes_validates_minimal_payload() -> None:
    from schema import ActionItem, Decision, MeetingNotes

    notes = MeetingNotes.model_validate(
        {
            "title": "Test",
            "date": "2026-04-22",
            "attendees": ["A", "B"],
            "summary": "Two-line summary.",
            "action_items": [
                {"text": "Do X", "owner": "A", "due": "Friday"},
                {"text": "Do Y", "owner": "B", "due": None},
            ],
            "decisions": [
                {"text": "We will Z", "rationale": "Because reason."},
            ],
        }
    )
    assert len(notes.action_items) == 2
    assert isinstance(notes.action_items[0], ActionItem)
    assert isinstance(notes.decisions[0], Decision)


def test_for_attendee_slices_correctly() -> None:
    from schema import MeetingNotes

    notes = MeetingNotes(
        title="t",
        attendees=["Alice", "Bob"],
        summary="s",
        action_items=[
            {"text": "X", "owner": "Alice"},
            {"text": "Y", "owner": "Bob"},
            {"text": "Z", "owner": "Alice"},
        ],
        decisions=[{"text": "D"}],
    )
    alice = notes.for_attendee("alice")  # case-insensitive match
    assert alice["name"] == "alice"
    assert len(alice["action_items"]) == 2
    assert alice["decisions"] == [{"text": "D", "rationale": None}]


# ─── Custom scorers (no LLM) ────────────────────────────────────────────────


def test_action_item_recall_scorer() -> None:
    from eval_suite import _score_action_recall
    from schema import MeetingNotes

    notes = MeetingNotes(
        summary="s",
        action_items=[
            {"text": "Draft RFC for auth-middleware", "owner": "Bob"},
            {"text": "Order trace inspector mockups", "owner": "Carol"},
        ],
    )
    score, _ = _score_action_recall(notes, ["auth-middleware", "trace inspector", "SDETs"])
    assert abs(score - (2 / 3)) < 1e-6


def test_owner_attribution_scorer_penalises_team() -> None:
    from eval_suite import _score_owner_attribution
    from schema import MeetingNotes

    notes = MeetingNotes(
        summary="s",
        action_items=[
            {"text": "X", "owner": "the team"},   # forbidden
            {"text": "Y", "owner": "Alice"},
        ],
    )
    score, reason = _score_owner_attribution(notes, expected_owners={"Alice", "Bob"})
    # well_attributed = 1/2; coverage of expected owners = 1/2 → 0.25
    assert score == 0.25
    assert "1/2 actions named-owner" in reason


def test_owner_attribution_scorer_zero_for_empty() -> None:
    from eval_suite import _score_owner_attribution
    from schema import MeetingNotes

    notes = MeetingNotes(summary="s", action_items=[])
    score, _ = _score_owner_attribution(notes, expected_owners={"Alice"})
    assert score == 0.0
