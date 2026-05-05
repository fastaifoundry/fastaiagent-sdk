"""
Pydantic schema for the structured ``MeetingNotes`` output.

The chain's three analyzer agents produce free-form JSON; the merge tool
validates and normalises into these models. Anything downstream
(``draft_followup_email``, the eval suite, the README rendering) reads
the typed model rather than the raw LLM JSON — so a malformed LLM
response surfaces at the merge step as a Pydantic validation error
instead of cascading into the followup tools.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """A concrete action assigned to a specific person with a deadline."""

    text: str = Field(..., description="What needs to be done, in one sentence.")
    owner: str = Field(..., description="Single attendee name; not 'team' or 'we'.")
    due: str | None = Field(
        None,
        description=(
            "Deadline as a human-readable date string (the LLM is told to use "
            "the format used in the transcript). None if not specified."
        ),
    )


class Decision(BaseModel):
    """A decision the meeting reached, with optional rationale."""

    text: str = Field(..., description="One-sentence statement of what was decided.")
    rationale: str | None = Field(
        None,
        description="One sentence on *why*, if the transcript stated it. None otherwise.",
    )


class MeetingNotes(BaseModel):
    """Structured meeting notes — the merged output of the analysis chain."""

    title: str = ""
    date: str | None = None
    attendees: list[str] = Field(default_factory=list)
    summary: str = Field(..., description="2–4 sentence executive summary.")
    action_items: list[ActionItem] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)

    def for_attendee(self, name: str) -> dict[str, Any]:
        """Slice the notes down to the parts relevant to one attendee.

        Used by ``draft_followup_email`` so each person gets a personalised
        email containing only their action items + the decisions that
        affected them.
        """
        their_actions = [a for a in self.action_items if a.owner.lower() == name.lower()]
        return {
            "name": name,
            "action_items": [a.model_dump() for a in their_actions],
            "decisions": [d.model_dump() for d in self.decisions],
            "summary": self.summary,
        }
