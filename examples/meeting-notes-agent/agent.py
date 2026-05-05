"""
Meeting Notes Agent — FastAIAgent SDK Template (Chain with parallel analysis)

A Chain-orchestrated meeting-notes generator. Reads a transcript (text or
PDF), runs three single-purpose analyzer agents *in parallel*, validates
the merged output against a Pydantic ``MeetingNotes`` schema, and
optionally drafts personalized followup emails per attendee.

Usage:
    python agent.py                                   # default sample transcript
    python agent.py --transcript fixtures/sample_transcript.md
    python agent.py --transcript meeting.pdf
    python agent.py --transcript ... --notify         # also send followup emails

Companion files:
    workflow.py        — Chain DAG + analyzer system prompts
    tools.py           — load_transcript, analyze_meeting, merge_into_notes,
                         draft_followup_email
    schema.py          — Pydantic MeetingNotes / ActionItem / Decision
    eval_suite.py      — golden meeting + custom completeness scorers
    streaming_demo.py  — tail the trace store as the chain runs
    replay_demo.py     — fork the chain at any node, rerun
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

import fastaiagent as fa

from schema import MeetingNotes
from tools import draft_followup_email, make_deps
from workflow import build_chain


def _pretty_print(notes: MeetingNotes) -> None:
    print("\n" + "═" * 64)
    print(f"  {notes.title}" + (f" — {notes.date}" if notes.date else ""))
    print("═" * 64)

    print("\nAttendees:")
    for name in notes.attendees:
        print(f"  • {name}")

    print("\nSummary:")
    print(f"  {notes.summary}")

    print("\nAction items:")
    if not notes.action_items:
        print("  (none)")
    for a in notes.action_items:
        due = f" — due {a.due}" if a.due else ""
        print(f"  □ [{a.owner}] {a.text}{due}")

    print("\nDecisions:")
    if not notes.decisions:
        print("  (none)")
    for d in notes.decisions:
        rationale = f"\n      ↳ {d.rationale}" if d.rationale else ""
        print(f"  ★ {d.text}{rationale}")
    print()


async def run(transcript_path: str, notify: bool = False) -> None:
    chain = build_chain()
    deps = make_deps()
    ctx = fa.RunContext(state=deps)
    execution_id = f"meeting-{uuid.uuid4().hex[:8]}"

    print(f"\nTranscript: {transcript_path}")
    print(f"Execution: {execution_id}")
    print("─" * 64)

    result = await chain.aexecute(
        {"path": transcript_path},
        execution_id=execution_id,
        context=ctx,
    )

    if result.status != "completed":
        print(f"Chain ended with status={result.status!r} — see {result.pending_interrupt}")
        return

    # The merge tool's output is the final state.output payload.
    notes_dict = result.final_state.get("output") or {}
    if not isinstance(notes_dict, dict) or "summary" not in notes_dict:
        print("Could not parse a complete MeetingNotes from the chain output:")
        print(json.dumps(notes_dict, indent=2, default=str))
        return
    notes = MeetingNotes.model_validate(notes_dict)
    _pretty_print(notes)

    if notify and notes.attendees:
        print("─" * 64)
        print("Drafting followup emails…\n")
        for name in notes.attendees:
            receipt = await draft_followup_email.aexecute(
                {"notes_json": json.dumps(notes.model_dump()), "attendee_name": name},
                context=ctx,
            )
            payload = receipt.output if hasattr(receipt, "output") else receipt
            if isinstance(payload, dict) and payload.get("skipped"):
                print(f"  • {name:<20} skipped — {payload.get('reason')}")
            elif isinstance(payload, dict) and payload.get("sent"):
                print(f"  • {name:<20} sent to {payload.get('to')} ({payload.get('msg_id')})")
            else:
                print(f"  • {name:<20} unknown send status: {payload}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Meeting Notes Chain Agent")
    parser.add_argument(
        "--transcript",
        default="fixtures/sample_transcript.md",
        help="Path to a meeting transcript (.md / .txt / .pdf)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="After producing notes, draft per-attendee followup emails",
    )
    parser.add_argument("--connect", action="store_true", help="Connect to FastAIAgent Platform")
    args = parser.parse_args()

    if args.connect:
        api_key = os.getenv("FASTAIAGENT_API_KEY")
        project = os.getenv("FASTAIAGENT_PROJECT", "meeting-notes-agent")
        target = os.getenv("FASTAIAGENT_TARGET", "https://app.fastaiagent.net")
        if api_key:
            fa.connect(api_key=api_key, target=target, project=project)
            print(f"Connected to FastAIAgent Platform (project: {project})")

    asyncio.run(run(args.transcript, notify=args.notify))


if __name__ == "__main__":
    main()
