"""
Tools — what the personal assistant can call.

Lightweight by design: notes-style scratchpad + introspection. The
real work happens in ``ComposableMemory`` (see ``memory_setup.py``);
the tools here mostly let the user interact with that memory directly
when they want to.

  * ``add_note``   — write a durable note. Stored on disk under
                     ``.fastaiagent/notes.jsonl``; the agent's
                     ``VectorBlock`` will also pick it up via
                     ``on_message`` because the call appears in the
                     conversation.
  * ``search_notes`` — explicit semantic search over the notes log
                     (independent of the VectorBlock — useful when the
                     user wants a ranked list rather than a contextual
                     recall).
  * ``list_facts`` — show the facts ``FactExtractionBlock`` has
                     captured. Demystifies what the assistant "knows".
  * ``today``      — current ISO date. Cheap; the StaticBlock pins
                     today's date at startup, but a multi-day session
                     would let it drift without an explicit tool.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import fastaiagent as fa
from fastaiagent.agent.memory import ComposableMemory
from fastaiagent.agent.memory_blocks import FactExtractionBlock

_HERE = Path(__file__).resolve().parent
_NOTES_PATH = _HERE / ".fastaiagent" / "notes.jsonl"


@dataclass
class AssistantDeps:
    """Shared dependencies. Most importantly: the live ``ComposableMemory``
    instance, so tools can introspect it (list_facts, etc.) without
    needing to plumb through the agent's internals."""

    memory: ComposableMemory | None = None
    notes: list[dict] = field(default_factory=list)


def make_deps(memory: ComposableMemory | None = None) -> AssistantDeps:
    deps = AssistantDeps(memory=memory)
    if _NOTES_PATH.exists():
        with _NOTES_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        deps.notes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return deps


# ─── Tools ───────────────────────────────────────────────────────────────────


@fa.tool()
def add_note(text: str, ctx: fa.RunContext[AssistantDeps]) -> str:
    """Save a durable note. Use this whenever the user asks you to
    remember something specific — a meeting time, a name, a preference,
    a TODO. The note is appended to ``.fastaiagent/notes.jsonl`` and
    available to ``search_notes``."""
    note = {"text": text.strip(), "ts": time.time()}
    _NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _NOTES_PATH.open("a") as f:
        f.write(json.dumps(note) + "\n")
    ctx.state.notes.append(note)
    return f"Saved note ({len(ctx.state.notes)} total)."


@fa.tool()
def search_notes(query: str, ctx: fa.RunContext[AssistantDeps]) -> str:
    """Search the user's saved notes for relevant entries. Returns the
    top-3 substring matches ranked by recency among matches.

    For semantic recall over the *entire* conversation history, the
    agent's ``VectorBlock`` is already doing that automatically — call
    this tool when the user explicitly asked you to look at saved notes.
    """
    if not ctx.state.notes:
        return "(no notes saved yet)"
    needle = query.lower().strip()
    if not needle:
        # No query — return the most recent 3.
        recent = sorted(ctx.state.notes, key=lambda n: n["ts"], reverse=True)[:3]
    else:
        matches = [n for n in ctx.state.notes if needle in n["text"].lower()]
        recent = sorted(matches, key=lambda n: n["ts"], reverse=True)[:3]
    if not recent:
        return f"No notes matched {query!r}."
    return "\n".join(f"- {n['text']}" for n in recent)


@fa.tool()
def list_facts(ctx: fa.RunContext[AssistantDeps]) -> str:
    """Show the facts the assistant has automatically extracted from
    prior conversations. These come from ``FactExtractionBlock`` —
    every turn an LLM call extracts durable statements about the user
    and persists them to a deduplicated list."""
    memory = ctx.state.memory
    if memory is None:
        return "(memory introspection unavailable in this run)"
    fact_block: FactExtractionBlock | None = next(
        (b for b in memory.blocks if isinstance(b, FactExtractionBlock)), None
    )
    if fact_block is None or not fact_block._facts:
        return "(no facts extracted yet — chat for a few turns and try again)"
    return "\n".join(f"- {f}" for f in fact_block._facts)


@fa.tool()
def today(ctx: fa.RunContext[AssistantDeps]) -> str:
    """Return today's ISO date. Use this for any "when" questions —
    the StaticBlock pins today's date at startup but it goes stale
    after midnight without a fresh tool call."""
    from datetime import date

    return date.today().isoformat()
