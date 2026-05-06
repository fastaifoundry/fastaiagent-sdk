"""Smoke tests — no live LLM calls.

Coverage:
  * imports of every example module
  * memory_setup wires all 4 block types in the expected order
  * StaticBlock renders identity text on every turn (no LLM)
  * VectorBlock indexes a message via on_message + recalls it via render
  * Tools work standalone (with strict ctx required, post-bug-fix)
  * Notes round-trip — add_note + search_notes
  * memory.save / memory.load round-trip across a fresh ComposableMemory
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
    import memory_setup  # noqa: F401
    import tools  # noqa: F401


# ─── Memory wiring ──────────────────────────────────────────────────────────


def test_build_memory_wires_all_four_blocks() -> None:
    from fastaiagent.agent.memory_blocks import (
        FactExtractionBlock,
        StaticBlock,
        SummaryBlock,
        VectorBlock,
    )

    from memory_setup import build_memory

    memory = build_memory()
    block_types = [type(b) for b in memory.blocks]
    # Order matters for rendering: static fragment first, then summary,
    # then semantic recall, then extracted facts.
    assert block_types == [StaticBlock, SummaryBlock, VectorBlock, FactExtractionBlock]


def test_static_block_renders_identity_on_every_turn(monkeypatch) -> None:
    monkeypatch.setenv("USER_NAME", "Test User")
    monkeypatch.setenv("USER_ROLE", "Test Role")
    monkeypatch.setenv("USER_TZ", "UTC")
    from memory_setup import build_memory

    memory = build_memory()
    rendered = memory.get_context()
    # First fragment is the StaticBlock's SystemMessage.
    assert any(
        "Test User" in str(m.content) and "Test Role" in str(m.content)
        for m in rendered
    ), f"StaticBlock identity not rendered: {[m.content for m in rendered]}"


def test_vector_block_indexes_then_recalls() -> None:
    """Add a message via the live ComposableMemory, then query VectorBlock
    semantically and assert it surfaces."""
    from fastaiagent.agent.memory_blocks import VectorBlock
    from fastaiagent.llm.message import UserMessage

    from memory_setup import build_memory

    memory = build_memory()
    memory.add(UserMessage("I deployed our service on Argo CD with custom hooks last quarter."))
    memory.add(UserMessage("Looking forward to the holidays."))

    vector = next(b for b in memory.blocks if isinstance(b, VectorBlock))
    rendered = vector.render("argo")
    assert rendered, "VectorBlock did not return any fragments for an in-corpus query"
    body = " ".join(str(m.content) for m in rendered).lower()
    assert "argo" in body


# ─── Tools ──────────────────────────────────────────────────────────────────


def test_add_note_persists_to_disk(tmp_path, monkeypatch) -> None:
    """add_note writes to ``.fastaiagent/notes.jsonl`` next to tools.py.
    We patch the path constant so the test doesn't pollute the repo."""
    import tools

    notes_path = tmp_path / "notes.jsonl"
    monkeypatch.setattr(tools, "_NOTES_PATH", notes_path)

    deps = tools.AssistantDeps()  # no live memory needed
    ctx = fa.RunContext(state=deps)
    result = asyncio.run(tools.add_note.aexecute({"text": "remember the milk"}, context=ctx))
    assert result.error is None
    assert "Saved note" in result.output

    assert notes_path.exists()
    line = notes_path.read_text().strip()
    assert "remember the milk" in line
    assert deps.notes[0]["text"] == "remember the milk"


def test_search_notes_substring_match() -> None:
    import tools

    deps = tools.AssistantDeps(notes=[
        {"text": "Alice's birthday is May 11", "ts": 1.0},
        {"text": "buy oat milk", "ts": 2.0},
        {"text": "Bob's deploy is Thursday", "ts": 3.0},
    ])
    ctx = fa.RunContext(state=deps)
    result = asyncio.run(tools.search_notes.aexecute({"query": "milk"}, context=ctx))
    assert result.error is None
    assert "oat milk" in result.output
    assert "birthday" not in result.output


# ─── Persistence round-trip ─────────────────────────────────────────────────


def test_memory_save_load_round_trip(tmp_path) -> None:
    from fastaiagent.agent.memory_blocks import FactExtractionBlock
    from fastaiagent.llm.message import UserMessage

    from memory_setup import build_memory, save_memory

    # Inject a fact directly so we don't need an LLM call.
    memory = build_memory()
    fact_block = next(b for b in memory.blocks if isinstance(b, FactExtractionBlock))
    fact_block._facts = ["The user's name is Riley.", "The user lives in Lisbon."]

    save_memory(memory, tmp_path / "mem")
    assert (tmp_path / "mem" / "primary.json").exists()

    # Fresh memory; load from the saved dir; facts should round-trip.
    fresh = build_memory(memory_dir=tmp_path / "mem")
    fresh_facts = next(b for b in fresh.blocks if isinstance(b, FactExtractionBlock))
    assert fresh_facts._facts == ["The user's name is Riley.", "The user lives in Lisbon."]


# ─── Eval session shape (no LLM) ────────────────────────────────────────────


def test_eval_suite_canonical_session_well_formed() -> None:
    from eval_suite import CANONICAL_TURNS, MUST_EXTRACT_FACTS

    # ≥ summarize_every (4) so SummaryBlock fires at least once.
    assert len(CANONICAL_TURNS) >= 5
    # Each must-extract fact actually appears in the session content —
    # otherwise our scorer is asking the LLM to invent facts.
    joined = " ".join(CANONICAL_TURNS).lower()
    for needle in MUST_EXTRACT_FACTS:
        assert needle.lower() in joined, f"{needle!r} not present in canonical turns"
