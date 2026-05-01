"""Phase 4 tests — Swarm + Supervisor pass-through of multimodal content.

Mock-free: no LLM calls. The structural assertions (snapshot round-trip,
shared-state survival across the Swarm boundary, normalized agent input
on Supervisor delegation) are the actual surface multimodal must work
across.

Spec test #9 — Swarm with multimodal handoff (the round-trip variant; the
real-LLM handoff path is exercised by the OpenAI/Anthropic e2e gates).
"""

from __future__ import annotations

from pathlib import Path

from fastaiagent import PDF, Image
from fastaiagent.agent.swarm import SwarmState, _restore_state, _swarm_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def test_swarm_snapshot_serializes_image_in_shared_state() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    state = SwarmState(
        shared={"photo": img, "claim_id": "C-101"},
        handoff_count=1,
        path=["triage", "photo_assessor"],
        last_reason="image present",
    )

    snap = _swarm_snapshot(
        iteration=2,
        current="photo_assessor",
        current_input="evaluate damage",
        original_input="file a claim",
        state=state,
        accumulated_tool_calls=[],
        total_tokens=42,
    )

    shared = snap["shared_context"]
    assert isinstance(shared["photo"], dict)
    assert shared["photo"]["type"] == "image"
    assert shared["claim_id"] == "C-101"
    assert snap["active_agent"] == "photo_assessor"
    assert snap["path"] == ["triage", "photo_assessor"]


def test_swarm_restore_rehydrates_image_from_dict_marker() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    state = SwarmState(shared={"photo": img})
    snap = _swarm_snapshot(
        iteration=0,
        current="triage",
        current_input="x",
        original_input="x",
        state=state,
        accumulated_tool_calls=[],
        total_tokens=0,
    )

    restored = _restore_state(snap)
    assert isinstance(restored.shared["photo"], Image)
    assert restored.shared["photo"].data == img.data
    assert restored.shared["photo"].media_type == "image/jpeg"


def test_swarm_restore_rehydrates_pdf_in_shared_state() -> None:
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    state = SwarmState(shared={"contract": pdf, "step": "policy_review"})
    snap = _swarm_snapshot(
        iteration=3,
        current="policy_agent",
        current_input="check coverage",
        original_input="file a claim",
        state=state,
        accumulated_tool_calls=[],
        total_tokens=10,
    )

    restored = _restore_state(snap)
    assert isinstance(restored.shared["contract"], PDF)
    assert restored.shared["contract"].data == pdf.data
    assert restored.shared["step"] == "policy_review"


def test_swarm_round_trip_preserves_path_and_handoff_count() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    state = SwarmState(
        shared={"photo": img},
        handoff_count=2,
        path=["a", "b", "c"],
        last_reason="needs photo expert",
    )

    snap = _swarm_snapshot(
        iteration=4,
        current="c",
        current_input="x",
        original_input="x",
        state=state,
        accumulated_tool_calls=[],
        total_tokens=0,
    )
    restored = _restore_state(snap)

    assert restored.handoff_count == 2
    assert restored.path == ["a", "b", "c"]
    assert restored.last_reason == "needs photo expert"


def test_swarm_snapshot_is_json_serializable() -> None:
    """The whole point of swapping ``dict()`` for ``_serialize_for_checkpoint`` —
    the snapshot must be valid JSON so the SQLite checkpointer accepts it."""
    import json

    img = Image.from_file(FIXTURES / "cat.jpg")
    pdf = PDF.from_file(FIXTURES / "contract.pdf")
    state = SwarmState(shared={"photo": img, "doc": pdf, "n": 3})
    snap = _swarm_snapshot(
        iteration=0,
        current="x",
        current_input="x",
        original_input="x",
        state=state,
        accumulated_tool_calls=[],
        total_tokens=0,
    )

    # Must round-trip through json without raising.
    encoded = json.dumps(snap)
    decoded = json.loads(encoded)
    restored = _restore_state(decoded)
    assert isinstance(restored.shared["photo"], Image)
    assert restored.shared["photo"].data == img.data
    assert isinstance(restored.shared["doc"], PDF)
    assert restored.shared["doc"].data == pdf.data
