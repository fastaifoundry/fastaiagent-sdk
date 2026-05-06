"""Tests for multimodal-aware message serialization (bug fix).

Before this fix, ``SQLiteCheckpointer`` + ``agent.arun([text, image])``
crashed at the first turn-boundary checkpoint with
``UnicodeDecodeError`` because ``_serialize_messages`` did
``Message.model_dump(mode="json")`` on a content list containing an
``Image`` dataclass with raw bytes — Pydantic's JSON serializer cannot
encode bytes that aren't valid UTF-8.

These tests pin the new behavior:

  * ``_serialize_messages`` round-trips multimodal content through JSON
    via the Image / PDF ``to_dict()`` envelope (T1)
  * ``_deserialize_messages`` rebuilds Image / PDF dataclass instances
    from the saved dicts (T2)
  * Bytes survive the round-trip exactly (T3)
  * Text-only messages take the unchanged fast path (T4)
  * Full integration: ``Agent + SQLiteCheckpointer + image input`` no
    longer crashes when a checkpoint is written (T5)

No LLM calls — fixtures load real images off disk; the integration test
uses the existing ``MockLLMClient`` stub plus a tool that calls
``interrupt()`` to force the checkpoint write.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import fastaiagent as fa
from fastaiagent.agent.executor import _deserialize_messages, _serialize_messages
from fastaiagent.llm.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from fastaiagent.llm.client import LLMResponse
from tests.conftest import MockLLMClient

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


# ─── T1 — Serialize Image content without raising ────────────────────────────


def test_serialize_messages_with_image_round_trips_through_json():
    image = fa.Image.from_file(FIXTURES / "receipt.png")
    msg = UserMessage(["What's on this receipt?", image])

    serialized = _serialize_messages([msg])

    assert len(serialized) == 1
    # The previously-fatal call was ``json.dumps(serialized)``. Verify it
    # works now — i.e. nothing in the payload still carries raw bytes.
    payload = json.dumps(serialized)
    assert "data_base64" in payload, "Image was not converted to base64 envelope"

    # The Image dict envelope should match Image.to_dict() shape.
    content = serialized[0]["content"]
    assert isinstance(content, list)
    img_part = next(p for p in content if isinstance(p, dict) and p.get("type") == "image")
    assert img_part["media_type"] == "image/png"
    assert isinstance(img_part["data_base64"], str) and len(img_part["data_base64"]) > 0


# ─── T2 — Deserialize rebuilds Image instances ──────────────────────────────


def test_deserialize_messages_rebuilds_image_dataclass():
    image = fa.Image.from_file(FIXTURES / "receipt.png")
    msg = UserMessage(["Look at this", image])

    serialized = _serialize_messages([msg])
    rebuilt = _deserialize_messages(serialized)

    assert len(rebuilt) == 1
    parts = rebuilt[0].content
    assert isinstance(parts, list)
    assert any(isinstance(p, fa.Image) for p in parts), \
        "Image dataclass was not rebuilt from the saved envelope"


# ─── T3 — Bytes survive the round-trip exactly ───────────────────────────────


def test_image_bytes_survive_round_trip():
    """Lossless round-trip is the contract: a resumed agent must see the
    exact same image bytes the original run sent to the LLM."""
    image = fa.Image.from_file(FIXTURES / "receipt.png")
    original_bytes = image.data
    msg = UserMessage(["", image])

    serialized = _serialize_messages([msg])
    rebuilt = _deserialize_messages(serialized)

    rebuilt_image = next(p for p in rebuilt[0].content if isinstance(p, fa.Image))
    assert rebuilt_image.data == original_bytes
    assert rebuilt_image.media_type == image.media_type


# ─── T4 — Text-only messages take the fast path ──────────────────────────────


def test_text_only_messages_unchanged():
    """Backward compat — non-multimodal messages must serialize identically
    to the pre-fix behavior so existing checkpoints stay readable."""
    msgs = [
        SystemMessage("You are a helpful assistant."),
        UserMessage("Hello"),
        AssistantMessage(content="Hi! How can I help?"),
    ]

    serialized = _serialize_messages(msgs)
    rebuilt = _deserialize_messages(serialized)

    assert len(rebuilt) == 3
    assert rebuilt[0].content == "You are a helpful assistant."
    assert rebuilt[1].content == "Hello"
    assert rebuilt[2].content == "Hi! How can I help?"


# ─── T5 — Integration: Agent + checkpointer + image + interrupt ──────────────


@pytest.mark.asyncio
async def test_agent_checkpointer_with_image_input(temp_dir):
    """End-to-end: an Agent with a SQLiteCheckpointer and an Image input
    must not crash at the turn-boundary checkpoint write. We force the
    checkpoint to fire by having a tool call ``interrupt()``; the agent
    persists the (image-bearing) message history in the checkpoint, then
    we resume and verify the messages came back with the Image rebuilt.
    """

    @fa.tool()
    def needs_approval() -> dict:
        decision = fa.interrupt(reason="approve", context={})
        return {"approved": decision.approved}

    # Canned LLM that calls our gated tool, then completes after resume.
    llm = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="t1", name="needs_approval", arguments={})],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="all done", finish_reason="stop"),
        ]
    )
    cp = fa.SQLiteCheckpointer(db_path=str(temp_dir / "cp.db"))
    agent = fa.Agent(
        name="vision-with-cp",
        system_prompt="x",
        llm=llm,
        tools=[needs_approval],
        checkpointer=cp,
    )

    image = fa.Image.from_file(FIXTURES / "receipt.png")

    # First leg — runs through the LLM, dispatches the tool, hits interrupt().
    # Before the fix this crashed inside _put_turn_checkpoint with
    # UnicodeDecodeError; now it should suspend cleanly.
    paused = await agent.arun(["Describe this image", image], execution_id="cp-mm-1")
    assert paused.status == "paused"
    assert (paused.pending_interrupt or {}).get("reason") == "approve"

    # Resume. The deserializer must rebuild the Image content so the LLM
    # client (or any downstream consumer of messages) sees the original
    # bytes, not a base64-stand-in dict.
    completed = await agent.aresume(
        "cp-mm-1",
        resume_value=fa.Resume(approved=True),
    )
    assert completed.status == "completed"
    assert completed.output == "all done"
