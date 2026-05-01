"""Test #14 — sending an image to a non-vision model raises a clear SDK error
*before* any HTTP is issued.

This exercises the real ``Message.to_provider_dict`` path used by every
``LLMClient._call_<provider>``. No HTTP, no mocks — the check fires inside
``format_multimodal_message`` and the network is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent import Image, LLMClient
from fastaiagent._internal.errors import NonVisionModelError
from fastaiagent.llm.message import Message, MessageRole

FIXTURES = Path(__file__).parent / "fixtures" / "multimodal"


def test_non_vision_openai_model_raises_via_message_format() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    msg = Message(role=MessageRole.user, content=["describe", img])
    with pytest.raises(NonVisionModelError) as exc_info:
        msg.to_provider_dict("openai", model="gpt-3.5-turbo", is_vision_capable=False)
    assert "gpt-3.5-turbo" in str(exc_info.value)


def test_non_vision_anthropic_model_raises_via_message_format() -> None:
    img = Image.from_file(FIXTURES / "cat.jpg")
    msg = Message(role=MessageRole.user, content=["describe", img])
    with pytest.raises(NonVisionModelError):
        msg.to_provider_dict("anthropic", model="claude-2.1", is_vision_capable=False)


def test_text_only_message_does_not_raise_on_non_vision_model() -> None:
    msg = Message(role=MessageRole.user, content="just text")
    out = msg.to_provider_dict("openai", model="gpt-3.5-turbo", is_vision_capable=False)
    assert out["content"] == "just text"


def test_llmclient_raises_via_to_provider_dict_helper_kwargs() -> None:
    """LLMClient must surface the same error before HTTP when its config says non-vision."""
    img = Image.from_file(FIXTURES / "cat.jpg")
    client = LLMClient(provider="openai", model="gpt-3.5-turbo")
    msg = Message(role=MessageRole.user, content=["x", img])
    with pytest.raises(NonVisionModelError):
        msg.to_provider_dict("openai", **client._provider_dict_kwargs())
