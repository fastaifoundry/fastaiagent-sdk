"""Tests for ``fastaiagent.multimodal.registry``."""

from __future__ import annotations

from fastaiagent.multimodal.registry import is_vision_capable, supports_native_pdf


def test_openai_gpt_4o_is_vision() -> None:
    assert is_vision_capable("openai", "gpt-4o") is True
    assert is_vision_capable("openai", "gpt-4o-2024-11-20") is True
    assert is_vision_capable("openai", "gpt-4o-mini") is True


def test_openai_gpt_3_5_is_not_vision() -> None:
    assert is_vision_capable("openai", "gpt-3.5-turbo") is False


def test_openai_gpt_4_base_explicitly_excluded() -> None:
    # The exact string "gpt-4" (no -turbo, no -o) is the legacy non-vision model.
    assert is_vision_capable("openai", "gpt-4") is False
    assert is_vision_capable("openai", "gpt-4-32k") is False


def test_anthropic_claude_3_5_is_vision() -> None:
    assert is_vision_capable("anthropic", "claude-3-5-sonnet-20241022") is True
    assert is_vision_capable("anthropic", "claude-sonnet-4-6") is True
    assert is_vision_capable("anthropic", "claude-opus-4-7") is True


def test_anthropic_claude_2_is_not_vision() -> None:
    assert is_vision_capable("anthropic", "claude-2.1") is False


def test_ollama_llava_is_vision_other_is_not() -> None:
    assert is_vision_capable("ollama", "llava:13b") is True
    assert is_vision_capable("ollama", "llama3.2-vision:11b") is True
    assert is_vision_capable("ollama", "llama3.2:8b") is False


def test_custom_provider_always_vision() -> None:
    assert is_vision_capable("custom", "anything-goes") is True


def test_unknown_provider_is_not_vision() -> None:
    assert is_vision_capable("madeup", "model") is False


def test_native_pdf_anthropic_only() -> None:
    assert supports_native_pdf("anthropic", "claude-sonnet-4-6") is True
    assert supports_native_pdf("anthropic", "claude-3-5-sonnet-20241022") is True
    assert supports_native_pdf("anthropic", "claude-2.1") is False
    assert supports_native_pdf("openai", "gpt-4o") is False
    assert supports_native_pdf("ollama", "llava:13b") is False
