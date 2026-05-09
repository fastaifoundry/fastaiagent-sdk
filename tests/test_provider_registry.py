"""Unit tests for ``fastaiagent.llm.providers`` registry.

No network, no LLM. Validates the public surface of the preset registry
and the seed presets shipped in v1.8.0.
"""

from __future__ import annotations

import pytest

from fastaiagent.llm.providers import (
    ProviderPreset,
    get_preset,
    list_presets,
    list_provider_keys,
    register_provider,
    reserved_keys,
    unregister_provider,
)

SEED_PRESET_KEYS = {
    "gemini",
    "groq",
    "openrouter",
    "deepseek",
    "together",
    "fireworks",
    "perplexity",
    "mistral",
    "lmstudio",
    "vllm",
    "sambanova",
    "cerebras",
}


def test_seed_presets_present() -> None:
    keys = {p.key for p in list_presets()}
    missing = SEED_PRESET_KEYS - keys
    assert not missing, f"missing seed presets: {missing}"


def test_list_provider_keys_includes_builtins_and_presets() -> None:
    keys = set(list_provider_keys())
    assert {"openai", "anthropic", "ollama", "azure", "bedrock", "custom"} <= keys
    assert "gemini" in keys and "groq" in keys


def test_groq_preset_shape() -> None:
    p = get_preset("groq")
    assert p is not None
    assert p.base_url == "https://api.groq.com/openai/v1"
    assert p.env_var == "GROQ_API_KEY"
    assert p.wire == "openai_compat"
    assert p.cap("tools") is True
    assert p.cap("streaming") is True


def test_gemini_preset_uses_native_wire() -> None:
    p = get_preset("gemini")
    assert p is not None
    assert p.wire == "native_gemini"
    assert p.env_var == "GEMINI_API_KEY"


def test_register_and_unregister_round_trip() -> None:
    preset = ProviderPreset(
        key="my-internal-llm-test",
        base_url="https://internal.test/v1",
        env_var="MY_INTERNAL_KEY",
        default_model="house-7b",
        wire="openai_compat",
        capabilities={"tools": True, "streaming": True},
        description="Test preset",
    )
    register_provider(preset)
    try:
        assert get_preset("my-internal-llm-test") is preset
        assert "my-internal-llm-test" in list_provider_keys()
    finally:
        unregister_provider("my-internal-llm-test")
    assert get_preset("my-internal-llm-test") is None


def test_register_duplicate_raises_without_overwrite() -> None:
    preset = ProviderPreset(
        key="dup-test",
        base_url="https://x.test/v1",
        env_var="X",
        default_model="m",
        wire="openai_compat",
    )
    register_provider(preset)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_provider(preset)
        # Overwrite=True is allowed.
        replacement = ProviderPreset(
            key="dup-test",
            base_url="https://y.test/v1",
            env_var="Y",
            default_model="m2",
            wire="openai_compat",
        )
        register_provider(replacement, overwrite=True)
        again = get_preset("dup-test")
        assert again is not None
        assert again.base_url == "https://y.test/v1"
    finally:
        unregister_provider("dup-test")


def test_reserved_keys_cannot_be_registered() -> None:
    for reserved in reserved_keys():
        with pytest.raises(ValueError, match="reserved"):
            register_provider(
                ProviderPreset(
                    key=reserved,
                    base_url="x",
                    env_var="X",
                    default_model="m",
                    wire="openai_compat",
                )
            )


def test_unregister_unknown_is_noop() -> None:
    # Should not raise.
    unregister_provider("definitely-not-registered-xyz")


def test_capability_default() -> None:
    p = get_preset("perplexity")
    assert p is not None
    assert p.cap("tools") is False
    # Missing capability returns the default.
    assert p.cap("nonexistent_cap", "fallback") == "fallback"
