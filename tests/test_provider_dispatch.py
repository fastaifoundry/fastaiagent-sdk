"""LLMClient preset resolution + dispatch routing.

No network. Verifies that LLMClient honours preset-shipped defaults for
``base_url`` and ``api_key``, that the dispatch routes to the right
backend (``_call_openai`` for openai_compat presets, the native Gemini
wire for gemini), and that capability fallbacks rewrite the request body
where the preset declares missing native support.
"""

from __future__ import annotations

import os

import pytest

from fastaiagent._internal.errors import LLMError
from fastaiagent.llm.client import _BUILTIN_PROVIDERS, LLMClient
from fastaiagent.llm.providers import ProviderPreset, register_provider, unregister_provider


def test_builtin_providers_unchanged() -> None:
    """Built-in providers don't gain a preset and keep historical behaviour."""
    client = LLMClient(provider="openai", model="gpt-4o-mini", api_key="sk-test")
    assert client._preset is None
    assert client.base_url == "https://api.openai.com/v1"
    assert client.api_key == "sk-test"


def test_preset_provider_resolves_base_url() -> None:
    client = LLMClient(provider="groq", model="llama-3.1-70b-versatile", api_key="gsk-test")
    assert client._preset is not None
    assert client._preset.key == "groq"
    assert client.base_url == "https://api.groq.com/openai/v1"


def test_preset_provider_resolves_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fixture-key")
    client = LLMClient(provider="openrouter", model="openai/gpt-4o-mini")
    assert client.api_key == "or-fixture-key"


def test_preset_provider_explicit_api_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    client = LLMClient(provider="groq", api_key="explicit-arg")
    assert client.api_key == "explicit-arg"


def test_unknown_provider_raises_with_helpful_listing() -> None:
    client = LLMClient(provider="totally-not-real", model="x", api_key="x")
    with pytest.raises(LLMError) as excinfo:
        client._get_provider_fn()
    msg = str(excinfo.value)
    # Built-ins + presets should both appear.
    assert "groq" in msg
    assert "openai" in msg
    assert "register_provider" in msg


def test_openai_compat_preset_routes_to_call_openai() -> None:
    client = LLMClient(provider="groq", api_key="gsk-x")
    assert client._get_provider_fn() == client._call_openai


def test_native_gemini_preset_routes_to_dedicated_wire() -> None:
    client = LLMClient(provider="gemini", api_key="g-x")
    fn = client._get_provider_fn()
    # The Gemini wire is wrapped in a closure that delegates to the
    # ``acomplete_gemini`` free function; the closure name reflects that.
    assert fn != client._call_openai
    # And it has not picked up the openai-compat path either.
    assert fn != client._call_anthropic


def test_capability_response_format_fallback_rewrites_system_prompt() -> None:
    """Presets that declare no native response_format augment the system prompt."""
    register_provider(
        ProviderPreset(
            key="no-rf-test",
            base_url="http://localhost:9999/v1",
            env_var="NO_RF_KEY",
            default_model="m",
            wire="openai_compat",
            capabilities={"tools": True, "response_format": False, "streaming": True},
        )
    )
    try:
        from fastaiagent.llm.message import UserMessage

        client = LLMClient(provider="no-rf-test", api_key="x")
        body, _headers = client._build_openai_body(
            [UserMessage("hi")],
            tools=None,
            response_format={"type": "json_object"},
        )
        assert "response_format" not in body
        # System prompt should now contain the augmentation guidance.
        sys_msg = next((m for m in body["messages"] if m.get("role") == "system"), None)
        assert sys_msg is not None
        assert "JSON" in sys_msg["content"]
    finally:
        unregister_provider("no-rf-test")


def test_capability_parallel_tool_calls_dropped_when_unsupported() -> None:
    register_provider(
        ProviderPreset(
            key="no-ptc-test",
            base_url="http://localhost:9998/v1",
            env_var="NO_PTC_KEY",
            default_model="m",
            wire="openai_compat",
            capabilities={
                "tools": True,
                "response_format": "native",
                "streaming": True,
                "parallel_tool_calls": False,
            },
        )
    )
    try:
        from fastaiagent.llm.message import UserMessage

        client = LLMClient(
            provider="no-ptc-test",
            api_key="x",
            parallel_tool_calls=True,
        )
        body, _headers = client._build_openai_body(
            [UserMessage("hi")],
            tools=[{"function": {"name": "search", "parameters": {"type": "object"}}}],
        )
        assert "parallel_tool_calls" not in body
    finally:
        unregister_provider("no-ptc-test")


def test_max_tokens_uses_classic_field_for_preset_providers() -> None:
    """Classic Chat Completions APIs (Groq, OpenRouter, ...) reject
    max_completion_tokens. Preset providers should use ``max_tokens``."""
    from fastaiagent.llm.message import UserMessage

    client = LLMClient(provider="groq", api_key="x", max_tokens=128)
    body, _headers = client._build_openai_body([UserMessage("hi")], tools=None)
    assert body.get("max_tokens") == 128
    assert "max_completion_tokens" not in body


def test_max_tokens_uses_new_field_for_openai() -> None:
    from fastaiagent.llm.message import UserMessage

    client = LLMClient(provider="openai", api_key="sk-x", max_tokens=128)
    body, _headers = client._build_openai_body([UserMessage("hi")], tools=None)
    assert body.get("max_completion_tokens") == 128
    assert "max_tokens" not in body


def test_builtins_constant_matches_reserved_keys() -> None:
    """Sanity: the in-module list and the registry's reserved set agree."""
    from fastaiagent.llm.providers import reserved_keys

    # ``test`` is reserved by the registry but not a built-in code path —
    # the testing package serves it. So _BUILTIN_PROVIDERS is a strict
    # subset of reserved_keys().
    assert _BUILTIN_PROVIDERS <= reserved_keys()


def test_unknown_provider_no_api_key_required_for_init() -> None:
    """Construction should succeed even without an API key — failure is
    deferred to first use so callers can introspect the preset first."""
    # No env var set, no api_key passed
    if "MISTRAL_API_KEY" in os.environ:
        pytest.skip("MISTRAL_API_KEY is set; this test would not exercise the missing-key path")
    client = LLMClient(provider="mistral")
    assert client.api_key is None
    assert client._preset is not None
