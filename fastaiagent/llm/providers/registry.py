"""Provider preset registry for fastaiagent.llm.LLMClient.

Lets users add new OpenAI-compatible providers (Groq, OpenRouter, DeepSeek,
Together, Fireworks, Perplexity, Mistral, LM Studio, vLLM, SambaNova,
Cerebras, ...) without forking ``client.py``. Native-wire providers (Gemini)
also register here so the dispatcher can route to the right implementation.

Usage:

    from fastaiagent.llm.providers import register_provider, ProviderPreset

    register_provider(ProviderPreset(
        key="my-internal-llm",
        base_url="https://llm.internal.corp/v1",
        env_var="INTERNAL_LLM_KEY",
        default_model="house-7b",
        wire="openai_compat",
    ))

    client = LLMClient(provider="my-internal-llm")  # picks up base_url + env_var

The registry is a public, additive surface. Built-in keys
(``openai``, ``anthropic``, ``ollama``, ``azure``, ``bedrock``, ``custom``)
are reserved and cannot be re-registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

WireType = Literal["openai_compat", "native_gemini"]

# Reserved keys that take the existing built-in code paths in
# ``LLMClient._get_provider_fn``. Registering one of these raises.
_RESERVED_KEYS = frozenset({"openai", "anthropic", "ollama", "azure", "bedrock", "custom", "test"})


@dataclass(frozen=True)
class ProviderPreset:
    """Configuration for an OpenAI-compatible (or native-Gemini) provider."""

    key: str
    base_url: str
    env_var: str
    default_model: str
    wire: WireType = "openai_compat"
    capabilities: dict[str, object] = field(default_factory=dict)
    description: str = ""

    def cap(self, name: str, default: object = None) -> object:
        """Look up a capability flag with a default."""
        return self.capabilities.get(name, default)


_REGISTRY: dict[str, ProviderPreset] = {}


def register_provider(preset: ProviderPreset, *, overwrite: bool = False) -> None:
    """Register a new provider preset.

    Raises ``ValueError`` if the key collides with a reserved built-in or
    with an existing preset (unless ``overwrite=True``).
    """
    if preset.key in _RESERVED_KEYS:
        raise ValueError(
            f"Provider key '{preset.key}' is reserved by a built-in. "
            f"Reserved keys: {sorted(_RESERVED_KEYS)}."
        )
    if not overwrite and preset.key in _REGISTRY:
        raise ValueError(
            f"Provider '{preset.key}' is already registered. "
            f"Pass overwrite=True to replace it."
        )
    _REGISTRY[preset.key] = preset


def get_preset(key: str) -> ProviderPreset | None:
    """Return the preset for ``key`` or ``None`` if unknown."""
    return _REGISTRY.get(key)


def list_presets() -> list[ProviderPreset]:
    """Return a copy of all registered presets, sorted by key."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def list_provider_keys() -> list[str]:
    """Return all known provider keys (built-ins + registered presets)."""
    return sorted(_RESERVED_KEYS | set(_REGISTRY))


def unregister_provider(key: str) -> None:
    """Remove a registered preset. No-op if not present."""
    _REGISTRY.pop(key, None)


def reserved_keys() -> frozenset[str]:
    """Built-in provider keys that have first-class code paths."""
    return _RESERVED_KEYS
