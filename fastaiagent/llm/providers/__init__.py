"""Provider preset registry + native-wire providers.

Public API:
    from fastaiagent.llm.providers import (
        ProviderPreset,
        register_provider,
        get_preset,
        list_presets,
        list_provider_keys,
    )

Importing this package eagerly registers built-in presets (Gemini, Groq,
OpenRouter, DeepSeek, Together, Fireworks, Perplexity, Mistral, LM Studio,
vLLM, SambaNova, Cerebras) so ``LLMClient(provider="groq", ...)`` works
out of the box.
"""

# Eagerly run the seed-preset registrations.
from fastaiagent.llm.providers import _presets as _presets  # noqa: F401
from fastaiagent.llm.providers.registry import (
    ProviderPreset,
    get_preset,
    list_presets,
    list_provider_keys,
    register_provider,
    reserved_keys,
    unregister_provider,
)

__all__ = [
    "ProviderPreset",
    "register_provider",
    "unregister_provider",
    "get_preset",
    "list_presets",
    "list_provider_keys",
    "reserved_keys",
]
