"""Built-in provider presets.

Importing this module eagerly registers presets for Gemini, Groq, OpenRouter,
DeepSeek, Together, Fireworks, Perplexity, Mistral, LM Studio, vLLM,
SambaNova, and Cerebras with the registry. Capabilities are best-effort;
when a preset claims ``response_format=False``, ``LLMClient`` falls back to
system-prompt augmentation via ``_augment_system_for_response_format``.
"""

from __future__ import annotations

from fastaiagent.llm.providers.registry import ProviderPreset, register_provider

_OPENAI_COMPAT_CAPS = {
    "tools": True,
    "response_format": "native",
    "streaming": True,
    "parallel_tool_calls": False,
}


def _register_builtins() -> None:
    register_provider(
        ProviderPreset(
            key="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            env_var="GEMINI_API_KEY",
            # 2.5-flash is the current default for new keys (Google deprecated
            # 2.0-flash for new users in 2026). The legacy 2.0-flash names
            # still work for older keys; pass model= explicitly to override.
            default_model="gemini-2.5-flash",
            wire="native_gemini",
            capabilities={
                "tools": True,
                "response_format": "native",
                "streaming": True,
                "parallel_tool_calls": False,
            },
            description="Google Gemini via native generativelanguage.googleapis.com.",
        )
    )
    register_provider(
        ProviderPreset(
            key="groq",
            base_url="https://api.groq.com/openai/v1",
            env_var="GROQ_API_KEY",
            default_model="llama-3.1-70b-versatile",
            wire="openai_compat",
            capabilities=_OPENAI_COMPAT_CAPS,
            description="Groq — fast inference for open models.",
        )
    )
    register_provider(
        ProviderPreset(
            key="openrouter",
            base_url="https://openrouter.ai/api/v1",
            env_var="OPENROUTER_API_KEY",
            default_model="openai/gpt-4o-mini",
            wire="openai_compat",
            capabilities=_OPENAI_COMPAT_CAPS,
            description="OpenRouter — unified access to many model providers.",
        )
    )
    register_provider(
        ProviderPreset(
            key="deepseek",
            base_url="https://api.deepseek.com/v1",
            env_var="DEEPSEEK_API_KEY",
            default_model="deepseek-chat",
            wire="openai_compat",
            capabilities={
                "tools": True,
                "response_format": "native",
                "streaming": True,
                "parallel_tool_calls": False,
            },
            description="DeepSeek — OpenAI-compatible.",
        )
    )
    register_provider(
        ProviderPreset(
            key="together",
            base_url="https://api.together.xyz/v1",
            env_var="TOGETHER_API_KEY",
            default_model="meta-llama/Llama-3.1-70B-Instruct-Turbo",
            wire="openai_compat",
            capabilities=_OPENAI_COMPAT_CAPS,
            description="Together AI — open models.",
        )
    )
    register_provider(
        ProviderPreset(
            key="fireworks",
            base_url="https://api.fireworks.ai/inference/v1",
            env_var="FIREWORKS_API_KEY",
            default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            wire="openai_compat",
            capabilities=_OPENAI_COMPAT_CAPS,
            description="Fireworks — open models.",
        )
    )
    register_provider(
        ProviderPreset(
            key="perplexity",
            base_url="https://api.perplexity.ai",
            env_var="PERPLEXITY_API_KEY",
            default_model="llama-3.1-sonar-small-128k-online",
            wire="openai_compat",
            capabilities={
                "tools": False,
                "response_format": False,
                "streaming": True,
                "parallel_tool_calls": False,
            },
            description="Perplexity — search-augmented chat.",
        )
    )
    register_provider(
        ProviderPreset(
            key="mistral",
            base_url="https://api.mistral.ai/v1",
            env_var="MISTRAL_API_KEY",
            default_model="mistral-large-latest",
            wire="openai_compat",
            capabilities=_OPENAI_COMPAT_CAPS,
            description="Mistral AI — proprietary + open models.",
        )
    )
    register_provider(
        ProviderPreset(
            key="lmstudio",
            base_url="http://localhost:1234/v1",
            env_var="LMSTUDIO_API_KEY",
            default_model="local-model",
            wire="openai_compat",
            capabilities={
                "tools": True,
                "response_format": "native",
                "streaming": True,
                "parallel_tool_calls": False,
            },
            description="LM Studio — local model server. No API key required.",
        )
    )
    register_provider(
        ProviderPreset(
            key="vllm",
            base_url="http://localhost:8000/v1",
            env_var="VLLM_API_KEY",
            default_model="local-model",
            wire="openai_compat",
            capabilities=_OPENAI_COMPAT_CAPS,
            description="vLLM — local high-throughput model server.",
        )
    )
    register_provider(
        ProviderPreset(
            key="sambanova",
            base_url="https://api.sambanova.ai/v1",
            env_var="SAMBANOVA_API_KEY",
            default_model="Meta-Llama-3.1-70B-Instruct",
            wire="openai_compat",
            capabilities={
                "tools": True,
                "response_format": False,
                "streaming": True,
                "parallel_tool_calls": False,
            },
            description="SambaNova Cloud.",
        )
    )
    register_provider(
        ProviderPreset(
            key="cerebras",
            base_url="https://api.cerebras.ai/v1",
            env_var="CEREBRAS_API_KEY",
            default_model="llama3.1-70b",
            wire="openai_compat",
            capabilities={
                "tools": True,
                "response_format": False,
                "streaming": True,
                "parallel_tool_calls": False,
            },
            description="Cerebras — ultra-fast inference.",
        )
    )


_register_builtins()
