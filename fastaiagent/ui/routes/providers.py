"""Read-only endpoint exposing the LLM provider preset registry.

The local UI uses this to populate dropdowns (Playground provider picker,
trace-detail badge metadata) without baking the list into the React
bundle. Returns one entry per *known* provider — built-in
(``openai``, ``anthropic``, ``ollama``, ``azure``, ``bedrock``,
``custom``, ``test``) and registered presets — with capability flags so
the UI can grey out fields a provider doesn't support.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from fastaiagent.llm.providers import get_preset, list_provider_keys, reserved_keys
from fastaiagent.ui.deps import require_session

router = APIRouter(prefix="/api/providers", tags=["providers"])


# Defaults for the built-in providers — kept in sync with
# ``LLMClient._default_base_url`` and the API key fall-throughs in
# ``_build_openai_body`` / ``_build_anthropic_body``. ``test`` is the
# stand-in provider exposed by ``fastaiagent.testing.TestModel``.
_BUILTIN_INFO: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "wire": "builtin_openai",
        "capabilities": {
            "tools": True,
            "response_format": "native",
            "streaming": True,
            "parallel_tool_calls": True,
        },
        "description": "OpenAI Chat Completions.",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-3-5-haiku-latest",
        "wire": "builtin_anthropic",
        "capabilities": {
            "tools": True,
            "response_format": "system_prompt",
            "streaming": True,
            "parallel_tool_calls": False,
        },
        "description": "Anthropic Messages API.",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "env_var": "",
        "default_model": "llama3.1",
        "wire": "builtin_ollama",
        "capabilities": {
            "tools": True,
            "response_format": "native",
            "streaming": True,
            "parallel_tool_calls": False,
        },
        "description": "Ollama — local model server.",
    },
    "azure": {
        "base_url": "",
        "env_var": "OPENAI_API_KEY",
        "default_model": "",
        "wire": "builtin_openai",
        "capabilities": {
            "tools": True,
            "response_format": "native",
            "streaming": True,
            "parallel_tool_calls": True,
        },
        "description": "Azure OpenAI.",
    },
    "bedrock": {
        "base_url": "",
        "env_var": "",
        "default_model": "",
        "wire": "builtin_bedrock",
        "capabilities": {
            "tools": True,
            "response_format": False,
            "streaming": False,
            "parallel_tool_calls": False,
        },
        "description": "AWS Bedrock (via boto3).",
    },
    "custom": {
        "base_url": "",
        "env_var": "OPENAI_API_KEY",
        "default_model": "",
        "wire": "openai_compat",
        "capabilities": {
            "tools": True,
            "response_format": "native",
            "streaming": True,
            "parallel_tool_calls": True,
        },
        "description": "Custom OpenAI-compatible endpoint.",
    },
    "test": {
        "base_url": "",
        "env_var": "",
        "default_model": "test-model",
        "wire": "test",
        "capabilities": {
            "tools": True,
            "response_format": "native",
            "streaming": True,
            "parallel_tool_calls": False,
        },
        "description": "Deterministic stand-in (fastaiagent.testing.TestModel).",
    },
}


def _entry_for(key: str) -> dict[str, Any]:
    if key in _BUILTIN_INFO:
        info = dict(_BUILTIN_INFO[key])
        info["key"] = key
        info["builtin"] = True
        return info
    preset = get_preset(key)
    if preset is None:  # should not happen — list_provider_keys is the source
        return {
            "key": key,
            "base_url": "",
            "env_var": "",
            "default_model": "",
            "wire": "unknown",
            "capabilities": {},
            "description": "",
            "builtin": False,
        }
    return {
        "key": preset.key,
        "base_url": preset.base_url,
        "env_var": preset.env_var,
        "default_model": preset.default_model,
        "wire": preset.wire,
        "capabilities": dict(preset.capabilities),
        "description": preset.description,
        "builtin": False,
    }


@router.get("")
def list_providers(_user: str = Depends(require_session)) -> dict[str, Any]:
    """Return all known providers (built-ins + registered presets).

    Shape:

        {
          "providers": [
            {
              "key": "groq",
              "base_url": "https://api.groq.com/openai/v1",
              "env_var": "GROQ_API_KEY",
              "default_model": "llama-3.1-70b-versatile",
              "wire": "openai_compat",
              "capabilities": {"tools": true, "response_format": "native", ...},
              "description": "...",
              "builtin": false,
            },
            ...
          ],
          "reserved": ["openai", "anthropic", "ollama", "azure", "bedrock", "custom", "test"]
        }
    """
    keys = list_provider_keys()
    providers = [_entry_for(k) for k in keys]
    return {
        "providers": providers,
        "reserved": sorted(reserved_keys()),
    }
