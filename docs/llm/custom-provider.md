# Custom Providers

If you run an internal LLM gateway, or want to use a vendor that
fastaiagent doesn't ship a preset for, register it once at app start-up:

```python
from fastaiagent.llm.providers import register_provider, ProviderPreset

register_provider(ProviderPreset(
    key="my-internal-llm",
    base_url="https://llm.internal.corp/v1",
    env_var="INTERNAL_LLM_KEY",
    default_model="house-7b",
    wire="openai_compat",
    capabilities={
        "tools": True,
        "response_format": "native",
        "streaming": True,
        "parallel_tool_calls": False,
    },
    description="Internal LLM gateway behind corp SSO.",
))
```

Now anywhere in your codebase:

```python
from fastaiagent import Agent, LLMClient

agent = Agent(name="bot", llm=LLMClient(provider="my-internal-llm",
                                        model="house-7b"))
```

`base_url` and `api_key` are filled in from the preset; capabilities flow
into the body builder so the request shape matches what your gateway
expects.

## Wire types

| Wire | Use it when |
|---|---|
| `openai_compat` | Your endpoint speaks OpenAI Chat Completions (most third-party APIs). |
| `native_gemini` | Reserved for the Google `generativelanguage` protocol; not user-extensible today. |

## Capability flags

| Key | Type | What it controls |
|---|---|---|
| `tools` | `bool` | Whether to forward `tools=` on requests. |
| `response_format` | `"native"`, `"system_prompt"`, or `False` | If `False`, fastaiagent augments the system prompt with JSON-only instructions instead of sending `response_format` (which would 400 on providers without native support). |
| `streaming` | `bool` | Whether `astream()` is supported. |
| `parallel_tool_calls` | `bool` | Whether to forward `parallel_tool_calls=` (some providers reject the field). |

Unknown capability keys are accepted and stored — useful for downstream
tooling that wants to read them off the preset.

## Reserved keys

These six keys are reserved by fastaiagent's first-class code paths and
cannot be re-registered: `openai`, `anthropic`, `ollama`, `azure`,
`bedrock`, `custom`, `test`.

## Removing a preset at runtime

```python
from fastaiagent.llm.providers import unregister_provider

unregister_provider("my-internal-llm")
```

This is mostly useful in tests; for application code, register once at
import time and leave it in place.

## Visibility in the local UI

Custom presets show up automatically at `GET /api/providers`, so the
Playground dropdown picks them up the next time the page is opened.
