# LLM Providers

`LLMClient` ships with first-class support for the providers in the table
below. Each preset resolves the right `base_url` and reads the API key
from the canonical environment variable, so this is the entire
configuration:

```python
from fastaiagent import Agent, LLMClient

agent = Agent(name="bot", llm=LLMClient(provider="groq",
                                        model="llama-3.1-70b-versatile"))
```

Set the matching env var (e.g. `export GROQ_API_KEY=…`) and you're done.

## Built-in providers

| Key | Wire | Default model | API key env var | Tools | Streaming | `response_format` |
|---|---|---|---|---|---|---|
| `openai` | OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` | ✓ | ✓ | native |
| `anthropic` | Anthropic | (specify) | `ANTHROPIC_API_KEY` | ✓ | ✓ | system-prompt |
| `ollama` | Ollama | (specify) | _(none, local)_ | ✓ | ✓ | native |
| `azure` | OpenAI-compat | (specify) | `OPENAI_API_KEY` | ✓ | ✓ | native |
| `bedrock` | Bedrock (boto3) | (specify) | _(AWS creds)_ | ✓ | ✗ | ✗ |
| `custom` | OpenAI-compat | (specify) | `OPENAI_API_KEY` | ✓ | ✓ | native |
| `test` | _(stand-in)_ | `test-model` | _(none)_ | ✓ | ✓ | n/a |

## Preset providers (registered automatically in v1.8.0)

| Key | Wire | Default model | API key env var | Tools | Streaming | `response_format` |
|---|---|---|---|---|---|---|
| `gemini` | native | `gemini-2.5-flash` | `GEMINI_API_KEY` | ✓ | ✓ | native (`responseSchema`) |
| `groq` | OpenAI-compat | `llama-3.1-70b-versatile` | `GROQ_API_KEY` | ✓ | ✓ | native |
| `openrouter` | OpenAI-compat | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` | ✓ | ✓ | native |
| `deepseek` | OpenAI-compat | `deepseek-chat` | `DEEPSEEK_API_KEY` | ✓ | ✓ | native |
| `together` | OpenAI-compat | `meta-llama/Llama-3.1-70B-Instruct-Turbo` | `TOGETHER_API_KEY` | ✓ | ✓ | native |
| `fireworks` | OpenAI-compat | `accounts/fireworks/models/llama-v3p1-70b-instruct` | `FIREWORKS_API_KEY` | ✓ | ✓ | native |
| `perplexity` | OpenAI-compat | `llama-3.1-sonar-small-128k-online` | `PERPLEXITY_API_KEY` | ✗ | ✓ | system-prompt |
| `mistral` | OpenAI-compat | `mistral-large-latest` | `MISTRAL_API_KEY` | ✓ | ✓ | native |
| `lmstudio` | OpenAI-compat | `local-model` | _(none, `http://localhost:1234/v1`)_ | ✓ | ✓ | native |
| `vllm` | OpenAI-compat | `local-model` | _(none, `http://localhost:8000/v1`)_ | ✓ | ✓ | native |
| `sambanova` | OpenAI-compat | `Meta-Llama-3.1-70B-Instruct` | `SAMBANOVA_API_KEY` | ✓ | ✓ | system-prompt |
| `cerebras` | OpenAI-compat | `llama3.1-70b` | `CEREBRAS_API_KEY` | ✓ | ✓ | system-prompt |

`response_format` column meanings:

- **native** — provider exposes `response_format` (or
  `generationConfig.responseSchema` for Gemini); the LLM returns valid
  JSON on the wire.
- **system-prompt** — fastaiagent injects JSON-only instructions into the
  system prompt as a fallback. The provider returns the same shape, just
  without native enforcement.
- **✗** — not supported; the field is dropped silently.

## Capability fallbacks

When a preset declares a capability as missing, `LLMClient` does the safe
thing instead of erroring:

- `response_format=False` → fastaiagent injects JSON-only guidance into the
  system prompt instead of sending `response_format` over the wire.
- `parallel_tool_calls=False` → fastaiagent drops the field rather than
  passing it along (some providers 400 if it's set).

## Listing providers programmatically

```python
from fastaiagent.llm.providers import list_provider_keys, list_presets

print(list_provider_keys())     # built-ins + presets
for p in list_presets():
    print(p.key, p.base_url, p.env_var, p.capabilities)
```

The same data is exposed at `GET /api/providers` in the local UI for
dropdowns and analytics.

## Local UI integration

Every registered provider — built-in or preset — appears in the
[Prompt Playground](../ui/playground.md) provider dropdown automatically
(via `GET /api/playground/models`). No UI rebuild is required when you
register a new preset; refresh the page and the dropdown picks it up.
Providers whose API-key env var is not set show up disabled with a
tooltip pointing to the right variable.

## Need a provider that isn't listed?

See [Custom providers](custom-provider.md) for `register_provider()` —
add an internal LLM gateway or a new vendor in five lines.
