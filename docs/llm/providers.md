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

## TLS verification (corporate gateways, self-signed certs)

By default `LLMClient` verifies the provider's TLS certificate against the
public CA roots (certifi). When the provider sits behind a corporate gateway or
proxy that presents a private/self-signed certificate — common with **Azure
OpenAI on Azure ML** — pass `verify`:

```python
from fastaiagent import Agent, LLMClient

# Trust a corporate CA bundle (the issuing chain — root + intermediates, PEM):
llm = LLMClient(
    provider="azure",
    model="<deployment>",
    base_url="https://<gateway>/openai/v1",
    api_key="<key>",
    verify="/path/to/corporate-ca.pem",
)

# Or disable verification entirely (development only — see warning below):
llm = LLMClient(provider="azure", model="<deployment>",
                base_url="https://<gateway>/openai/v1", verify=False)

agent = Agent(name="bot", llm=llm)
```

`verify` accepts the same shapes as httpx:

| Value | Meaning |
|---|---|
| `True` *(default)* | Verify against the public CA roots. |
| `False` | **Disable** verification. Emits a security warning; LLM traffic can be intercepted. Use only in development. |
| `"/path/to/ca.pem"` | Trust this PEM CA bundle. Must contain the **issuing CA chain**, not the server's leaf certificate. |
| `ssl.SSLContext` | A fully custom context (advanced). |

!!! warning
    `verify=False` turns off certificate checking for all LLM calls on that
    client. Prefer supplying the gateway's CA bundle via `verify="<path>"`.

**Without code (e.g. an Azure ML `score.py` deployment):** set the
`FASTAIAGENT_LLM_VERIFY` environment variable — `false`/`true` or a CA-bundle
path. It applies when `verify` is left at its default, so you can configure it
via the deployment's `environment_variables`.

`SSL_CERT_FILE` is also honored (it sets the process-wide trust store), but it
must point at the issuing CA chain — pointing it at the server's leaf
certificate yields "unable to get local issuer certificate".

## Bring your own client (Azure OpenAI, classic API, managed identity)

The built-in `azure` provider targets the v1 endpoint
(`https://<resource>.openai.azure.com/openai/v1`) and Bearer auth. For the
**classic** Azure surface — the `/openai/deployments/<deployment>/chat/completions?api-version=...`
URL, or **Entra ID / managed-identity** auth via `azure_ad_token_provider`
— pass a pre-built `openai` SDK client and let `LLMClient` delegate the HTTP
call to it. The client's `base_url`, `api_version`, auth (including
**token refresh**), and `http_client` (e.g. `verify=False`) are all reused;
`LLMClient` still builds the request, applies tools/structured output, emits
`llm.azure.*` spans, and parses the response.

```python
import httpx
from openai import AzureOpenAI
from fastaiagent import Agent, LLMClient

azure = AzureOpenAI(
    azure_endpoint="https://<gateway-or-resource>",
    api_version="2024-10-21",
    azure_ad_token_provider=get_token,        # managed identity — refreshes automatically
    http_client=httpx.Client(verify=False),   # gateway TLS handled by the openai client
)

llm = LLMClient(provider="azure", model="<deployment-name>", openai_client=azure)
agent = Agent(name="bot", llm=llm)
```

Notes:

- `model` is the **Azure deployment name** (what the openai client uses as the
  deployment in the URL).
- When `openai_client` is set, `LLMClient`'s own `base_url` / `api_key` /
  `verify` are **ignored** on that path — the injected client owns transport,
  auth, and TLS.
- Works with sync (`OpenAI`/`AzureOpenAI`) and async
  (`AsyncOpenAI`/`AsyncAzureOpenAI`) clients; a sync client is run off the event
  loop so it never blocks. Streaming is supported too.
- Because token refresh is delegated to the openai SDK, the **same code runs in
  a notebook and an Azure ML `score.py` deployment** without per-request token
  management.

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
