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

The default model is what `LLMClient(provider=...)` uses when `model=` is
omitted — since 1.91.0. Before, every provider fell back to `gpt-4o-mini`, an
OpenAI id no preset provider serves, so an omitted model could never work.

| Key | Wire | Default model | API key env var | Tools | Streaming | `response_format` |
|---|---|---|---|---|---|---|
| `gemini` | native | `gemini-3.8-flash` | `GEMINI_API_KEY` | ✓ | ✓ | native (`responseSchema`) |
| `groq` | OpenAI-compat | `openai/gpt-oss-120b` | `GROQ_API_KEY` | ✓ | ✓ | native |
| `openrouter` | OpenAI-compat | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` | ✓ | ✓ | native |
| `deepseek` | OpenAI-compat | `deepseek-chat` | `DEEPSEEK_API_KEY` | ✓ | ✓ | native |
| `together` | OpenAI-compat | `meta-llama/Llama-3.1-70B-Instruct-Turbo` | `TOGETHER_API_KEY` | ✓ | ✓ | native |
| `fireworks` | OpenAI-compat | `accounts/fireworks/models/llama-v3p1-70b-instruct` | `FIREWORKS_API_KEY` | ✓ | ✓ | native |
| `perplexity` | OpenAI-compat | `sonar` | `PERPLEXITY_API_KEY` | ✗ | ✓ | system-prompt |
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

**Bedrock** uses the Converse API through `boto3` (credentials and region from
the usual AWS chain; `LLMClient(..., region=...)` overrides the region).
The region is part of `LLMClient.to_dict()` since 1.91.0, so a replayed or
plane-dispatched Bedrock agent calls the same region; before, it was dropped and
the rebuilt client used `us-east-1`. It is the one constructor extra that is
serialized — others may be credentials and stay local.
`base_url`, when set, is the Converse endpoint — a VPC endpoint or a proxy.
Tools are sent as `toolConfig`; `tool_choice="required"` maps to Converse's
`any`, and `"none"` is not available on Converse, so it logs a warning and the
model may still call a tool.

!!! warning "Fixed in 1.90.0"
    Before 1.90.0 the Bedrock provider sent messages in the OpenAI shape, which
    `boto3` refuses before sending (`ParamValidationError`, even for a plain
    "hi"), and never sent tools — the table above said it did. Verified against
    the Converse wire format with a real `boto3` client; not yet against a live
    AWS account.

## Gemini 3.5 and later — the recommended Gemini models

Use a current Gemini model: `gemini-3.8-flash` (the default), `gemini-4-argon`,
`gemini-3.5-flash-lite`, or the `gemini-flash-latest` / `gemini-pro-latest`
aliases. Every one of them runs the full agent loop — tools, `run` and `astream`,
pause and resume — and is checked live by `tests/e2e/test_gemini_latest_live.py`.

```python
from fastaiagent import Agent, LLMClient

agent = Agent(
    name="assistant",
    llm=LLMClient(provider="gemini", model="gemini-3.8-flash"),  # GEMINI_API_KEY
    tools=[...],
)
```

**Thought signatures are handled for you.** Gemini 3.5+ attaches an opaque
`thoughtSignature` to the first function call of a turn and rejects the next
request (HTTP 400, "Function call is missing a thought_signature") unless it is
sent back. The SDK keeps it on `ToolCall.signature`, returns it on every later
request, and saves it in checkpoints so a resumed run sends it too. It never
appears on a span. A tool turn the model did not sign — history from another
provider, `messages=`, a checkpoint written before 1.91.0 — is sent with Google's
validator-skip value instead, so switching a conversation onto Gemini works.

**Pricing** comes from Google's published list prices, including the increases
Google has announced: `gemini-4-argon` doubles on 2026-12-01 and
`gemini-3.6-flash` / `gemini-3.8-flash` double on 2027-01-01, and a cost is priced
at the rate in force on the day it is computed. `gemini-3.5-flash`,
`gemini-3.7-flash` and the `-latest` aliases are not on Google's pricing page, so
their runs report `cost_known=False` (and a `cost_limit` rule cannot pass them)
until you give them a rate — see [Cost tracking](../ui/cost-tracking.md).

!!! warning "Fixed in 1.91.0"
    Before 1.91.0 the provider dropped the signature, so a tool-using agent failed
    on its **second turn** on every Gemini model after 2.5 — `run` and `astream`
    alike — and Gemini 3.x had no rates, so every run was unpriced. `gemini-2.5-*`
    was unaffected and still works.

**Gemini thinking tokens are completion tokens.** A Gemini thinking model reports
`thoughtsTokenCount` beside the reply's own tokens; Google bills it at the output
rate. Since 1.90.0 `completion_tokens` (and so `AgentResult.cost`, the
`cost_limit` guardrail and the streamed `Usage`) include it. Before, the thinking
was counted in `tokens_used` but never priced, and `astream` reported fewer
tokens than `run`.

## What a model can take: `model_info`

```python
from fastaiagent import LLMClient

info = LLMClient(provider="anthropic", model="claude-haiku-4-5").model_info
info.context_window      # 200000
info.max_output_tokens   # 64000
info.supports_reasoning  # True
info.supports_cache      # True
info.supports_vision     # True
info.supports_pdf        # True
```

`model_info` returns a `ModelInfo` (since 1.91.0). The figures come from the
providers' own sources only: OpenAI's model pages, Anthropic's Models API
(`max_input_tokens` / `max_tokens`) and Gemini's Models API (`inputTokenLimit` /
`outputTokenLimit` / `thinking`). It covers GPT-6, GPT-5.x, GPT-4.1/4o and the
o-series; Claude 5.x and 4.x; and Gemini 4, 3.x and 2.5. A model it does not know
gets `None` for every field — unknown, never guessed. `supports_vision` and
`supports_pdf` are read from the same registry the request formatter uses, so
they cannot disagree with what a request does.

**Anthropic and Bedrock send the model's own maximum** as `max_tokens` when
neither the client nor the call sets one — 64,000 for `claude-haiku-4-5`, 128,000
for Claude 4.6+ and 5.x. Before 1.91.0 they sent 4096, which cut long answers
short (`finish_reason="max_tokens"`). A model `model_info` does not know still
gets 4096. Set `max_tokens=` to cap replies yourself.

## When a model fails: retries, rate limits, fallbacks

```python
from fastaiagent import LLMClient

llm = LLMClient(
    provider="openai",
    model="gpt-5-mini",
    max_retries=2,
    fallbacks=[
        LLMClient(provider="anthropic", model="claude-haiku-4-5"),
        LLMClient(provider="gemini", model="gemini-3.8-flash"),
    ],
)
```

- **Retries** (`max_retries`, default 0) cover a `429`, a `5xx` and — since
  1.91.0 — a refused connection, a reset or a timeout.
- **A `429` is an `LLMRateLimitError`** (a subclass of `LLMProviderError`, so
  existing handlers still catch it) with `retry_after`: the wait the provider
  asked for in `Retry-After` or `retry-after-ms`. A retry waits exactly that
  (capped at 60 s) instead of the blind 1 s, 2 s, 4 s… backoff.
- **`fallbacks`** are tried in order once a client's own retries are spent, on a
  rate limit, a server error, a connection error, a timeout, or a **context
  overflow** (the prompt is too long for that model — OpenAI's
  `context_length_exceeded`, Anthropic's "prompt is too long", Gemini's token
  limit). A plain `400` is not retried elsewhere: every model would refuse the
  same bad request. Each attempt is its own `llm.*` span, and the run is priced
  at the model that answered.
- **A stream falls back only before its first event.** After that the caller
  already holds part of one model's answer, so the error is raised instead.
- **`max_concurrency=N`** caps how many calls one client has in flight; the rest
  wait.

Fallbacks and `max_concurrency` are local: they are not part of
`LLMClient.to_dict()`, so a plane-distributed or replayed agent runs its one
model. Live-checked: a real OpenAI and a real Anthropic context overflow each
fell back (`tests/e2e/test_llm_resilience_live.py`).

## Provider options the SDK has no parameter for

```python
llm = LLMClient(
    provider="openai",
    model="gpt-5-mini",
    extra_body={"service_tier": "flex", "safety_identifier": "user-123"},
    extra_headers={"X-Client-Team": "billing"},
)
llm.complete(messages, extra_body={"service_tier": "priority"})  # per call, merged over
```

`extra_body` is merged into every request body and `extra_headers` into its
headers, for every provider — OpenAI and every OpenAI-compatible preset,
Anthropic, Gemini, Ollama, and an injected OpenAI SDK client (as its own
`extra_body` / `extra_headers`). Bedrock sends `extra_body` as Converse's
`additionalModelRequestFields`. The SDK's own body keys (`model`, `messages`,
tools…) win over `extra_body`, and the credential header cannot be replaced.
**Neither is part of `to_dict()`** — a header often carries a credential.

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

## OpenAI Decisions API

`LLMClient.decide()` / `adecide()` (1.84.0) call OpenAI's `POST /v1/decisions`.
That works with `provider="openai"`, with `provider="custom"` behind a gateway
that serves `/decisions`, and with an injected `openai_client`. With an injected
client it uses `client.decisions.create` on openai ≥ 3.26, and the client's
generic `post` on older versions, so no upgrade is needed. Every other provider
raises `LLMError`. See [OpenAI Decisions API](decisions.md).

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

Every provider the *server process* knows about — the built-ins and the
presets shipped in `fastaiagent.llm.providers` — appears in the
[Prompt Playground](../ui/playground.md) provider dropdown automatically
(via `GET /api/playground/models`). No UI rebuild is required. Providers
whose API-key env var is not set show up disabled with a tooltip pointing
to the right variable.

!!! warning "Presets you register yourself are not picked up"
    `fastaiagent ui` starts a standalone server that does **not** import your
    application code, so a `register_provider(...)` call in your project
    never runs in that process and won't appear in the dropdown. To use such
    a provider from the Playground, either register it somewhere the UI
    process imports, or drive it from your own code where the registration
    has run.

    Model *ids* are a different matter and need no code at all: the model
    field accepts free text, and `.fastaiagent/models.json` changes the
    suggestions. See
    [Customising the model list](../ui/playground.md#customising-the-model-list).

## Need a provider that isn't listed?

See [Custom providers](custom-provider.md) for `register_provider()` —
add an internal LLM gateway or a new vendor in five lines. What a preset's
capability flags do to the request body, with the body recorded, is in
[The Model Call Breaks at the Boundaries](llm-boundaries.md#3-between-a-providers-name-and-the-code-that-serves-it-a-name-a-wire-a-set-of-flags).
