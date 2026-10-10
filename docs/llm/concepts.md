# Concepts & Mental Model

This page explains **what** `LLMClient` is, **why** the SDK puts a layer between
your agent and the model provider, and **the concept of how** that layer works —
the normalize → call → parse cycle, how tools stay provider-agnostic, and where
numbers like cost and latency actually come from. For the provider list see
[Providers](providers.md); to add your own see [Custom Providers](custom-provider.md).

## What it is

`LLMClient` is the one object every agent, chain, judge, and simulated user
talks to when it needs a model. You give it a provider and a model:

```python
llm = LLMClient(provider="openai", model="gpt-4.1")
```

…and everything above it — the agent loop, guardrails, evals — is written
against a single, stable shape: you pass `Message` objects in and get an
`LLMResponse` back.

## Why the layer exists

Because providers genuinely disagree, and not just cosmetically. Anthropic
hoists the system prompt out of the message list into its own field, expresses
assistant tool calls as `tool_use` content blocks, and sends tool *results* back
as **user** turns. Parameter names differ (`max_tokens` vs
`max_completion_tokens`). Usage keys differ (`input_tokens` vs
`prompt_tokens`). Finish reasons differ (`end_turn` vs `stop`). Features differ
— some accept a JSON schema for structured output, some don't; some reject
`parallel_tool_calls` outright.

Without a layer, every one of those differences leaks upward and your agent code
grows provider branches. With it, swapping `provider="openai"` for
`provider="anthropic"` changes one line and nothing else.

## The concept of how: normalize → call → parse

The whole design is a sandwich. **Both ends are provider-neutral; only the
middle is provider-shaped.**

```
Message[]  ──normalize──▶  provider wire body  ──HTTP──▶  provider JSON
                                                              │
LLMResponse  ◀──parse──────────────────────────────────────────┘
(content, tool_calls, usage, finish_reason, latency_ms, parsed)
```

1. **Normalize** — your `Message` list is rendered into the provider's actual
   wire shape (`to_provider_dict(provider)`), including the structural surgery
   Anthropic needs.
2. **Call** — one HTTP request, wrapped in an `llm.<provider>.<model>` span
   carrying GenAI semantic-convention attributes (`gen_ai.request.model`,
   `gen_ai.usage.*`, `gen_ai.response.finish_reason`).
3. **Parse** — the provider's response is mapped back into a uniform
   `LLMResponse`. Token keys are renamed, finish reasons are translated, and
   tool-call `arguments` are JSON-decoded from strings into real dicts so
   callers never parse JSON themselves.

!!! info "Verified against a live run"
    A call returned `content='PONG'`, `finish_reason='stop'`, `usage` with
    `prompt_tokens`/`completion_tokens`, and `latency_ms=1554`. A tool-enabled
    call returned `finish_reason='tool_calls'` and a `ToolCall` whose
    `arguments` was already a **`dict`** (`{'city': 'Paris'}`), not a JSON
    string.

### Tools: OpenAI format as the internal lingua franca

Tools are passed as a plain `list[dict]` in **OpenAI function-calling shape**,
and every non-OpenAI adapter converts *out of* that shape (Anthropic's
`input_schema`, Gemini's native format). That's a deliberate design choice: one
canonical tool representation everything else translates from — which is why
`Tool.to_openai_format()` is what the agent loop produces regardless of which
provider is configured.

### Two tiers of provider

There are far fewer implementations than there are provider names:

- **Built-ins** — `openai`, `anthropic`, `ollama`, `azure`, `bedrock`, `custom`
  have real code paths (Azure and `custom` reuse the OpenAI-compatible one).
- **Presets** — everything else is *configuration over an existing wire*: a
  `ProviderPreset` supplies `base_url`, the API-key env var, a default model,
  and a `wire` type. Most collapse onto the OpenAI-compatible adapter; Gemini
  has its own native wire.

!!! info "Verified against a live run"
    19 provider keys resolve from just **12 presets** plus the built-ins — e.g.
    `deepseek`, `fireworks`, and `cerebras` all declare `wire="openai_compat"`,
    while `gemini` declares `wire="native_gemini"`. Adding a provider usually
    means registering a preset, not writing an adapter.

### Capabilities are graceful degradation, not documentation

A preset's capability flags change *behavior*, quietly. If a provider can't take
a native `response_format`, the JSON schema is injected into the system prompt
instead. If it would 400 on `parallel_tool_calls`, that field is dropped. You
still get a result — just via the soft path. Worth knowing when you're
comparing providers and one seems "worse" at structured output.

## Structured output, in three layers

The same feature exists at increasing strictness:

1. **`LLMClient.acomplete(output_type=...)`** — one shot. The schema is sent,
   the response is parsed into `.parsed`; on a parse failure `.parsed` is
   `None`.
2. **Agent, strict mode** — for OpenAI/Azure the schema is sent in strict form
   (`additionalProperties: false`, everything required).
3. **Agent, self-correction** — on a parse failure the agent re-asks with the
   *human-readable* parse error ("Your previous response could not be used:
   …reply with ONLY the JSON value"), up to `output_retries` (default 2). That's
   why parse errors are written as prose: they're fed back to the model.

!!! info "Verified against a live run"
    `complete([...], output_type=City)` returned raw content
    `'{"name":"Paris","country":"France"}'` and a `.parsed` that was a real
    `City` instance.

## Streaming is the same call in a different shape

`astream()` yields typed events — `TextDelta`, `ToolCallStart`, `ToolCallEnd`,
`Usage`, `StreamDone` (plus `HandoffEvent` for swarms) — while `stream()` folds
those same events back into a normal `LLMResponse`. Inside an agent, the tool
loop re-yields events to you as they arrive while accumulating text and usage,
and deliberately **suppresses `StreamDone` between turns** so a multi-turn tool
loop looks like one continuous stream.

## Where the numbers come from — read this before trusting a dashboard

- **Latency** is measured **client-side**, wall-clock around the provider call.
  It includes network time, so it's "what your process experienced," not the
  provider's compute time.
- **Cost is not reported by the provider.** It's computed locally from a
  pricing table by **longest-prefix match** on the model name — so
  `gpt-4o-mini-2024-07-18` resolves to the `gpt-4o-mini` rate, and an unknown
  model yields **no cost at all** rather than a wrong one.

!!! info "Verified against a live run"
    `compute_cost_usd("gpt-4o-mini-2024-07-18", 1M, 1M)` → `0.75` via prefix
    match; `compute_cost_usd("totally-unknown-model", …)` → `None`.

## Two defaults worth knowing

- **Retries are off by default** (`max_retries=0`). When enabled, retries fire
  only on `429` and `5xx`, with exponential backoff (1s, 2s, 4s… capped at 30s)
  — driven by a normalized `LLMProviderError` that carries the HTTP status, so
  the policy is provider-agnostic.
- **The HTTP timeout is a fixed 120s** at the transport layer, not a constructor
  argument.

## One subtlety when debugging traces

There are **two serializers**, and only one is the wire format. `to_provider_dict()`
builds the real request. `to_openai_format()` builds a compact *summary* used
for spans and logs — deliberately, so that writing a trace never base64-encodes
an image or renders a PDF just to produce a log line. So if you read
`gen_ai.request.messages` on a span and see `{"type": "image", "size_bytes": …}`
instead of the image payload, that's by design.

## Imports

Top-level `fastaiagent` exports `LLMClient`, `Message`, `StreamEvent`, and
`TextDelta`. The message factories (`UserMessage`, `SystemMessage`, …),
`ToolCall`, `LLMResponse`, and the provider registry come from
`fastaiagent.llm`.

## Next steps

- [The Model Call Breaks at the Boundaries](llm-boundaries.md) — the six places a call goes wrong, each with a diagram and a proof you can run
- [Providers](providers.md) — the supported provider table, env vars, capabilities
- [Custom Providers](custom-provider.md) — register a preset for a gateway or new vendor
- [Structured Output](../structured-output/index.md) — `output_type`, strict mode, retries
- [Streaming](../streaming/index.md) — consuming the event stream
- [Agents — the run loop](../agents/concepts.md#the-run-loop) — who calls this and when
