# Framework Integrations

FastAIAgent ships **two** flavours of integration:

1. **SDK auto-tracing** — patches the OpenAI / Anthropic SDKs so every
   `chat.completions.create()` and `messages.create()` call lands in
   the local trace store with full token / cost / payload data. Use
   when your code calls the provider SDK directly.

2. **Universal agent harness** — wraps **LangChain / LangGraph**,
   **CrewAI**, and **PydanticAI** agents with FastAIAgent's
   observability, eval, guardrails, prompt registry, and KB. Use when
   your agent is built in one of those frameworks and you want the
   same Local UI surface without rewriting.

## Pick your starting point

- I have an existing **LangGraph** / **CrewAI** / **PydanticAI** agent →
  start with [Universal harness — overview](overview.md), then read the
  framework-specific guide.
- I'm calling the **OpenAI / Anthropic SDK directly** → keep reading
  this page.

## Universal harness — per framework

- [Overview & feature matrix](overview.md)
- [LangChain / LangGraph](langchain.md)
- [CrewAI](crewai.md)
- [PydanticAI](pydanticai.md)

---

## OpenAI SDK auto-tracing

Traces all `openai.chat.completions.create()` calls.

```python
import fastaiagent.integrations.openai as openai_integration

openai_integration.enable()

import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
# Trace lands at .fastaiagent/local.db; view with `fastaiagent ui`
```

**What's captured per call:**

- `gen_ai.system`: `"openai"`
- `gen_ai.request.model`: model name
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`
- `gen_ai.request.messages` / `gen_ai.response.content`
- Tool calls + finish reason
- Latency

## Anthropic SDK auto-tracing

```python
import fastaiagent.integrations.anthropic as anthropic_integration

anthropic_integration.enable()

import anthropic
client = anthropic.Anthropic()
client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Same span shape as the OpenAI integration — `gen_ai.system="anthropic"`
and the rest mirrors.

## Disabling

Each integration's `disable()` restores the originals (best-effort —
some surfaces, like LangChain's configure-hook registry, don't expose
a public unregister; the integration's idempotency flag stops the
handler from creating new spans regardless).

```python
openai_integration.disable()
anthropic_integration.disable()
```

## Trace export

Every traced call goes to the same local SQLite store, regardless of
which integration produced it. Use the standard FastAIAgent trace
export tooling — see the [Trace export](../tracing/index.md) docs.
