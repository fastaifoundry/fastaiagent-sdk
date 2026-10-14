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

Beyond these, **any in-process OpenTelemetry / OpenInference / OpenLLMetry
instrumentor** can be captured and rendered richly with a single opt-in call —
`fastaiagent.enable_otel_capture()`. See
[Capture any OTel / OpenInference framework](../tracing/third-party-otel.md).

## Pick your starting point

- I have an existing **LangGraph** / **CrewAI** / **PydanticAI** agent →
  start with [Universal harness — overview](overview.md), then read the
  framework-specific guide.
- I'm calling the **OpenAI / Anthropic SDK directly** → keep reading
  this page.

## Connecting to the control plane

Tracing works out of the box; **linking those traces to an agent on the
Enterprise plane needs a name**. Pass `name=` to `with_guardrails(...)` (or use
`agent_name(...)`) so the root span carries `agent.name`, and call
`register_agent()` so the plane has an agent with that name to match. See
[Connecting foreign agents to the control plane](overview.md#connecting-foreign-agents-to-the-control-plane).

## Universal harness — per framework

- [Overview & feature matrix](overview.md)
- [LangChain / LangGraph](langchain.md)
- [CrewAI](crewai.md)
- [PydanticAI](pydanticai.md)

## Borrowing SDK primitives — any runtime

The harness above traces your framework. The other half is *enforcement and
scoring*: run plane-authored guardrails and platform scorers inside a runtime
that isn't `fa.Agent`, and emit the results as standard OpenInference spans on
the exporter you already have — no second exporter, no rewrite.

See [Guardrails & evals without the runtime](primitives-without-the-runtime.md).

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

**Sync and async clients, streamed or not** (*since 1.89.0*; `AsyncOpenAI`
was not traced). With `stream=True`, the span lasts until the stream ends:
usage (pass `stream_options={"include_usage": True}`), content, tool calls and
finish reason are recorded, and an error part-way through ends the span ERROR
before it is re-raised. Before, the span closed as soon as `create` returned
the stream, with no usage or content, and a mid-stream failure never reached
the trace. Closing a stream early ends the span cleanly. The stream object you
get back is the client's own, so its helpers keep working;
`client.chat.completions.stream(...)` goes through `create` and is covered.
Exception text on these spans is scrubbed of credentials.

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
and the rest mirrors. The same streaming rules apply (*since 1.89.0*):
`AsyncAnthropic` is traced, a `stream=True` span stays open until the stream
ends and records usage from `message_start`/`message_delta`, and a mid-stream
failure ends it ERROR. `client.messages.stream(...)` (sync and async, as a
context manager) is traced too; its span opens when the `with` block enters
and follows the same contract.

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
