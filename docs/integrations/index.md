# Integrations

`fastaiagent.integrations` patches the **OpenAI and Anthropic SDKs**, so every
`chat.completions.create()` and `messages.create()` call lands in the local
trace store with tokens, cost and payloads. Use it when your code calls the
provider SDK directly.

Any other in-process **OpenTelemetry / OpenInference / OpenLLMetry**
instrumentor is captured with one opt-in call, `fastaiagent.enable_otel_capture()`.
See [Capture any OTel / OpenInference framework](../tracing/third-party-otel.md).

## Pick your starting point

- I have a **LangGraph**, **CrewAI** or **Pydantic AI** agent →
  [Use fastaiagent with your existing framework](frameworks.md). The
  framework adapters were removed in 1.96.0; that page shows the route that
  replaces them.
- I want plane-authored guardrails and scorers inside my own runtime →
  [Guardrails & evals without the runtime](primitives-without-the-runtime.md).
- I'm calling the **OpenAI / Anthropic SDK directly** → keep reading
  this page.

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

Each integration's `disable()` restores the original methods.

```python
openai_integration.disable()
anthropic_integration.disable()
```

## Trace export

Every traced call goes to the same local SQLite store, regardless of
which integration produced it. Use the standard FastAIAgent trace
export tooling — see the [Trace export](../tracing/index.md) docs.
