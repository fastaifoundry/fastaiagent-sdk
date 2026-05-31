# Capture any OpenTelemetry / OpenInference framework

FastAIAgent can capture and richly render spans from **any in-process
OpenTelemetry / OpenInference / OpenLLMetry instrumentor** — not just its
first-party LangChain / CrewAI / PydanticAI harness. One opt-in call,
`fastaiagent.enable_otel_capture()`, does it.

`python examples/otel-openinference/capture.py` runs a plain OpenAI call (no
FastAIAgent agent) with the OpenInference OpenAI instrumentor active and capture
turned on, then prints the canonical attributes that landed in the local trace
store. Open `fastaiagent ui` afterwards to see the span render with model,
tokens, cost, and Input/Output content.

## Setup

```bash
pip install "fastaiagent[openai]" openinference-instrumentation-openai
export OPENAI_API_KEY=...
```

## What this shows

1. A third-party instrumentor (`OpenAIInstrumentor`) emitting OpenInference-style
   spans for a call path FastAIAgent doesn't own.
2. `enable_otel_capture()` joining the active tracer provider and normalizing
   those foreign spans into the canonical `gen_ai.*` keys the Local UI reads.
3. The captured span rendering richly in the Traces list and trace detail page.

See [docs/tracing/third-party-otel.md](../../docs/tracing/third-party-otel.md)
for the full convention-mapping table and call-order notes.
