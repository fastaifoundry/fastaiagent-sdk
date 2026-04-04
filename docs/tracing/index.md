# Tracing

The SDK provides OTel-native (OpenTelemetry) tracing that records every LLM call, tool execution, and chain step. Traces are stored locally in SQLite by default and can be exported to any OTel-compatible backend (Jaeger, Datadog, Grafana, etc.).

## Quick Start

Tracing is automatic — every agent and chain execution creates spans:

```python
from fastaiagent import Agent, LLMClient

agent = Agent(
    name="support-bot",
    system_prompt="Be helpful.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

# This execution is automatically traced
result = agent.run("Hello")

# View traces via CLI
# $ fastaiagent traces list
```

## Manual Tracing with Context Manager

Wrap any code block in a trace span:

```python
from fastaiagent.trace import trace_context

with trace_context("my-operation") as span:
    span.set_attribute("custom.key", "value")
    # ... your code here ...
    result = do_work()
```

### Nested Spans

Spans nest automatically — inner spans become children of the outer span:

```python
with trace_context("parent-operation") as parent:
    parent.set_attribute("step", "start")

    with trace_context("step-1") as child1:
        # This span is a child of parent-operation
        do_step_1()

    with trace_context("step-2") as child2:
        do_step_2()
```

This creates a trace tree:
```
parent-operation
├── step-1
└── step-2
```

## Local Storage

All traces are stored automatically in a local SQLite database at `.fastaiagent/traces.db`. No configuration needed.

### Querying Traces

```python
from fastaiagent.trace import TraceStore

store = TraceStore()

# List recent traces
traces = store.list_traces(last_hours=24)
for t in traces:
    print(f"{t.trace_id[:12]}  {t.name}  spans={t.span_count}  {t.start_time}")

# Get a specific trace with all spans
trace = store.get_trace("abc123def456...")
print(f"Name: {trace.name}")
print(f"Status: {trace.status}")
print(f"Spans: {len(trace.spans)}")

for span in trace.spans:
    print(f"  {span.name}  {span.start_time} → {span.end_time}")
    print(f"    Attributes: {span.attributes}")

# Search traces by name or attributes
results = store.search("support-bot")

# Export as JSON
json_str = store.export("abc123def456...", format="json")
```

### TraceSummary

Returned by `list_traces()` and `search()`:

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | Unique trace identifier |
| `name` | `str` | Root span name |
| `start_time` | `str` | ISO timestamp |
| `status` | `str` | OK, ERROR, UNSET |
| `span_count` | `int` | Number of spans |
| `duration_ms` | `int` | Total duration |

### TraceData

Returned by `get_trace()`:

| Field | Type | Description |
|-------|------|-------------|
| `trace_id` | `str` | Unique trace identifier |
| `name` | `str` | Root span name |
| `start_time` | `str` | ISO timestamp |
| `end_time` | `str` | ISO timestamp |
| `status` | `str` | OK, ERROR, UNSET |
| `metadata` | `dict` | Trace-level metadata |
| `spans` | `list[SpanData]` | All spans in the trace |

### SpanData

| Field | Type | Description |
|-------|------|-------------|
| `span_id` | `str` | Unique span identifier |
| `trace_id` | `str` | Parent trace |
| `parent_span_id` | `str \| None` | Parent span (None for root) |
| `name` | `str` | Span name (e.g., "llm.chat_completion") |
| `start_time` | `str` | ISO timestamp |
| `end_time` | `str` | ISO timestamp |
| `status` | `str` | OK, ERROR, UNSET |
| `attributes` | `dict` | Key-value metadata |
| `events` | `list[dict]` | Span events |

## GenAI Semantic Conventions

The SDK follows the OpenTelemetry GenAI semantic conventions for LLM-related attributes:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `gen_ai.system` | LLM provider | `"openai"`, `"anthropic"` |
| `gen_ai.request.model` | Model name | `"gpt-4.1"` |
| `gen_ai.request.temperature` | Temperature | `0.7` |
| `gen_ai.request.max_tokens` | Max tokens | `1000` |
| `gen_ai.usage.input_tokens` | Prompt tokens | `150` |
| `gen_ai.usage.output_tokens` | Completion tokens | `45` |
| `gen_ai.response.finish_reasons` | Stop reasons | `["stop"]` |

### FastAIAgent Custom Attributes

| Attribute | Description |
|-----------|-------------|
| `fastai.agent.name` | Agent name |
| `fastai.chain.name` | Chain name |
| `fastai.chain.node_id` | Current node in chain |
| `fastai.chain.iteration` | Cycle iteration count |
| `fastai.tool.name` | Tool being executed |
| `fastai.checkpoint.id` | Checkpoint ID |
| `fastai.guardrail.name` | Guardrail name |
| `fastai.guardrail.passed` | Whether guardrail passed |
| `fastai.cost.total_usd` | Accumulated cost |

### Setting Attributes Programmatically

```python
from fastaiagent.trace.span import set_genai_attributes, set_fastai_attributes

with trace_context("my-llm-call") as span:
    set_genai_attributes(
        span,
        system="openai",
        model="gpt-4.1",
        input_tokens=150,
        output_tokens=45,
    )
    set_fastai_attributes(
        span,
        **{"agent.name": "support-bot", "cost.total_usd": 0.003},
    )
```

## Exporting to External Backends

### OTLP (Jaeger, Grafana, Datadog)

```python
from fastaiagent.trace import add_exporter
from fastaiagent.trace.export import create_otlp_exporter

# HTTP exporter (most common)
exporter = create_otlp_exporter(
    endpoint="http://localhost:4318/v1/traces",
    headers={"Authorization": "Bearer my-token"},
)
add_exporter(exporter)

# gRPC exporter
exporter = create_otlp_exporter(
    endpoint="http://localhost:4317",
    protocol="grpc",
)
add_exporter(exporter)
```

Requires: `pip install fastaiagent[otel-export]`

### Any OTel SpanExporter

```python
from fastaiagent.trace import add_exporter

# Use any OTel-compatible exporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
add_exporter(ConsoleSpanExporter())
```

Traces are always stored locally AND sent to exporters — adding an exporter doesn't replace local storage.

## Custom Storage Path

```python
from fastaiagent._internal.config import SDKConfig

# Via environment variable
# export FASTAIAGENT_TRACE_DB_PATH=/custom/path/traces.db

# Or via config
from fastaiagent.trace.storage import TraceStore
store = TraceStore(db_path="/custom/path/traces.db")
```

## CLI Commands

```bash
# List recent traces
fastaiagent traces list
fastaiagent traces list --last-hours 1

# Export a trace as JSON
fastaiagent traces export <trace_id>
fastaiagent traces export abc123def456 --format json
```

## Disabling Tracing

```bash
export FASTAIAGENT_TRACE_ENABLED=false
```

Or pass `trace=False` to agent/chain execution:

```python
result = agent.run("Hello", trace=False)
```

## Resetting the Tracer

For testing or reconfiguration:

```python
from fastaiagent.trace import reset

reset()  # Shuts down existing provider, clears singleton
# Next trace operation creates a fresh provider
```

## Architecture

```
Your Code
    │
    ▼
OTel TracerProvider (singleton)
    │
    ├── LocalStorageProcessor → SQLite (.fastaiagent/traces.db)
    │
    ├── BatchSpanProcessor → OTLP Exporter (Jaeger, Datadog, etc.)
    │
    └── BatchSpanProcessor → Any additional exporters
```

- **LocalStorageProcessor** writes every span to SQLite as it completes
- **BatchSpanProcessor** batches spans for efficient export to remote backends
- Multiple exporters can run simultaneously
- The TracerProvider is a singleton — initialized on first use, reused globally

---

## Platform Export

When connected to the FastAIAgent Platform, traces are automatically sent to the platform dashboard alongside local SQLite storage. No code changes needed.

```python
import fastaiagent as fa

fa.connect(api_key="fa-...", project="my-project")

# Every agent.run() now sends traces to both local SQLite and platform
result = agent.run("Help me")
# View in platform dashboard: execution traces, token costs, latency
```

**Manual backfill** — publish existing local traces to the platform:

```python
trace_store = TraceStore()
traces = trace_store.list_traces(limit=100)
for t_summary in traces:
    trace_data = trace_store.get_trace(t_summary.trace_id)
    trace_data.publish()  # sends to platform
```

If the platform is unreachable, traces are safe in local SQLite. No operation fails because the platform is down.

---

## Next Steps

- [Replay](../replay/index.md) — Debug agent execution with fork-and-rerun
- [Integrations](../integrations/index.md) — Auto-trace OpenAI, Anthropic, LangChain, CrewAI
- [Agents](../agents/index.md) — Build agents with automatic tracing
