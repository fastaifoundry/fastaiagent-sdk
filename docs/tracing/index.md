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

# Every result includes a trace_id
print(result.trace_id)  # e.g. "b6acf1ef2c2779bbc2fcf80802ae0534"

# Use it to replay and debug later
from fastaiagent.trace import Replay
replay = Replay.load(result.trace_id)
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

### Agent Reconstruction Attributes (used by Replay)

Every `agent.run()` root span carries enough metadata for [Agent Replay](../replay/index.md) to reconstruct the agent from a stored trace and rerun it. These are always captured (structural, not payload):

| Attribute | Description |
|-----------|-------------|
| `agent.name` | Agent name |
| `agent.input` | Input passed to `agent.run()` |
| `agent.output` | Final output |
| `agent.tokens_used` | Total tokens consumed |
| `agent.latency_ms` | Wall-clock duration |
| `agent.config` | JSON-encoded `AgentConfig` (max_iterations, temperature, max_tokens, etc.) |
| `agent.tools` | JSON-encoded list of tool schemas (name, description, parameters) |
| `agent.guardrails` | JSON-encoded list of guardrails (name, position, blocking, type) |
| `agent.llm.provider` | LLM provider (`openai`, `anthropic`, ...) |
| `agent.llm.model` | Model id |
| `agent.llm.config` | JSON-encoded `LLMClient.to_dict()` (api_key stripped) |

Tool invocations emit their own `tool.{name}` span with:

| Attribute | Description |
|-----------|-------------|
| `tool.name` | Tool name |
| `tool.status` | `ok` / `error` / `unknown` |
| `tool.args` | JSON-encoded arguments (payload-gated — see below) |
| `tool.result` | JSON-encoded return value (payload-gated) |
| `tool.error` | Error string when status is `error` |

LLM calls emit `llm.{provider}.{model}` spans with standard GenAI attributes plus payload-gated `gen_ai.request.messages`, `gen_ai.request.tools`, `gen_ai.response.content`, `gen_ai.response.tool_calls`, and `gen_ai.response.finish_reason`.

### Payload Gating (`FASTAIAGENT_TRACE_PAYLOADS`)

Payload-bearing attributes — LLM messages, LLM response content, tool arguments, tool results, and resolved system prompts — can contain sensitive data. They default to **captured** so replay reconstruction works out of the box, but you can turn them off globally:

```bash
export FASTAIAGENT_TRACE_PAYLOADS=0
```

With payloads disabled:
- Structural metadata (`agent.config`, `agent.tools`, `agent.guardrails`, `agent.llm.config`, `gen_ai.system`, `gen_ai.request.model`, token counts, finish reasons, `tool.name`/`tool.status`) is still captured — traces remain useful for monitoring and performance analysis.
- Free-text payloads (messages, responses, prompts, tool args/results) are skipped.
- Replay reconstruction still works for agent config and tool schemas, but reruns lose the original resolved prompt if your code relied on span-captured prompts.

Defaults to `1` (on). Set to `0` in production environments handling PII if you do not otherwise scrub traces at the exporter layer.

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

Traces are always stored as SQLite. The database path can point to any filesystem location — local disk or a cloud-mounted volume.

### Local

```bash
export FASTAIAGENT_TRACE_DB_PATH=/data/my-project/traces.db
```

```python
from fastaiagent.trace import TraceStore
store = TraceStore(db_path="/data/my-project/traces.db")
```

### Cloud-Mounted Filesystems

Mount a cloud volume and point the trace path to it. SQLite works on any POSIX-compatible filesystem mount:

| Cloud Provider | Mount Tool | Example Path |
|----------------|-----------|--------------|
| **Azure Files** | Azure File Share (SMB/NFS) | `/mnt/azure-share/traces.db` |
| **AWS S3** | [Mountpoint for S3](https://github.com/awslabs/mountpoint-s3) or s3fs-fuse | `/mnt/s3-bucket/traces.db` |
| **AWS EFS** | NFS mount | `/mnt/efs/traces.db` |
| **GCS** | [Cloud Storage FUSE](https://cloud.google.com/storage/docs/gcs-fuse) | `/mnt/gcs-bucket/traces.db` |

```bash
# Azure Files example
export FASTAIAGENT_TRACE_DB_PATH=/mnt/azure-share/traces.db

# S3 via Mountpoint
export FASTAIAGENT_TRACE_DB_PATH=/mnt/s3-bucket/traces.db
```

```python
# Or set programmatically
store = TraceStore(db_path="/mnt/azure-share/traces.db")
```

> **Note:** SQLite requires a filesystem that supports file locking. Most cloud-mounted POSIX filesystems (Azure Files, EFS, GCS FUSE) support this. Object-storage mounts (S3 Mountpoint, s3fs-fuse) work for single-writer scenarios — avoid concurrent writes from multiple processes to the same SQLite file on these mounts.

See [Example 10](https://github.com/anthropics/fastaiagent-sdk/blob/main/examples/10_trace_query.py) for a runnable demo of trace querying with custom storage paths.

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

## Examples

- [Example 09](https://github.com/anthropics/fastaiagent-sdk/blob/main/examples/09_otel_export.py) — Export traces to OTel collectors (Jaeger, Datadog)
- [Example 10](https://github.com/anthropics/fastaiagent-sdk/blob/main/examples/10_trace_query.py) — Query, search, and export local traces with custom storage paths
