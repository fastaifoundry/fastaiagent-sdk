# Tracing Architecture (Internals)

This document explains how spans flow through the SDK — from creation inside `agent.run()` to local SQLite storage and platform export. It's written for contributors who need to modify the tracing layer, add new span attributes, debug missing spans, or understand the dual-sink model.

For the user-facing tracing guide (how to query traces, export to backends, disable tracing), see [docs/tracing/index.md](../tracing/index.md).

---

## Overview

Every `agent.run()` produces a tree of OTel spans. Each span records what happened (name), when (timestamps), what was involved (attributes), and whether it succeeded (status). The spans flow through two independent sinks simultaneously:

```
agent.run("hello")
    │
    ▼
OTel TracerProvider (singleton)
    │
    ├── LocalStorageProcessor ──► SQLite (.fastaiagent/local.db)
    │       (synchronous, every span, immediate)
    │
    └── BatchSpanProcessor ──► PlatformSpanExporter ──► POST /public/v1/traces/ingest
            (async, batched, only if fa.connect() was called)
```

Both sinks receive the same span data. The only difference is timing: SQLite is available the instant `agent.run()` returns; the platform may lag by a few hundred milliseconds due to batch buffering.

---

## TracerProvider Bootstrap

**File:** `fastaiagent/trace/otel.py`

The `TracerProvider` is a process-wide singleton, created on first use:

```python
def get_tracer_provider():
    global _provider
    if _provider is None:
        _provider = TracerProvider()
        _provider.add_span_processor(LocalStorageProcessor())
        otel_trace.set_tracer_provider(_provider)
    return _provider
```

At this point only `LocalStorageProcessor` is attached — traces go to SQLite only.

### When `fa.connect()` is called

**File:** `fastaiagent/client.py` (lines 126–133)

A second processor is added:

```python
exporter = PlatformSpanExporter()
processor = BatchSpanProcessor(exporter)
get_tracer_provider().add_span_processor(processor)
_connection._platform_processor = processor
```

Now every span goes to both sinks. The `_platform_processor` reference is stored so `fa.disconnect()` can call `force_flush()` and `shutdown()` on it later.

### When `add_exporter()` is called

**File:** `fastaiagent/trace/otel.py` (lines 35–39)

Any OTel-compatible `SpanExporter` (OTLP, ConsoleSpanExporter, Datadog, etc.) can be added:

```python
def add_exporter(exporter):
    get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))
```

This is how OTLP export works — `create_otlp_exporter()` returns an `OTLPSpanExporter`, and `add_exporter()` wraps it in a `BatchSpanProcessor` and attaches it to the provider. Multiple exporters can be active simultaneously.

---

## Span Creation Points

Three places in the SDK create spans during an agent run. Each wraps a `with tracer.start_as_current_span(...)` context manager. When the block exits, `on_end()` fires on all processors.

### 1. Root Agent Span

**File:** `fastaiagent/agent/agent.py` — `_arun_traced()`

Created by: `tracer.start_as_current_span(f"agent.{self.name}")`

Attributes set **before** execution:

| Attribute | Source | Payload-gated? |
|-----------|--------|----------------|
| `agent.name` | `self.name` | No |
| `agent.input` | The input string | No |
| `agent.config` | `json.dumps(self.config.model_dump())` | No |
| `agent.tools` | `json.dumps([t.to_dict() for t in self.tools])` | No |
| `agent.guardrails` | `json.dumps([g.to_dict() for g in self.guardrails])` | No |
| `agent.llm.provider` | `self.llm.provider` | No |
| `agent.llm.model` | `self.llm.model` | No |
| `agent.llm.config` | `json.dumps(self.llm.to_dict())` (api_key stripped by `to_dict`) | No |
| `agent.system_prompt` | `self._resolve_system_prompt(context)` | **Yes** |

Attributes set **after** execution:

| Attribute | Source | Payload-gated? |
|-----------|--------|----------------|
| `agent.output` | `result.output` | No |
| `agent.tokens_used` | `result.tokens_used` | No |
| `agent.latency_ms` | `result.latency_ms` | No |

The trace_id is extracted from the span context after the span is created:

```python
ctx = span.get_span_context()
result.trace_id = format(ctx.trace_id, "032x")
```

### 2. LLM Call Span

**File:** `fastaiagent/llm/client.py` — `acomplete()`

Created by: `tracer.start_as_current_span(f"llm.{self.provider}.{self.model}")`

This span wraps the provider dispatch — it fires for **every provider** (openai, anthropic, ollama, azure, bedrock, custom) because it wraps `acomplete()` at the dispatch level, not the individual `_call_openai` / `_call_anthropic` methods.

Attributes set **before** the provider call (in `acomplete()`):

| Attribute | Source | Payload-gated? |
|-----------|--------|----------------|
| `gen_ai.system` | `self.provider` | No |
| `gen_ai.request.model` | `self.model` | No |
| `gen_ai.request.temperature` | `kwargs.get("temperature", self.temperature)` | No |
| `gen_ai.request.max_tokens` | `kwargs.get("max_tokens", self.max_tokens)` | No |
| `gen_ai.request.messages` | JSON-serialized messages array | **Yes** |
| `gen_ai.request.tools` | JSON-serialized tool schemas | **Yes** |

Attributes set **after** the provider returns (in `_acomplete_with_retries()`):

| Attribute | Source | Payload-gated? |
|-----------|--------|----------------|
| `gen_ai.usage.input_tokens` | `response.usage["prompt_tokens"]` or `["input_tokens"]` | No |
| `gen_ai.usage.output_tokens` | `response.usage["completion_tokens"]` or `["output_tokens"]` | No |
| `gen_ai.response.content` | `response.content` | **Yes** |
| `gen_ai.response.tool_calls` | JSON-serialized tool calls from the response | **Yes** |
| `gen_ai.response.finish_reason` | `response.finish_reason` | No |

**Important design note for contributors:** the `integrations/openai.py` and `integrations/anthropic.py` modules also create spans, but those only fire when a user calls the bare vendor Python SDKs directly (e.g., `openai.chat.completions.create()`). The agent flow goes through `LLMClient.acomplete()` which uses raw `httpx.AsyncClient` calls and never imports the vendor SDKs — so the `acomplete()` span wrap is the one that matters for the agent flow. The integration module spans are a separate path for a separate use case.

### 3. Tool Invocation Span

**File:** `fastaiagent/agent/executor.py` — `_invoke_tool_with_span()`

Created by: `tracer.start_as_current_span(f"tool.{tool_name}")`

| Attribute | When set | Payload-gated? |
|-----------|----------|----------------|
| `tool.name` | Before execution | No |
| `tool.args` | Before execution (JSON-serialized arguments) | **Yes** |
| `tool.status` | After execution (`"ok"`, `"error"`, or `"unknown"`) | No |
| `tool.result` | After execution (JSON-serialized result) | **Yes** |
| `tool.error` | After execution (only when status is error) | No |

The `_invoke_tool_with_span()` helper is shared by both `execute_tool_loop` (non-streaming) and `stream_tool_loop` (streaming), so tool spans are emitted consistently regardless of execution mode.

---

## Span Tree Structure

A typical agent run with one tool call produces this tree:

```
agent.support-bot                              ← root span
│
├── llm.openai.gpt-4.1                        ← first LLM call (returns tool_calls)
│
├── tool.lookup_order                          ← tool execution
│
└── llm.openai.gpt-4.1                        ← second LLM call (returns final answer)
```

An agent run with no tool calls:

```
agent.support-bot
│
└── llm.openai.gpt-4.1                        ← single LLM call, returns content directly
```

An agent run that hits `max_iterations` (3 tool-calling rounds before the limit):

```
agent.support-bot
│
├── llm.openai.gpt-4.1                        ← returns tool_calls
├── tool.search                                ← tool 1
├── llm.openai.gpt-4.1                        ← returns tool_calls again
├── tool.search                                ← tool 2
├── llm.openai.gpt-4.1                        ← returns tool_calls again
├── tool.search                                ← tool 3
│
└── (MaxIterationsError raised — root span ends with ERROR status)
```

Parent-child relationships are established automatically by OTel's context propagation — a span created inside a `with` block becomes a child of the currently-active span.

---

## Sink 1: LocalStorageProcessor → SQLite

**File:** `fastaiagent/trace/storage.py`

### How `on_end()` works

When a span ends, `LocalStorageProcessor.on_end(span)` fires synchronously:

1. Extracts `trace_id` (32 hex chars) and `span_id` (16 hex chars) from the OTel span context
2. Extracts `parent_span_id` from `span.parent` (None for root spans)
3. Converts `span.attributes` dict to a JSON string via `json.dumps(dict(span.attributes), default=str)`
4. Converts OTel nanosecond timestamps to ISO 8601 strings
5. Extracts status code name ("OK", "ERROR", "UNSET")
6. INSERTs into the `spans` table

### SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT,
    start_time TEXT,
    end_time TEXT,
    status TEXT DEFAULT 'OK',
    attributes TEXT DEFAULT '{}',
    events TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans (trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans (start_time);
```

The `attributes` column holds the full attribute dict as a JSON string. Every attribute set on the span — whether structural or payload — is stored here verbatim. The payload gating decision happens at span-write time (in `_arun_traced`, `acomplete`, `_invoke_tool_with_span`), not at SQLite-write time. If an attribute was set on the span, it's in SQLite.

### How `get_trace()` reads it back

```python
rows = db.fetchall(
    "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
    (trace_id,),
)
for row in rows:
    SpanData(
        attributes=json.loads(row["attributes"]),  # JSON string → dict
        ...
    )
```

The reconstructed `SpanData.attributes` dict is exactly what was set on the original OTel span. This is what `Replay.load(trace_id)` reads, what `ForkedReplay.arerun()` uses for agent reconstruction, and what `TraceStore.get_trace()` returns.

### Default database path

`FASTAIAGENT_LOCAL_DB` env var, defaults to `.fastaiagent/local.db` relative to the working directory.

---

## Sink 2: PlatformSpanExporter → HTTP POST

**File:** `fastaiagent/trace/platform_export.py`

### How `export()` works

`BatchSpanProcessor` collects spans in a background thread and calls `export(spans)` when the batch fills up or the flush timeout fires:

1. Checks `_connection.is_connected` — if not connected, returns `SUCCESS` (silently drops)
2. Converts each OTel span to a dict (same shape as the SQLite row: trace_id, span_id, parent_span_id, name, timestamps, status, attributes dict, events list)
3. POSTs to `{target}/public/v1/traces/ingest` with the payload:

```json
{
    "project": "<project_id or project name>",
    "spans": [
        {
            "span_id": "05ce90ff1ed8ebc1",
            "trace_id": "b929b94c0b6921a0e11fafc29fbca489",
            "parent_span_id": "1f4c1a16079e982c",
            "name": "tool.lookup_order",
            "start_time": "2026-04-11T19:37:03.321685+00:00",
            "end_time": "2026-04-11T19:37:03.325100+00:00",
            "status": "OK",
            "attributes": {
                "tool.name": "lookup_order",
                "tool.args": "{\"order_id\": \"ORD-001\"}",
                "tool.status": "ok",
                "tool.result": "MacBook Pro 16-inch, shipped 2026-04-01"
            },
            "events": []
        }
    ]
}
```

Headers: `X-API-Key: {api_key}`, `Content-Type: application/json`, `User-Agent: fastaiagent-sdk/{version}`

4. Returns `SpanExportResult.SUCCESS` regardless of HTTP status — the platform is a best-effort sink, SQLite is the source of truth. Failures are logged but never propagated to the user.

### Flush and disconnect

`fa.disconnect()` calls `_platform_processor.force_flush(timeout_millis=5000)` to drain any remaining batched spans, then `shutdown()` to stop the background thread.

---

## Payload Gating

**File:** `fastaiagent/trace/span.py`

```python
def trace_payloads_enabled() -> bool:
    return os.environ.get("FASTAIAGENT_TRACE_PAYLOADS", "1") != "0"
```

Default: **on** (payloads captured). Set `FASTAIAGENT_TRACE_PAYLOADS=0` to disable.

### What's gated vs what's always captured

| Category | Always captured | Payload-gated (skipped when `=0`) |
|----------|----------------|-----------------------------------|
| Agent | name, input, output, tokens, latency, config, tools, guardrails, llm.provider, llm.model, llm.config | system_prompt |
| LLM | system, model, temperature, max_tokens, input/output tokens, finish_reason | request.messages, request.tools, response.content, response.tool_calls |
| Tool | name, status, error | args, result |

The check is called at the point where the attribute would be set, not at storage time. So the decision is baked into the span before it reaches either sink. Once a span is stored without payloads, there's no way to recover them retroactively.

### Where the check is called

| File | Method | Gated attributes |
|------|--------|-----------------|
| `agent/agent.py` | `_arun_traced()` | `agent.system_prompt` |
| `llm/client.py` | `acomplete()` | `gen_ai.request.messages`, `gen_ai.request.tools` |
| `llm/client.py` | `_acomplete_with_retries()` | `gen_ai.response.content`, `gen_ai.response.tool_calls` |
| `agent/executor.py` | `_invoke_tool_with_span()` | `tool.args`, `tool.result` |

---

## How Replay Uses the Stored Spans

**File:** `fastaiagent/trace/replay.py`

### Two loading paths — same output shape

Replays can be loaded from either local SQLite or the platform. Both paths produce the same `TraceData` / `SpanData` shape, so `fork_at()`, `rerun()`, and `compare()` work identically regardless of the source.

**`Replay.load(trace_id)` — from local SQLite:**

1. `TraceStore.get_trace(trace_id)` queries SQLite for all spans with that trace_id
2. Each row's `attributes` JSON is deserialized into `SpanData.attributes`
3. `Replay._build_steps()` converts each `SpanData` into a `ReplayStep`

**`Replay.from_platform(trace_id)` — from the platform API:**

1. `api.get(f"/public/v1/traces/{trace_id}")` fetches the trace from the platform
2. The platform returns a different schema than local SQLite — field names differ and attributes are split across `input` and `output` dicts (see [platform-api.md](platform-api.md#trace-fetch-replayfrom_platformtrace_id) for the full mapping table)
3. `from_platform()` maps each platform span to `SpanData`: `s["id"]` → `span_id`, `s["input"] + s["output"]` → merged `attributes`, `trace_id` propagated from the trace envelope
4. The resulting `TraceData` is identical in shape to what `Replay.load()` produces — downstream code sees no difference

When `ForkedReplay.arerun()` is called:

1. `_find_root_span()` locates the `agent.*` span (the one with no parent)
2. `_build_agent_dict()` reads reconstruction attributes from the root span:
   - `agent.config` → JSON.parse → `AgentConfig`
   - `agent.tools` → JSON.parse → `Tool.from_dict()` (resolves via `ToolRegistry`)
   - `agent.guardrails` → JSON.parse → `Guardrail.from_dict()`
   - `agent.llm.config` → JSON.parse → `LLMClient.from_dict()`
   - `agent.system_prompt` → used as the prompt (if captured; else empty)
   - `agent.input` → used as the default input
3. `_apply_agent_modifications()` applies `modify_prompt`/`modify_config`
4. `Agent.from_dict(agent_dict)` reconstructs the agent
5. `agent.arun(new_input)` re-executes — producing a NEW trace with its own spans

The key insight: **the quality of the replay depends entirely on the quality of the stored span attributes**. If `FASTAIAGENT_TRACE_PAYLOADS=0` was set when the trace was recorded, the system prompt won't round-trip (but all structural metadata still will, so the agent can still be reconstructed — it just gets an empty system prompt unless the user overrides via `modify_prompt()`).

---

## Common Contributor Mistakes

These are real mistakes discovered during the quality gate development. Each one cost at least an hour to debug.

### 1. Patching `acomplete()` instead of `_call_openai()` in tests

If you monkeypatch `LLMClient.acomplete` directly, you bypass the OTel span wrap added in the same method. The test runs, the agent produces output, but **no `llm.*` spans are emitted**. The correct approach is to patch the provider-specific method (`_call_openai`, `_call_anthropic`, `_call_ollama`) which sits one level below the span wrap.

**How to spot it:** if your test's `Replay.load()` shows only `agent.*` and `tool.*` spans but no `llm.*` spans, you've patched too high.

### 2. Forgetting that `integrations/openai.py` is NOT the agent flow

The `integrations/openai.py` module patches `openai.resources.chat.completions.Completions.create()` — the bare OpenAI Python SDK. `LLMClient` doesn't use the OpenAI SDK. It makes raw HTTP calls with `httpx.AsyncClient`. So the integration module's spans only fire for users calling the vendor SDK directly, never for `agent.run()`.

If you're adding a new attribute to LLM spans and you put it in `integrations/openai.py`, it won't show up on traces from the agent flow. Put it in `LLMClient.acomplete()` or `_acomplete_with_retries()`.

### 3. Not JSON-encoding complex attributes

OTel span attributes are typed: they accept `str`, `int`, `float`, `bool`, and sequences thereof. They do NOT accept dicts or arbitrary objects. If you do `span.set_attribute("agent.config", self.config.model_dump())`, you'll get a silent no-op or a runtime error depending on the OTel SDK version.

Always JSON-encode: `span.set_attribute("agent.config", json.dumps(self.config.model_dump()))`. The storage layer and replay layer expect JSON strings for complex attributes and call `json.loads()` when reading them back.

### 4. Assuming `reset()` cleans up exporters

`fastaiagent.trace.otel.reset()` shuts down the TracerProvider and sets the singleton to None. The **next** `get_tracer_provider()` call creates a fresh provider with only `LocalStorageProcessor`. Any previously-added exporters (platform, OTLP, etc.) are gone. If your test calls `reset()` and then expects platform export to still work, it won't.

This is correct behavior for test isolation but a footgun in production code. Never call `reset()` outside of tests.

---

## Files Reference

| File | What it does |
|------|-------------|
| `fastaiagent/trace/otel.py` | TracerProvider singleton, `get_tracer()`, `add_exporter()`, `reset()` |
| `fastaiagent/trace/storage.py` | `LocalStorageProcessor` (on_end → SQLite), `TraceStore` (query API), `SpanData`/`TraceData` models |
| `fastaiagent/trace/platform_export.py` | `PlatformSpanExporter` (on export → HTTP POST to platform) |
| `fastaiagent/trace/export.py` | `create_otlp_exporter()` factory |
| `fastaiagent/trace/span.py` | `set_genai_attributes()`, `set_fastai_attributes()`, `trace_payloads_enabled()` |
| `fastaiagent/trace/replay.py` | `Replay`, `ForkedReplay`, `ReplayResult`, `ComparisonResult` |
| `fastaiagent/agent/agent.py` | Root agent span creation in `_arun_traced()` |
| `fastaiagent/llm/client.py` | LLM span creation in `acomplete()` + `_acomplete_with_retries()` |
| `fastaiagent/agent/executor.py` | Tool span creation in `_invoke_tool_with_span()` |
| `fastaiagent/client.py` | `connect()` / `disconnect()` — platform exporter lifecycle |
| `fastaiagent/integrations/openai.py` | Bare OpenAI SDK span patch (NOT the agent flow) |
| `fastaiagent/integrations/anthropic.py` | Bare Anthropic SDK span patch (NOT the agent flow) |
