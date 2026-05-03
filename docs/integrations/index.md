# Framework Integrations

The SDK can auto-trace calls made by external AI frameworks — OpenAI SDK, Anthropic SDK, LangChain, and CrewAI. Enable tracing with one line, then use the framework as normal. All calls are captured as OTel spans and stored locally.

## Why Integrations?

You might already have agents built with LangChain or direct OpenAI calls. Instead of rewriting them, enable auto-tracing to get:

- Full execution traces stored locally
- Token usage tracking per call
- Latency measurement
- Tool call capture
- Export to any OTel backend (Jaeger, Datadog, etc.)

## OpenAI SDK

Traces all `openai.chat.completions.create()` calls.

```python
import fastaiagent.integrations.openai

# Enable — patches the OpenAI SDK globally
fastaiagent.integrations.openai.enable()

# Use OpenAI as normal — all calls are traced
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}],
)

# Traces are stored in .fastaiagent/local.db
# View with: fastaiagent traces list
```

**What's captured per call:**
- `gen_ai.system`: `"openai"`
- `gen_ai.request.model`: model name
- `gen_ai.usage.input_tokens`: prompt tokens
- `gen_ai.usage.output_tokens`: completion tokens
- Latency, tool calls

```python
# Disable when done
fastaiagent.integrations.openai.disable()
```

Requires: `pip install fastaiagent[openai]`

## Anthropic SDK

Traces all `anthropic.messages.create()` calls.

```python
import fastaiagent.integrations.anthropic

# Enable
fastaiagent.integrations.anthropic.enable()

# Use Anthropic as normal
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)

# Disable
fastaiagent.integrations.anthropic.disable()
```

**What's captured per call:**
- `gen_ai.system`: `"anthropic"`
- `gen_ai.request.model`: model name
- `gen_ai.usage.input_tokens`: input tokens
- `gen_ai.usage.output_tokens`: output tokens

Requires: `pip install fastaiagent[anthropic]`

## LangChain

A `BaseCallbackHandler` subclass that opens spans on LangChain's LLM and
tool lifecycle hooks.

```python
import fastaiagent.integrations.langchain

# Enable
fastaiagent.integrations.langchain.enable()

# Get the callback handler for LangChain
handler = fastaiagent.integrations.langchain.get_callback_handler()

# Pass to your LangChain LLM/agent/chain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4.1")
response = llm.invoke(
    [HumanMessage(content="Hello")],
    config={"callbacks": [handler]},
)

# Disable
fastaiagent.integrations.langchain.disable()
```

**Hooks instrumented:**
- `on_llm_start` / `on_llm_end` — `langchain.llm.<model>` span
- `on_tool_start` / `on_tool_end` — `langchain.tool.<name>` span
- `on_llm_error` / `on_tool_error` — closes the matching open span on failure

Spans are stored in `.fastaiagent/local.db` and visible through
`fastaiagent traces list`.

Requires (full): `pip install "fastaiagent[langchain]" langchain langchain-openai`
(the `[langchain]` extra only pulls `langchain-core`).

## CrewAI

CrewAI is currently supported as **runtime interop**: a `Crew` can run in
the same process as fastaiagent agents (and after `enable()`) without
conflict. The `enable()` / `disable()` calls validate that `crewai` is
importable; they do not yet install instrumentation that auto-traces
CrewAI tasks.

```python
import fastaiagent.integrations.crewai

# Validate crewai is installed and mark the integration enabled
fastaiagent.integrations.crewai.enable()

# Use CrewAI as normal — runs alongside fastaiagent
from crewai import Agent, Task, Crew
# ... your CrewAI code ...

fastaiagent.integrations.crewai.disable()
```

The interop guarantee is exercised by `tests/e2e/test_gate_crewai.py`,
which runs a real one-task Crew on `gpt-4.1` and asserts that
fastaiagent agents and tracing remain functional before and after the
Crew runs.

Auto-tracing of CrewAI task/agent lifecycle events is **not yet
implemented**.

Requires: `pip install fastaiagent[crewai]`

## Manual Tracing (Any Framework)

For frameworks without a dedicated integration, use the `trace_context` context manager:

```python
from fastaiagent.trace import trace_context

with trace_context("my-custom-framework") as span:
    span.set_attribute("framework", "my-framework")
    span.set_attribute("model", "my-model")
    
    # Your framework code here
    result = my_framework.run("Hello")
    
    span.set_attribute("tokens", result.token_count)
```

This works with **any** code — not just AI frameworks. Wrap database calls, API requests, or any operation you want to trace.

## Viewing Traces

All integrations store traces in the same local SQLite database:

```bash
# List all traces (from any integration)
fastaiagent traces list

# Export a specific trace
fastaiagent traces export <trace_id>
```

```python
from fastaiagent.trace import TraceStore

store = TraceStore()
traces = store.list_traces(last_hours=1)
for t in traces:
    print(f"{t.name}  spans={t.span_count}")
```

## Exporting to External Backends

Traces from all integrations can be exported to OTel-compatible backends:

```python
from fastaiagent.trace import add_exporter
from fastaiagent.trace.export import create_otlp_exporter

# Export to Jaeger, Datadog, Grafana, etc.
add_exporter(create_otlp_exporter("http://localhost:4318/v1/traces"))

# Now all traces (from any integration) are sent to both:
# 1. Local SQLite (.fastaiagent/local.db)
# 2. Your OTel backend
```

Requires: `pip install fastaiagent[otel-export]`

## How It Works

Each integration uses monkey-patching or callback handlers to intercept framework calls:

| Integration | Mechanism | What's Patched |
|-------------|-----------|---------------|
| OpenAI | Monkey-patch | `Completions.create` |
| Anthropic | Monkey-patch | `Messages.create` |
| LangChain | Callback handler | Pass via `config={"callbacks": [handler]}` — opens/closes spans on LLM and tool lifecycle |
| CrewAI | Import validation | `enable()` validates `crewai` is importable; no instrumentation hooked yet |

**Important:**
- `enable()` must be called **before** making framework calls
- `disable()` restores the original (unpatched) functions
- Multiple integrations can be enabled simultaneously
- Enabling an already-enabled integration is a no-op

## Missing Dependency Handling

If the framework isn't installed, `enable()` raises a clear error:

```python
import fastaiagent.integrations.openai

try:
    fastaiagent.integrations.openai.enable()
except ImportError as e:
    print(e)
    # "OpenAI SDK is required. Install with: pip install fastaiagent[openai]"
```

## Installing Integrations

```bash
# Individual
pip install "fastaiagent[openai]"
pip install "fastaiagent[anthropic]"
pip install "fastaiagent[langchain]"
pip install "fastaiagent[crewai]"

# All integrations
pip install "fastaiagent[all]"
```

## Complete Example

Trace both OpenAI and LangChain calls in the same application:

```python
import fastaiagent.integrations.openai
import fastaiagent.integrations.langchain
from fastaiagent.trace import add_exporter
from fastaiagent.trace.export import create_otlp_exporter

# Enable tracing for both frameworks
fastaiagent.integrations.openai.enable()
fastaiagent.integrations.langchain.enable()

# Export to Jaeger
add_exporter(create_otlp_exporter("http://localhost:4318/v1/traces"))

# --- Your application code ---

# Direct OpenAI calls — traced
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Summarize this document"}],
)

# LangChain agent — traced
handler = fastaiagent.integrations.langchain.get_callback_handler()
result = my_langchain_agent.invoke(
    {"input": "Analyze the summary"},
    config={"callbacks": [handler]},
)

# View all traces together
# $ fastaiagent traces list
# Both OpenAI and LangChain traces appear in the same list
```

---

## Next Steps

- [Tracing](../tracing/index.md) — How traces are stored and exported
- [Replay](../replay/index.md) — Debug traced executions with fork-and-rerun
- [Agents](../agents/index.md) — Build agents with the SDK
