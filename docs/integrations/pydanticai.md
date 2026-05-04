# PydanticAI

PydanticAI 0.1+ ships its own OpenTelemetry instrumentation
(`Agent.instrument_all()`), and those spans already use GenAI
semconv. So the harness here is deliberately thin — `pa.enable()`
flips that on, then wraps `Agent.run` / `run_sync` / `run_stream`
with a thin parent OTel span tagged `fastaiagent.framework=pydanticai`
so the UI filter and analytics rollups work alongside everything else.

```bash
pip install "fastaiagent[pydanticai]"  # pydantic-ai>=0.1
```

## 1. Auto-tracing

```python
from fastaiagent.integrations import pydanticai as pa
from pydantic_ai import Agent

pa.enable()  # idempotent — calls Agent.instrument_all + wraps run/run_sync/run_stream

agent = Agent("openai:gpt-4o-mini", system_prompt="be terse")
result = agent.run_sync("What colour is the sky?")
```

| Span | Captures |
|---|---|
| `pydanticai.agent.{name}` (root) | `fastaiagent.framework=pydanticai`, `framework.version`, `gen_ai.system`, `gen_ai.request.model`, system prompt (200 chars), input + output payloads, tokens from `RunResult.usage()`, computed cost |
| `chat <model>` (PydanticAI's own) | request / response messages, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, response model parameters |
| Tool spans | PydanticAI emits these natively when an `@agent.tool_plain` is invoked |

The `name` portion of the root span comes from `agent.name` /
`agent._name` if set, otherwise the bare model id.

**Note**: PydanticAI's instrumentation occasionally goes silent when
another OTel `set_tracer_provider` call resets the global state from
underneath it (a known PydanticAI / LogFire interaction). Our wrapper
span still fires and still carries the framework + GenAI tags
regardless — that's why `test_harness_pydanticai.py::test_10_autotrace_openai`
asserts on the wrapper span attributes (which we own) rather than the
inner `chat <model>` span (which can be missing).

## 2. Eval

```python
import fastaiagent as fa
from fastaiagent.integrations import pydanticai as pa

evaluable = pa.as_evaluable(agent)
results = fa.evaluate(
    evaluable,
    dataset=[{"input": "Capital of France?", "expected": "Paris"}],
    scorers=["exact_match"],
)
```

The PydanticAI adapter is **async** — `fa.evaluate` runs cases under
`asyncio.gather`, and `Agent.run_sync` can't be invoked from inside
an already-running event loop. The eval framework auto-awaits coroutines
returned from the agent function, so the async adapter slots in without
a code change in the caller.

Output extraction prefers `RunResult.output` (≥1.0) and falls back to
`.data` for older PydanticAI releases.

## 3. Guardrails

```python
from fastaiagent.guardrail.builtins import no_pii
from fastaiagent.integrations import pydanticai as pa
from fastaiagent.integrations._registry import GuardrailBlocked

guarded = pa.with_guardrails(
    agent,
    name="support-bot",
    input_guardrails=[no_pii(position="input")],
)

try:
    result = guarded.run_sync("My SSN is 123-45-6789, summarise.")
except GuardrailBlocked as e:
    print(f"blocked: {e}")
```

Block-only semantics — see [Overview → Limitations](overview.md#limitations).
Wraps `run`, `run_sync`, and `run_stream`.

For `run_stream`: input guardrails fire before the inner async cm opens;
output guardrails are not supported on the streaming path because they'd
need to buffer the full stream (defeating the purpose). If you need an
output guardrail for a stream, fall back to `run_sync`.

## 4. Prompt registry

```python
system_prompt = pa.prompt_from_registry("support-system", agent="support-bot")
agent = Agent("openai:gpt-4o-mini", system_prompt=system_prompt)
```

Returns the raw template string. PydanticAI's `Agent(system_prompt=...)`
takes a plain string. If your template has `{{var}}` placeholders, call
`PromptRegistry().get(slug).format(**kw)` and pass the result.

## 5. Knowledge base as a tool

```python
from fastaiagent.integrations import pydanticai as pa

search_kb = pa.kb_as_tool("support-kb", top_k=5, agent="support-bot")
agent = Agent("openai:gpt-4o-mini", tools=[search_kb])
```

Returns a plain function `search_<kb_name>(query: str) -> str` with the
right `__name__` and `__doc__` so PydanticAI's tool registration
(`Agent(tools=[fn])` or `@agent.tool_plain`) picks up a sensible name
and description.

## 6. Register the agent

```python
pa.register_agent(agent, name="support-bot")
```

Writes the model, provider, system prompt (1000 chars), and registered
tools into the `external_agents` table. PydanticAI agents are
single-agent, so there is no graph topology to capture — the Local UI
shows the dependency graph (model + tools + harness layers) without a
separate workflow visualisation.

## Examples

- `examples/56_trace_pydanticai.py`
- `examples/58_guardrail_pydanticai.py`
