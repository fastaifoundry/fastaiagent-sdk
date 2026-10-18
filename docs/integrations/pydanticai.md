# PydanticAI

PydanticAI 0.1+ ships its own OpenTelemetry instrumentation
(`Agent.instrument_all()`), and those spans already use GenAI
semconv. So the harness here is deliberately thin — `pa.enable()`
flips that on, then wraps `Agent.run` / `run_sync` / `run_stream`
with a thin parent OTel span tagged `fastaiagent.framework=pydanticai`
so the UI filter and analytics rollups work alongside everything else.

```bash
pip install "fastaiagent[pydanticai]"  # pydantic-ai 1.105–1.106, genai-prices<0.1
```

The extra pins the 1.x line deliberately: on later 1.x releases, and with
`genai-prices` 0.1 or newer, Pydantic AI reports zero tokens and cost for every
run.

**On Pydantic AI 2.x**, install it yourself instead of the extra (Python 3.11+):

```bash
pip install fastaiagent "pydantic-ai>=2.57,<3" "genai-prices>=0.1.13"
```

Verified on 2026-10-18 against the live OpenAI and Anthropic APIs: 2.57.0 with
genai-prices 0.1.13 records real token counts and cost. The extra itself stays
on 1.x for now because Pydantic AI 2.x needs `openai>=3.26` and
`anthropic>=1.12.1`, newer than the SDK's `[openai]` and `[anthropic]` extras
allow. Moving all three together is a separate change.

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
Wraps `run`, `run_sync`, and `run_stream`. Each check writes one
`guardrail_events` row (`framework=pydanticai`) whenever tracing is on.

For `run_stream`: input guardrails fire before the inner async cm opens;
output guardrails are not supported on the streaming path because they'd
need to buffer the full stream (defeating the purpose). If you need an
output guardrail for a stream, fall back to `run_sync`.

A `run_stream` that fails, including one that fails while the stream is
opening (a 4xx on connect), ends its root span with status ERROR (*since
1.89.0*). Before, a failure on open left the span attached and never ended, so
it was never exported and the next run nested under it.

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

## Linking traces to a control-plane agent

Pass `name=` so the root span carries `agent.name`, and call `register_agent()`
so the plane has an agent with that name to match:

```python
from fastaiagent.integrations import pydanticai as integ

integ.enable()
guarded = integ.with_guardrails(obj, name="my-agent")
integ.register_agent(guarded, name="my-agent")   # local registry + plane
```

Unnamed runs emit no `agent.name` and are not linked. See
[the harness overview](overview.md#connecting-foreign-agents-to-the-control-plane).

