# Use fastaiagent with your existing framework

Your agent is built in LangGraph, CrewAI or Pydantic AI, and you are not going
to rewrite it as a `fa.Agent`. You can still use tracing, the Local UI,
guardrails, evals, knowledge bases, prompts and the control plane with it. This
page shows how. Every snippet runs as written against the framework's own
public API. None of it imports framework code from `fastaiagent`.

!!! note "Removed in 1.96.0"
    The `fastaiagent.integrations.langchain`, `.crewai` and `.pydanticai`
    adapters and the `[langchain]`, `[crewai]` and `[pydanticai]` extras are
    gone. They wrapped each framework's internals, so they broke whenever a
    framework changed. The OpenTelemetry route below needs no adapter. The
    [last section](#moving-from-the-removed-integrations) maps each removed
    function to its replacement.

| You want | How |
|---|---|
| Traces in the Local UI | An OpenInference instrumentor + `fa.enable_otel_capture()` |
| A named run (agent name, prompt link, plane linkage) | A root span from `fa.trace_context(...)` |
| Guardrails | `guardrail.execute(...)` around the call |
| Evals, each case linked to its trace | `fa.evaluate(...)` on a plain function |
| Knowledge-base search | `kb.search(...)` inside one of the framework's own tools |
| Registry prompts | `PromptRegistry().get(...).format(...)` as the system prompt |

## 1. Trace the run

Install the framework's OpenInference instrumentor, instrument, and call
`fa.enable_otel_capture()`. The spans land in `.fastaiagent/local.db` and render
in `fastaiagent ui` with model, tokens, cost, input and output.
[Capture any OTel / OpenInference framework](../tracing/third-party-otel.md)
explains what the capture does.

=== "LangGraph"

    ```bash
    pip install fastaiagent langgraph langchain-openai openinference-instrumentation-langchain
    ```

    ```python
    import fastaiagent as fa
    from openinference.instrumentation.langchain import LangChainInstrumentor

    fa.enable_otel_capture()
    LangChainInstrumentor().instrument()

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    agent = create_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[], system_prompt="Be brief.")
    agent.invoke({"messages": [("user", "Where is order A100?")]})
    ```

=== "CrewAI"

    ```bash
    pip install fastaiagent crewai openinference-instrumentation-crewai openinference-instrumentation-openai
    ```

    ```python
    import fastaiagent as fa
    from openinference.instrumentation.crewai import CrewAIInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    fa.enable_otel_capture()
    CrewAIInstrumentor().instrument()
    OpenAIInstrumentor().instrument()   # the model calls: tokens and cost

    from crewai import LLM, Agent, Crew, Task

    agent = Agent(role="Support", goal="Answer briefly", backstory="You help customers.",
                  llm=LLM(model="gpt-4o-mini"))
    task = Task(description="Where is order A100?", expected_output="A short answer", agent=agent)
    Crew(agents=[agent], tasks=[task]).kickoff()
    ```

=== "Pydantic AI"

    ```bash
    pip install fastaiagent pydantic-ai openinference-instrumentation-pydantic-ai
    ```

    Pydantic AI writes its own spans. Add OpenInference's span processor to a
    provider you own **before** `fa.enable_otel_capture()`, so the spans carry
    the input, output and span kinds the UI reads:

    ```python
    from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    import fastaiagent as fa

    provider = TracerProvider()
    provider.add_span_processor(OpenInferenceSpanProcessor())
    trace.set_tracer_provider(provider)
    fa.enable_otel_capture()   # joins your provider

    from pydantic_ai import Agent

    Agent.instrument_all(True)
    agent = Agent("openai:gpt-4o-mini", system_prompt="Be brief.")
    agent.run_sync("Where is order A100?")
    ```

    Without the processor, the run is still captured, but its spans have no
    input, output or span kind in the UI.

## 2. Name the run

The framework's root span says nothing about *which* agent ran
(`LangGraph`, `Crew_<uuid>.kickoff`, `agent run`). Open your own root span with
`fa.trace_context(...)` and put the run under it. `agent.name` gives the run its
name in the UI and is what the plane matches to an agent. `fastaiagent.prompt.name`
puts the run on that prompt's page.

```python
import fastaiagent as fa

with fa.trace_context(
    "agent.support",
    **{"agent.name": "support", "input.value": question},
) as span:
    answer = run_my_agent(question)          # your framework call
    span.set_attribute("output.value", answer)
```

The SDK does not invent a name. An unnamed run still traces, but it is not
linked to an agent.

## 3. Guardrails

Run the check yourself around the framework call. `execute` returns a verdict
and never raises for a failed check; you decide what a block means. Inside the
root span from step 2, it also writes a row to the **Guardrail events** page,
linked to the run's trace.

```python
import fastaiagent as fa

no_pii = fa.no_pii(position="input")


def guarded(question: str) -> str:
    with fa.trace_context("agent.support", **{"agent.name": "support", "input.value": question}):
        verdict = no_pii.execute(question)
        if verdict.action_taken == "blocked":
            return f"Blocked: {verdict.message}"
        if verdict.action_taken == "masked":
            question = verdict.modified_data
        return run_my_agent(question)
```

`action_taken` is `none` for a pass, otherwise `blocked`, `warned` or `masked`
(with the safe text in `modified_data`), following the rule's `action`. In async
code, `await no_pii.aexecute(question)`.

Guardrails authored on the plane come from
`fa.plane_guardrails_for_agent("support")`. For the plane to record a verdict,
emit it as a `GUARDRAIL` span with `fa.emit_guardrail(...)`.
[Guardrails & evals without the runtime](primitives-without-the-runtime.md)
covers both, and the `on_error` policy.

## 4. Evals

`fa.evaluate` takes any function from input to output:

```python
import fastaiagent as fa

results = fa.evaluate(
    agent_fn=run_my_agent,
    dataset=[
        {"input": "Where is order A100?", "expected_output": "shipped"},
        {"input": "Where is order B200?", "expected_output": "processing"},
    ],
    scorers=["contains"],
    run_name="support-nightly",
    agent_name="support",
)
print(results.summary())
```

`fa.evaluate` runs four cases at once (`concurrency=4`), each sync call on its
own thread. Two frameworks need a different function for that:

- **CrewAI**: one `Agent` cannot run twice at the same time ("Executor is
  already running"). Build the agent and the crew inside the function.
- **Pydantic AI**: pass an async function, so every case runs on one event
  loop. `run_sync` from several threads can hang once the model client has
  been used on another loop.

```python
async def run_my_agent(question: str) -> str:   # Pydantic AI
    return (await agent.run(question)).output
```

Since 1.96.0, each case links to the trace its call started, so the Evals page
opens the run behind a score. A function that returns an object with a
`trace_id` sets the link itself. The same function works in
[Agent CI](../evaluation/agent-ci.md) gates.

## 5. Knowledge bases

Search a `LocalKB` inside one of the framework's own tools:

```python
import fastaiagent as fa

kb = fa.LocalKB(name="policy", search_type="keyword")
kb.add("Customers may return items within 45 days of delivery.")


def search_policy(query: str) -> str:
    """Search the company policy knowledge base."""
    return "\n".join(hit.chunk.content for hit in kb.search(query, top_k=3))
```

Wrap `search_policy` with the framework's tool decorator (`@tool` from
`langchain_core.tools`, `@tool("search_policy")` from `crewai.tools`, or
`Tool(search_policy)` in Pydantic AI). With a root span from step 2, the
retrieval span lands in the same trace as the run.

## 6. Prompts

Use a registry prompt as the system prompt, and stamp its name on the root
span so the run shows on the prompt's page:

```python
import fastaiagent as fa

prompt = fa.PromptRegistry().get("support-system")
system_prompt = prompt.format(company="Northwind")

with fa.trace_context(
    "agent.support",
    **{"agent.name": "support", "fastaiagent.prompt.name": prompt.name},
):
    ...  # build the framework agent with system_prompt and run it
```

## 7. The control plane

`fa.connect(...)` exports these traces like any other. Two things link them to
an agent on the plane:

1. The root span from step 2, carrying `agent.name`.
2. An agent with that name on the plane. Push one that describes the agent:

```python
import fastaiagent as fa

fa.connect(api_key="fa_k_…", target="https://app.fastaiagent.net")
fa.Agent(
    name="support",
    system_prompt="You are a support agent.",
    llm=fa.LLMClient(provider="openai", model="gpt-4o-mini"),
).push()
```

The key needs the `agent:write` scope. See
[Registration](../platform/registration.md).

!!! warning "Pydantic AI on a provider you own"
    Spans recorded on your own `TracerProvider` (the Pydantic AI setup in step 1)
    are sent to the plane only when a fastaiagent span ends. Put the run inside
    the `fa.trace_context(...)` root span from step 2 and they go out with it.

## What is different from a native agent

- **Replay** rebuilds only a `fa.Agent` run. A framework run can be viewed and
  compared, not forked and rerun.
- **CrewAI** root spans are named `Crew_<uuid>.kickoff` and have no input, and
  the instrumentor writes some LLM spans with zero tokens. The root span from
  step 2 gives the run a name and an input.
- **Middleware, `@idempotent`, checkpoints and `interrupt()`** belong to the
  fastaiagent runtime. Use the framework's own equivalents.

## Moving from the removed integrations

| Removed | Use instead |
|---|---|
| `langchain.enable()`, `crewai.enable()`, `pydanticai.enable()` | The instrumentor + `fa.enable_otel_capture()` (step 1) |
| `with_guardrails(agent, ...)` | `guardrail.execute(...)` around the call (step 3) |
| `as_evaluable(agent)` | A plain function passed to `fa.evaluate` (step 4) |
| `kb_as_retriever(kb)`, `kb_as_tool(kb)` | `kb.search` inside a framework tool (step 5) |
| `prompt_from_registry(name)` | `PromptRegistry().get(name).format(...)` (step 6) |
| `register_agent(agent, name=...)` | `fa.Agent(name=...).push()` while connected (step 7) |
| `fastaiagent.integrations.agent_name(name)` | `fa.trace_context("agent.<name>", **{"agent.name": name})` (step 2) |
| `pip install "fastaiagent[langchain]"` (and `[crewai]`, `[pydanticai]`) | `pip install fastaiagent` plus the framework and its instrumentor |

The OpenAI and Anthropic SDK auto-tracing in
[`fastaiagent.integrations`](index.md) is unchanged.
