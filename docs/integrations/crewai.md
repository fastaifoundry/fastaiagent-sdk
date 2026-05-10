# CrewAI

The harness wraps any CrewAI `Crew` (sequential or hierarchical
process). Auto-tracing combines two interception surfaces because
CrewAI's callbacks alone don't see LLM-level events:

- **Method patches** for the structural spans — `Crew.kickoff`,
  `Agent.execute_task`, `Task.execute_sync` / `_async`.
- **Event-bus subscriptions** (`crewai.events.crewai_event_bus`) for
  LLM and tool spans — `LLMCallStartedEvent` / `LLMCallCompletedEvent`,
  `ToolUsageStartedEvent` / `ToolUsageFinishedEvent` /
  `ToolUsageErrorEvent`.

By the time the LLM event fires, the OTel current span is already the
`crewai.agent.{role}` span we opened in the structural patch, so child
spans nest under the right parent automatically.

```bash
pip install "fastaiagent[crewai]"  # crewai>=1.0
```

## 1. Auto-tracing

```python
from fastaiagent.integrations import crewai as ca
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

ca.enable()  # idempotent — patches CrewAI's Crew/Agent/Task and event bus

llm = LLM(model="openai/gpt-4o-mini", temperature=0)
researcher = Agent(role="Researcher", goal="Answer concisely.",
                   backstory="...", llm=llm, verbose=False)
task = Task(description="Capital of France?", expected_output="One word.",
            agent=researcher)
crew = Crew(agents=[researcher], tasks=[task],
            process=Process.sequential, verbose=False)

result = crew.kickoff()
```

| Span | Captures |
|---|---|
| `crewai.crew.{name}` (root) | inputs, output (`.raw`), agent/task counts, process type, `fastaiagent.framework=crewai` |
| `crewai.agent.{role}` | role, goal, backstory (200 chars), model, task description, output |
| `crewai.task.{slug}` | description, expected output, assigned agent role, output |
| `llm.{provider}.{model}` | request messages, response, token usage from CrewAI's `LLMCallCompletedEvent.usage`, computed cost |
| `tool.{tool_name}` | args, output, latency |

The provider in `llm.{provider}.{model}` is inferred from the
`provider/model` litellm-style id (`openai/gpt-4o-mini` →
`openai`); for non-prefixed strings we fall back to keyword heuristics.

## 2. Eval

```python
import fastaiagent as fa
from fastaiagent.integrations import crewai as ca

evaluable = ca.as_evaluable(crew)
results = fa.evaluate(
    evaluable,
    dataset=[{"input": "Capital of France?", "expected": "Paris"}],
    scorers=["exact_match"],
)
```

Default `input_mapper` is `lambda s: {"input": s}` — i.e. the eval-case
input is passed as `crew.kickoff(inputs={"input": text})`. `output_mapper`
reads `CrewOutput.raw`. Override either if your crew expects a
different inputs key:

```python
evaluable = ca.as_evaluable(
    crew,
    input_mapper=lambda text: {"topic": text, "depth": "deep"},
)
```

The adapter opens an outer `eval.case` OTel span so `trace_id` is
captured per case.

## 3. Guardrails

```python
from fastaiagent.guardrail.builtins import no_pii
from fastaiagent.integrations import crewai as ca

guarded = ca.with_guardrails(
    crew,
    name="research-crew",
    input_guardrails=[no_pii(position="input")],
)
result = guarded.kickoff(inputs={"input": "..."})  # raises GuardrailBlocked on PII
```

Block-only semantics — see [Overview → Limitations](overview.md#limitations).
Wraps `kickoff` and `kickoff_async`.

## 4. Prompt registry

```python
backstory = ca.prompt_from_registry("researcher-backstory", agent="research-crew")
agent = Agent(
    role="Researcher",
    goal="Find primary sources and summarise them.",
    backstory=backstory,
    llm=llm,
)
```

Returns the raw template string. CrewAI's `Agent.role` / `goal` /
`backstory` and `Task.description` all take plain strings, so we don't
need a framework-native prompt object. If the template has `{{var}}`
placeholders, call `PromptRegistry().get(slug).format(**kw)` yourself
and pass the result.

## 5. Knowledge base as a tool

```python
from fastaiagent.integrations import crewai as ca

search_tool = ca.kb_as_tool(
    "support-kb",
    top_k=5,
    description="Search the support KB.",
)
agent = Agent(
    role="Support",
    goal="Answer customer questions using the KB.",
    backstory="You are a careful support specialist.",
    llm=llm,
    tools=[search_tool],
)
```

Returns a `crewai.tools.BaseTool` subclass. `_run(query)` searches
the named LocalKB and returns a Markdown-ish string with chunk
content + similarity scores so the LLM has something useful to read.

**Tool-name normalization:** the returned tool's `name` is
`f"search_{kb_name}"`. CrewAI (via pydantic) normalizes `BaseTool.name`
by replacing hyphens with underscores, so a KB called `"support-kb"`
will produce a span named `tool.search_support_kb` — not
`tool.search_support-kb`. Match on the underscored form when asserting
against trace contents.

## 6. Register the crew

```python
ca.register_agent(crew, name="research-crew")
```

Writes the agents, tasks, process type, and edges (sequential next /
hierarchical manager-to-worker / agent-owns-task) into the
`external_agents` table. The Local UI's `/agents/research-crew` page
renders the dependency graph + a sequential or tree workflow
visualization depending on the process.

## Model notes

* **gpt-5 family** (`openai/gpt-5`, `openai/gpt-5-mini`, `openai/gpt-5-nano`)
  rejects `temperature` values other than the default (`1`). Constructing
  the LLM as `LLM(model="openai/gpt-5-mini", temperature=0)` will raise
  `BadRequestError 400 — unsupported value`. Omit the parameter and let
  the model use its default, or set `temperature=1` explicitly. The
  integration captures spans/tokens/cost identically; this is a model-
  side API constraint, not a CrewAI or SDK quirk.
* **Token + cost capture on crewai 1.9.x**: `LLMCallCompletedEvent` on
  the 1.9 line does not carry a `usage` payload. The integration patches
  litellm's `TokenCalcHandler.log_success_event` to stash usage and
  reads it back in `_on_llm_completed`. Both `gen_ai.usage.*_tokens`
  and `fastaiagent.cost.total_usd` populate on 1.9.x as a result.
* **Cost is stored as `fastaiagent.cost.total_usd`** on the LLM span
  (note the namespace prefix). Cost is computed via
  `fastaiagent.ui.pricing.compute_cost_usd` — a prefix-matched lookup
  table covering `gpt-5*`, `gpt-4o*`, `gpt-4.1*`, `o1`/`o3`/`o4-mini`,
  `claude-3-*`, `claude-sonnet-4`, `claude-haiku-4`, `claude-opus-4`,
  `gemini-1.5/2.x`, and others. Unknown models leave cost unset.

## Version compatibility

This integration targets CrewAI 1.x. The 1.0 release stabilized
`Crew.kickoff_async`, the events bus surface, and `crewai.tools.BaseTool`
— going lower would require monkey-patching private internals and
breaking on each minor CrewAI update.

The plan considered loosening to `crewai>=0.80` to match the spec, but
kept `>=1.0` because CrewAI's public test gate (`tests/e2e/test_gate_crewai.py`)
already runs against `>=1.0` and we'd rather not regress that gate.
The CHANGELOG notes this deviation.

## Examples

- `examples/55_trace_crewai.py`
