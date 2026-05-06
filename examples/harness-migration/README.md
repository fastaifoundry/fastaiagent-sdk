# Universal Harness Migration

Wrap an existing **LangChain / LangGraph**, **CrewAI**, or **PydanticAI** agent with the FastAIAgent harness and get tracing, guardrails, prompt registry, KB-as-retriever, and `fa.evaluate()` — without rewriting the agent.

This template is the canonical demo of `fastaiagent.integrations.langchain` / `crewai` / `pydanticai`. The same `LocalKB`, the same registered `Prompt`, and the same `Guardrail` set drive three structurally-different agents — the only thing that differs across the three sub-examples is the tiny adapter call.

```
                  ┌──────────────────────┐
                  │ shared FastAIAgent   │
                  │   • LocalKB          │
                  │   • PromptRegistry   │
                  │   • Guardrails       │
                  └──────────┬───────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌──────────┐         ┌──────────┐         ┌─────────────┐
   │LangGraph │         │ CrewAI   │         │ PydanticAI  │
   │ create_  │         │  Crew    │         │   Agent     │
   │ agent()  │         │  +Task   │         │             │
   └─────┬────┘         └────┬─────┘         └──────┬──────┘
         │  lc_int.with_*    │ ca_int.with_*        │ pa_int.with_*
         ▼                   ▼                      ▼
       traces in /traces · guardrail events · evals in /evals · agents in /agents
```

**What this template demonstrates** (vs. native fa.Agent templates):

- **Auto-tracing every framework's calls** via `<integration>.enable()` — no per-call instrumentation.
- **Sharing a single KB across three frameworks** — `kb_as_retriever()` (LangChain) and `kb_as_tool()` (CrewAI / PydanticAI) read from the *same* on-disk `support-kb`.
- **Sharing a single registered prompt across three frameworks** — `register_support_prompt()` writes once; every framework reads the latest version.
- **Sharing a single guardrail set across three frameworks** — `with_guardrails(agent, input_guardrails=[...], output_guardrails=[...])` accepts the same `fa.Guardrail` type in all three.
- **`fa.evaluate()` against any of the three** — `as_evaluable(agent)` wraps each framework's native interface into the callable `fa.evaluate` expects.
- **External-agent registry** — `register_agent(agent, name="...")` makes the LangGraph / CrewAI / PydanticAI agent appear in the Local UI's `/agents` page with a framework badge alongside native fa.Agents.

---

## Quick Start

```bash
# from the SDK root
pip install -e .
cd examples/harness-migration
cp .env.example .env             # only OPENAI_API_KEY is required
pip install -r requirements.txt

# Install whichever framework(s) you want to wrap. Each is optional;
# the example sub-files only import their own framework.
pip install langchain langchain-openai langgraph     # for langchain_example.py
pip install crewai                                    # for crewai_example.py
pip install pydantic-ai                               # for pydanticai_example.py

# Run any of the three sub-examples — they all answer the same question
# against the same shared KB + prompt + guardrails.
python langchain_example.py
python crewai_example.py
python pydanticai_example.py

# Evaluate all three side-by-side on the same dataset
python eval_suite.py
python eval_suite.py --framework langchain     # only one
```

---

## Files

```
harness-migration/
├── README.md
├── .env.example
├── requirements.txt
├── shared/
│   ├── __init__.py
│   ├── kb.py            # the LocalKB every framework consumes
│   ├── prompts.py       # the registered Prompt every framework consumes
│   └── guardrails.py    # the Guardrail set every framework consumes
├── knowledge/
│   └── faq.md           # the KB content
├── langchain_example.py    # LangGraph create_agent() + lc_int.*
├── crewai_example.py       # CrewAI Crew + ca_int.*
├── pydanticai_example.py   # PydanticAI Agent + pa_int.*
├── eval_suite.py           # fa.evaluate against all three via as_evaluable
└── tests/
    └── test_smoke.py    # 10 offline regression tests
```

---

## How each integration is wired

### LangChain / LangGraph ([langchain_example.py](langchain_example.py))

```python
from fastaiagent.integrations import langchain as lc_int
from langchain.agents import create_agent

lc_int.enable()                                                    # auto-trace
system_prompt = register_support_prompt()                          # str
retriever = lc_int.kb_as_retriever("support-kb", top_k=3)          # BaseRetriever
kb_tool = Tool(name="search_knowledge_base", func=lambda q: ..., ...)
graph = create_agent(llm, tools=[kb_tool], system_prompt=system_prompt)
guarded = lc_int.with_guardrails(                                  # proxies .invoke etc.
    graph,
    name="lc-support-bot",
    input_guardrails=[...], output_guardrails=[...],
)
lc_int.register_agent(guarded, name="lc-support-bot")              # /agents listing
```

### CrewAI ([crewai_example.py](crewai_example.py))

```python
from fastaiagent.integrations import crewai as ca_int
from crewai import Agent, Crew, Process, Task

ca_int.enable()
system_prompt = register_support_prompt()                          # str
kb_tool = ca_int.kb_as_tool("support-kb", top_k=3, ...)            # BaseTool
agent = CrewAgent(role="...", goal="...", backstory=system_prompt, tools=[kb_tool], ...)
crew = Crew(agents=[agent], tasks=[Task(...)], process=Process.sequential)
guarded = ca_int.with_guardrails(crew, name="ca-support-bot", ...) # proxies .kickoff
ca_int.register_agent(guarded, name="ca-support-bot")
```

### PydanticAI ([pydanticai_example.py](pydanticai_example.py))

```python
from fastaiagent.integrations import pydanticai as pa_int
from pydantic_ai import Agent

pa_int.enable()
system_prompt = register_support_prompt()                          # str
kb_search = pa_int.kb_as_tool("support-kb", top_k=3)               # callable[(str), str]
agent = Agent("openai:gpt-4o-mini", system_prompt=system_prompt)

@agent.tool_plain
def search_knowledge_base(query: str) -> str:
    """Use this for FAQ / policy / billing / SSO / data export questions."""
    return kb_search(query)

guarded = pa_int.with_guardrails(agent, name="pa-support-bot", ...)
pa_int.register_agent(guarded, name="pa-support-bot")
```

---

## Cross-framework `fa.evaluate()` ([eval_suite.py](eval_suite.py))

```python
results = fa.evaluate(
    lc_int.as_evaluable(graph),
    dataset=EVAL_DATASET,
    scorers=["contains"],
    run_name="harness-migration-langchain",
    dataset_name="harness-migration-faq",
    agent_name="langchain-support-bot",
)
```

The same `EVAL_DATASET` runs against `lc_int.as_evaluable(graph)`, `ca_int.as_evaluable(crew)`, and `pa_int.as_evaluable(agent)`. Each adapter handles its framework's native run-shape (sync vs async, message-list vs string, etc.) and returns an `_EvaluableResult` with `.output: str` and `.trace_id: str` so per-case `EvalCaseRecord` rows can deep-link back to the trace in the Local UI.

`pa_int.as_evaluable` returns an **async** callable — `fa.evaluate()` awaits it. The other two return sync callables. You don't need to know which is which when you write the eval; `fa.evaluate()` figures it out via `asyncio.iscoroutine(...)`.

---

## Local UI

```bash
fastaiagent ui start             # http://127.0.0.1:7842
```

What this template populates:

- **`/traces`** — every `kickoff()`/`invoke()`/`run()` lands as a trace under the framework's name. The framework badge ("langchain", "crewai", "pydanticai") is on each row so you can filter.
- **`/agents`** — `lc-support-bot`, `ca-support-bot`, `pa-support-bot` listed alongside any native fa.Agents, courtesy of `register_agent()` writing to the `external_agents` table.
- **`/playground/support-prompt`** — edit the registered prompt; the next call from any of the three frameworks picks up the new version with no restart.
- **`/evals`** — three runs persisted as `harness-migration-langchain`, `harness-migration-crewai`, `harness-migration-pydanticai` against the same `harness-migration-faq` dataset. Click in for per-case `trace_id`-linked drill-down.
- **`/guardrails`** — events from input + output guardrail firings, identical shape regardless of the originating framework.

---

## Gotchas worth knowing

### KB path coupling — must be the SDK default

The integrations' `kb_as_retriever()` / `kb_as_tool()` re-instantiate `LocalKB(name=kb_name)` *with the default on-disk path* — they do not accept a `path=` kwarg. So the shared KB you create here MUST live at the default path (`~/.fastaiagent/kb/support-kb/`) or the integration will see an empty store.

Concretely: in `shared/kb.py` we do **not** pass `path=`. If you change that to `path="./.fastaiagent-kb"` to keep the index local to your project, the example's KB lookups will return zero documents.

### `pa_int.as_evaluable` returns an async callable

LangChain and CrewAI's `as_evaluable` returns sync; PydanticAI's returns async. `fa.evaluate()` handles both via `asyncio.iscoroutine` — but if you're calling the evaluable directly (without `fa.evaluate`), make sure you `await` the PydanticAI one.

### Framework deps are heavy

Each framework is several hundred MB of transitive dependencies. The example deliberately puts all three under "optional" in `requirements.txt` — install only the one(s) you actually need.

### CrewAI's own tracing

CrewAI ships its own tracing layer that conflicts with FastAIAgent's OTel setup ("Overriding of current TracerProvider is not allowed"). Set `CREWAI_DISABLE_TRACING=true` in your env to stop CrewAI from registering its tracer, or accept the warning — FastAIAgent's traces still land correctly.

### Concurrent runs of the same framework

`prompt_from_registry()` tracks lineage via a thread-local stack; spawning many concurrent `lc.invoke` / `crew.kickoff` from the same process can race the lineage attribution but does NOT corrupt the agent's output. If you see noisy lineage in `/agents`, run one framework at a time per process.

---

## Customising

**Migrate from one framework to fastaiagent gradually** — wrap the existing agent first (this template), confirm tracing + evals + guardrails behave the way you expect, then port piece by piece to native `fa.Agent`. The harness lets you take credit for the SDK's observability and prompt registry while the rewrite happens in the background.

**Add your own framework** — the integration interface is small (~6 functions). The SDK ships these three; if you want to add e.g. AutoGen, copy `fastaiagent/integrations/crewai.py` as a template and implement the same shape. Open a PR.

**Plug additional shared assets** — if you maintain shared `MCPTool` or `RESTTool` instances, you can register them once and reference them from each framework's tool list. The integrations re-resolve by name each time, so the source of truth stays in fastaiagent.

---

## What this example does NOT demonstrate

- **HITL** — the integrations don't yet expose `interrupt()` parity; if your wrapped agent needs HITL, port it to native `fa.Agent` first. See `examples/customer-support-agent/` for the HITL pattern.
- **Multi-agent within a single framework** — each sub-example is a single agent. CrewAI's multi-agent crews work, but I'm only demoing one role per example to keep the comparison apples-to-apples.
- **Fine-grained tool guardrails** — `with_guardrails()` runs at the input + output boundary; tool-call guardrails (which fa.Agent supports natively via `position=tool_call`) aren't exposed by the integrations.

---

## License

Apache 2.0 — same as the SDK.
