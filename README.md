# FastAIAgent SDK

**The open-source Python SDK to build, debug, evaluate, and ship production AI agents — in code you own.**

- 🔁 **Agent Replay** — fork a failing production trace, fix the prompt or tool, rerun, and save it as a regression test. *No other SDK can do this.*
- 🛡️ **Responsible AI Trust Layer** — groundedness, reflection, secrets, PII, toxicity, and topic controls as runtime guardrails — the things an enterprise review actually asks for, in one call.
- 🖥️ **Zero-ceremony Local UI** — ships inside the wheel. Span-tree traces, evals, prompt playground, cost, and workflow topology. No Docker, no signup, nothing phones home.
- ⏸️ **Durable & resumable** — crash-proof agents that pause for human approval *for days* and resume after a real `SIGKILL`.
- 🔌 **Universal & model-agnostic** — 20+ LLM providers; one-line instrumentation for LangGraph, CrewAI, PydanticAI, and any OpenTelemetry stack.

```bash
pip install fastaiagent
```

Runs fully standalone, or connect to the [FastAIAgent Platform](https://fastaiagent.net) for hosted observability, prompt management, and team collaboration.

[![PyPI](https://img.shields.io/pypi/v/fastaiagent?v=1.38.0)](https://pypi.org/project/fastaiagent/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://github.com/fastaifoundry/fastaiagent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fastaifoundry/fastaiagent-sdk/actions)
[![Python](https://img.shields.io/pypi/pyversions/fastaiagent)](https://pypi.org/project/fastaiagent/)

---

## Quickstart

```python
from fastaiagent import Agent, LLMClient

# Create an LLM client
llm = LLMClient(provider="openai", model="gpt-4o")

# Create an agent
agent = Agent(
    name="my-agent",
    system_prompt="You are a helpful assistant.",
    llm=llm,
)

# Run it
result = agent.run("What is the capital of France?")
print(result.output)
print(result.trace_id)  # every run is traced — use this ID for replay/debugging
```

## Providers

`LLMClient` ships with first-class presets for OpenAI, Anthropic, Azure,
Bedrock, Ollama — plus Gemini, Groq, OpenRouter, DeepSeek, Together,
Fireworks, Perplexity, Mistral, LM Studio, vLLM, SambaNova, and Cerebras.
Each preset resolves the right `base_url` and reads the API key from the
canonical environment variable, so this is the entire configuration:

```python
LLMClient(provider="groq",   model="llama-3.1-70b-versatile")  # GROQ_API_KEY
LLMClient(provider="gemini", model="gemini-2.0-flash")          # GEMINI_API_KEY
LLMClient(provider="openrouter", model="openai/gpt-4o-mini")    # OPENROUTER_API_KEY
```

Custom internal LLM gateways register in five lines via
`fastaiagent.llm.providers.register_provider`.
See [docs/llm/providers](docs/llm/providers.md).

## Testing your agents — deterministic, no network

Swap `LLMClient` for `TestModel` or `FunctionModel` (in
`fastaiagent.testing`) and your tests run offline with no flake:

```python
from fastaiagent.testing import TestModel
from fastaiagent.eval import case

@case(input="hello", expected="hi")
def test_greet(evaluate_one):
    agent = Agent(name="g", llm=TestModel(response="hi"))
    evaluate_one(agent.run, scorers=["exact_match"])
```

The pytest plugin auto-persists each tagged case to the local UI's
`/evals` page. See [docs/testing](docs/testing/index.md) and
[docs/evaluation/pytest](docs/evaluation/pytest.md).

### Build a Deep Research Agent in one file

Long-horizon research agents (the now-canonical *Scope → parallel Research → Write* pattern popularized by Open Deep Research) are a flagship template:

```sh
cd examples/deep-research-agent
pip install -r requirements.txt
export TAVILY_API_KEY=...   # optional — falls back to a local mock corpus
python agent.py --topic "Current state of MCP server adoption"
```

Sub-researchers run in parallel via `asyncio.gather`, each in its own context window; the writer composes one coherent Markdown report. Plan and findings are persisted as structured spans under `fastaiagent.research.*` for replay / inspection. See [`examples/deep-research-agent/`](examples/deep-research-agent/) and the [Templates docs](https://docs.fastaiagent.net/flagships/deep-research-agent/).

### Turn every production failure into a regression test

When a customer reports the agent did something wrong, capture the failing trace, fork it, swap the broken tool or prompt, replay, and save the fixed output as a regression case — all in five small scripts:

```sh
cd examples/regression-from-trace
pip install -r requirements.txt
zsh -lc 'python capture.py && python analyze.py && python fix.py && python save_test.py && python verify.py'
```

The template ships with a deliberately broken `lookup_order` tool whose silent failure mode (returns a fallback record stamped with the requested ID) is exactly the class of bug only the trace-replay loop can catch. After `fix.py` swaps in the fixed tool and `save_test.py` appends to `regression_dataset.jsonl`, `verify.py` runs `evaluate(...)` with `LLMJudge` and the same case stays caught forever. See [`examples/regression-from-trace/`](examples/regression-from-trace/) and the [Templates docs](https://docs.fastaiagent.net/flagships/regression-from-trace/) — includes before/after browser screenshots.

## Multimodal — images and PDFs as first-class inputs

```python
from pathlib import Path
from fastaiagent import Agent, LLMClient, Image, PDF

agent = Agent(name="claims", llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6"))

# Drop the photo and policy alongside this script to run the example.
if Path("damage.jpg").exists() and Path("policy.pdf").exists():
    result = agent.run([
        "Compare the photo to the policy and assess the claim.",
        Image.from_file("damage.jpg"),
        PDF.from_file("policy.pdf"),
    ])
    print(result.output)
```

The same code works against OpenAI, Azure, Anthropic, Bedrock, and Ollama —
provider-specific wire formatting (and OpenAI's tool-message workaround) is
handled inside `LLMClient`. See [docs/multimodal/](docs/multimodal/index.md).

## Debug a failing agent in 30 seconds

```python
from fastaiagent.trace import Replay

# Load a trace from a production failure
replay = Replay.load("trace_abc123")

# Step through to find the problem
replay.step_through()
# Step 3: LLM hallucinated the refund policy ← found it

# Fork at the failing step, fix, rerun
forked = replay.fork_at(step=3)
forked.modify_prompt("Always cite the exact policy section...")
result = forked.rerun()

# Save the corrected behavior as a regression case — every production
# failure becomes a permanent test that future eval runs will catch.
result.save_as_test(
    "regression_tests.jsonl",
    input="What is our refund policy?",
    expected_output=str(result.new_output),
    source_trace_id="trace_abc123",
)
```

**No other SDK can do this.**

## Pause for human approval. For days.

```python
from fastaiagent import Chain, FunctionTool, Resume, SQLiteCheckpointer, interrupt
from fastaiagent.chain.node import NodeType


def approve(amount: str):
    if int(amount) > 10_000:
        decision = interrupt(reason="manager_approval", context={"amount": int(amount)})
        return {"approved": decision.approved}
    return {"approved": True}


chain = Chain("refund-flow", checkpointer=SQLiteCheckpointer())
chain.add_node(
    "approve",
    tool=FunctionTool(name="approve_tool", fn=approve),
    type=NodeType.tool,
    input_mapping={"amount": "{{state.amount}}"},
)

from fastaiagent._internal.async_utils import run_sync

# First run — suspends and the process can exit cleanly.
result = chain.execute({"amount": 50_000}, execution_id="refund-abc")
assert result.status == "paused"

# Hours, days, or a server restart later, in any process:
result = run_sync(chain.aresume(
    "refund-abc",
    resume_value=Resume(approved=True, metadata={"approver": "alice"}),
))
assert result.status == "completed"
```

Crash-proof agents (real `SIGKILL` resumes at the last checkpoint),
SQLite locally / Postgres in production (same Protocol surface), the
`@idempotent` decorator that makes `charge_customer` safe to call
inside a paused node, and a built-in `/approvals` UI to drive the
resume from a browser. See [docs/durability/](docs/durability/index.md).

## See every trace, eval, and prompt in your browser — no Docker, no signup

```bash
pip install 'fastaiagent[ui]'
fastaiagent ui
```

Opens a polished web UI at `http://127.0.0.1:7842`. Every agent run you
execute lands here — span tree with Gantt-style timing, JSON-viewer
inspector, Agent Replay fork-and-rerun in the browser, eval runs with
pass-rate trend charts, prompt editor with version lineage, guardrail
events, agent scorecards, and a **read-only browser + search playground
for every `LocalKB`** you've built. Everything stored in one SQLite file at
`./.fastaiagent/local.db`. Bcrypt-hashed local auth. Nothing phones home.

![FastAIAgent Local UI — trace detail](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/03-trace-detail.png)

### See your Chain / Swarm / Supervisor topology rendered as a graph

Pass your runners to `build_app(runners=[...])` to enable the **interactive
React Flow topology view** at `/workflows/{type}/{name}` — agent / HITL /
function nodes, conditional edges, swarm handoffs, supervisor delegation
arrows, all with click-to-inspect node detail panels:

```python
import uvicorn
from fastaiagent import Agent, Chain
from fastaiagent.ui.server import build_app

researcher = Agent(name="researcher", llm=llm)
writer     = Agent(name="writer",     llm=llm)

chain = Chain("research-then-summarise")
chain.add_node("research",  agent=researcher)
chain.add_node("summarize", agent=writer)
chain.connect("research", "summarize")

# Register the chain so the topology endpoint can render it.
app = build_app(runners=[chain])
uvicorn.run(app, host="127.0.0.1", port=7843)
# → open http://127.0.0.1:7843/workflows/chain/research-then-summarise
```

Without `runners=[...]` the trace list, agent stats, and analytics still
populate from runtime spans — but `/workflows/chain/<name>` shows a
"No topology available" callout with the registration recipe above.
Same pattern works for `Swarm` and `Supervisor`. See
[examples/47_workflow_topology.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/47_workflow_topology.py)
and [docs/ui/workflow-visualization.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/workflow-visualization.md)
for the full reference.

### Iterate on prompts in the browser — Prompt Playground

The **Prompt Playground** at `/playground` is the inner-loop iteration
surface for prompts: pick one from the registry, fill in its
`{{variables}}`, choose a provider/model, click **Run**, watch the
response stream back token-by-token. Edit the template inline for
one-off experiments, attach an image for vision models, then click
**Save as eval case** to append the input/output pair to a JSONL
dataset that loads directly via `Dataset.from_jsonl()`. Every run emits
a trace tagged `fastaiagent.source = "playground"` so playground
experiments share the same observability surface as production runs.

![FastAIAgent Local UI — Prompt Playground](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/sprint2-3-playground-streamed-response.png)

See [docs/ui/playground.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/playground.md)
and [examples/49_prompt_playground.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/49_prompt_playground.py)
for the walkthrough.

### See what your agent is made of — Agent Dependency Graph

The **Dependencies** tab on any `/agents/{name}` page renders a
structural graph of the agent: every tool, knowledge base, prompt,
guardrail, and model appears as a node radiating out from the agent
centre. Tools that the LLM has called but weren't registered show up in
amber so hallucinated tool names are visible at a glance. For
**Supervisors** every Worker appears as a sub-agent with its own
subtree; for **Swarms** peers appear as siblings with handoff edges.
Click any node to inspect its details and jump to its own page.

![FastAIAgent Local UI — Agent Dependency Graph](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/sprint2-4-agent-dependency-graph.png)

See [docs/ui/agent-dependencies.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/agent-dependencies.md)
and [examples/50_agent_dependencies.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/50_agent_dependencies.py)
for the walkthrough.

### Debug what your guardrails did — Guardrail Event Detail

Every guardrail firing already shows up on `/guardrails`. Click any row
to open its **detail page** with three panels — *what triggered it*,
*which rule matched*, *what happened next* — plus an execution-context
timeline of the surrounding spans and the other guardrails that ran on
the same content. For `filtered` events the third panel renders a
before/after diff of the rewritten content; for `llm_judge` rules it
shows the judge prompt + response inline. A **Mark as false positive**
button flips a flag stored on the event row so you can curate signal
vs. noise without ever editing the DB — and a new `FP: yes / FP: no`
filter on the list page hides the noise once you've marked it.

![FastAIAgent Local UI — Guardrail Event Detail](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/sprint2-7-guardrail-detail-blocked.png)

See [docs/ui/guardrail-events.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/guardrail-events.md)
and [examples/51_guardrail_events.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/51_guardrail_events.py)
for the walkthrough.

### Compare any two traces — Trace Comparison

Generalises Replay's "original vs forked" diff to *any* two traces, so
you can answer "why did Monday's run differ from Friday's?", A/B-test
two prompts on the same input, or spot a regression after a model
change. Multi-select two rows on `/traces` → **Compare** in the action
bar; or use **Compare with…** on any trace detail page. The view
shows summary delta cards (duration, tokens, cost, span count) over a
span-aligned table — server-side alignment matches by name first then
position, classifying each row as `same` / `slower` / `faster` /
`different output` / `new in A` / `new in B`. Click any row to expand
side-by-side input / output / attributes diffs powered by
`react-diff-viewer-continued`. URL is bookmarkable:
`/traces/compare?a=<id>&b=<id>`.

![FastAIAgent Local UI — Trace Comparison](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/sprint3-2-trace-compare-summary.png)

See [docs/ui/trace-comparison.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/trace-comparison.md)
and [examples/52_trace_compare.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/52_trace_compare.py)
for the walkthrough.

### Curate eval cases inline — Eval Dataset Editor

Datasets stay JSONL on disk (same files `Dataset.from_jsonl()` already
loads) — but the editor at `/datasets` replaces the script-edit-rerun
loop with point-and-click CRUD. Add, edit, duplicate, delete cases;
upload images for multimodal cases (the typed-parts shape is preserved
on disk so cases stay framework-runnable); import / export JSONL with
line-numbered errors on bad input; and **Run eval** kicks off the
existing eval framework against the dataset and surfaces the resulting
`run_id` in `/evals`. The Playground's *Save as eval case* button now
combos over existing datasets with a `+ New` escape hatch so the
inner-loop iteration feeds outer-loop curation without copy-paste.

![FastAIAgent Local UI — Eval Dataset Editor](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/sprint3-6-dataset-detail.png)

See [docs/ui/datasets.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/datasets.md)
and [examples/53_dataset_editor.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/53_dataset_editor.py)
for the walkthrough.

**Or curate in bulk from traces.** `Dataset.from_traces(filter="favorites")` and
`fastaiagent eval curate --filter guardrail --out fixme.jsonl` turn captured agent
traces into dataset cases — every `agent.<name>` span (even nested inside a
chain/supervisor/swarm) becomes one case. Good traces become gold cases; guardrail
or failed traces come back marked `needs_review` for a human to fill in.
**Infrastructure-errored runs** (endpoint 500, timeout — the agent produced no
usable output) are dropped rather than curated as gold, so AutoLLM optimizes only
on agent-attributable failures; the curated set reports how many were dropped
(`ds.curation.coverage_summary()`). See
[docs/evaluation/curation.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/evaluation/curation.md)
and [examples/80_curate_from_traces.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/80_curate_from_traces.py).

### Find any trace — Richer Trace Filtering

The Traces filter bar is now production-grade. **FTS5-backed
full-text search** across LLM prompts and responses
(`gen_ai.prompt`, `gen_ai.response.text`, with `fastaiagent.*`
namespaced fallbacks) — sub-second on 1k spans, regression-tested,
with LIKE fallback for legacy DBs. **Custom date-range picker**
(`react-day-picker`) alongside the quick ranges (15m, 1h, 24h, 7d,
**30d**, All). **More filters** disclosure with duration and cost
ranges. **Saved filter presets** (project-scoped) — capture every
active filter, name it, one-click reapply. **300 ms debounced**
search. And **URL state**: every active filter mirrors into
`?key=value` query params, so refresh, bookmark, share, and
back/forward all preserve filter state.

![FastAIAgent Local UI — Richer Trace Filtering](https://raw.githubusercontent.com/fastaifoundry/fastaiagent-sdk/main/docs/ui/screenshots/sprint3-9-filters-expanded.png)

See [docs/ui/trace-filters.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/trace-filters.md)
and [examples/54_trace_filters.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/54_trace_filters.py)
for the walkthrough.

### Other Local UI surfaces

- **Multimodal trace rendering** — image thumbnails and PDF cards
  render inline in the trace input/output tabs, no raw base64.
  ([docs/ui/multimodal.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/multimodal.md))
- **Checkpoint inspector** at `/executions/{id}` — vertical timeline of
  every checkpoint with status, expandable state snapshots, automatic
  state diff between adjacent rows, and an idempotency-cache panel.
  ([docs/ui/checkpoint-inspector.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/checkpoint-inspector.md))
- **Cost tracking** at the bottom of `/analytics` — three tabs (by
  model / by agent / by chain node) backed by
  `GET /api/analytics/costs`. Reuses the same pricing table the
  per-trace cost column uses, so the numbers always agree.
  ([docs/ui/cost-tracking.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/cost-tracking.md))
- **Export trace as JSON** — Export button on every trace detail page
  opens a dialog with `Include image / PDF data` and
  `Include checkpoint state` toggles. Same payload from the CLI:

  ```bash
  fastaiagent export-trace --trace-id <id> --output trace.json
  ```

  ([docs/ui/export-trace.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/export-trace.md))
- **Project scoping** — every record the SDK writes carries a
  `project_id` resolved from `./.fastaiagent/config.toml` (created
  lazily on the first `agent.run()` from a fresh directory). Multiple
  projects can share one Postgres without cross-contamination; the
  header breadcrumb reads `Local UI // <project-id> // …`.
  ([docs/ui/projects.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/projects.md))

See [docs/ui/](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/index.md) for the full tour; the KB browser is documented at [docs/ui/kb.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/ui/kb.md).

## Evaluate agents systematically

```python
from fastaiagent.eval import evaluate

results = evaluate(
    agent_fn=my_agent.run,
    dataset="test_cases.jsonl",
    scorers=["correctness", "relevance"]
)
print(results.summary())
# correctness: 92% | relevance: 88%
```

## Simulate multi-turn conversations

`evaluate()` scores fixed input→output pairs. `simulate()` stress-tests
**multi-turn** behavior: a `Scenario` drives a conversation between a simulated
user (an LLM persona or a fixed script) and your agent, then a judge scores the
whole transcript against natural-language criteria.

```python
from fastaiagent import Agent, LLMClient, Scenario, SimulatedUser, simulate

agent = Agent(name="support", system_prompt="You are a support agent.",
              llm=LLMClient(provider="openai", model="gpt-4o-mini"))

scenario = Scenario(
    name="refund-request",
    user=SimulatedUser(persona="A frustrated customer who wants a refund."),
    success_criteria=["The agent explains the refund policy clearly and politely."],
    failure_criteria=["The agent is rude or refuses to help."],
)

results = simulate(scenario, agent)   # persists to the Local UI Simulations page
print(results.summary())
```

Runs land on the new **Simulations** UI surface — transcript bubbles,
per-criterion verdicts, and a deep-link from each turn into its trace. See
[docs/simulation](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/simulation/index.md).

### Close the loop — generate, score, and harden

Don't hand-write every test. **`generate_scenarios()`** introspects your agent and
proposes scenarios; **`Scorecard`** rolls any eval/sim run into a per-metric panel;
and **`harden()`** reads the *failures* and hands back concrete fixes — to the
instructions, tools, or guardrails — so each failed run tells you what to change.

```python
from fastaiagent import generate_scenarios, simulate, Scorecard, harden

scenarios = generate_scenarios(agent, n=8, llm=llm)   # auto-author tests
results   = simulate(scenarios, agent)
print(Scorecard.from_simulation(results).summary())   # per-metric roll-up
print(harden(agent, results, llm=llm).summary())      # recommended fixes
```

New named metrics `task_completion`, `hallucination`, and `reflection_quality` join
the existing scorer set. `harden()` is recommend-only — it never mutates your agent.
See [docs/evaluation/agent-hardening.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/evaluation/agent-hardening.md)
and [examples/74_agent_hardening.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/74_agent_hardening.py).

### AutoLLM — actually close the loop

Where `harden()` *recommends*, **`optimize()` (AutoLLM)** *applies and keeps the best*:
it proposes a prompt rewrite, re-evaluates on a held-out split, keeps the winner, and
holdout-guards it against overfitting — eval-driven prompt optimization grounded in
your own data. Greedy coordinate ascent + a metaprompt proposer (the Promptim/DSPy
family; MIPRO-style joint search is a documented `strategy="mipro"` upgrade path).

```python
from fastaiagent import optimize, OptimizeConfig

report = optimize(agent, dataset, ["exact_match"], persist=True)
print(report.summary())                 # baseline → accepted steps → holdout-guarded winner
tuned = report.apply_to(agent)          # a fresh agent with the winning prompt
```

Opt in to the **few-shot** and **learned-memory** levers via `OptimizeConfig(levers=…)`.
Runs persist to the Local UI under **AutoLLM** (trajectory + drill-down into each
candidate's eval run). See [docs/evaluation/optimization.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/evaluation/optimization.md)
and [examples/autollm/](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/autollm/).

## Responsible AI — the Trust Layer

The one question every enterprise review asks: *can you stop it hallucinating,
leaking, or going off-policy?* `responsible_ai()` composes that answer as runtime
guardrails — block hallucinations against your sources, leaked secrets, PII,
toxicity, and off-limits topics. The zero-dependency checks are on by default;
LLM-backed checks are opt-in, so the default bundle adds **no** extra LLM calls.

```python
from fastaiagent import Agent, LLMClient, responsible_ai

llm = LLMClient(provider="openai", model="gpt-4o-mini")
latest_context = ""  # set from your retrieval each turn

agent = Agent(
    name="support",
    llm=llm,
    guardrails=responsible_ai(
        # defaults: prompt_injection (input), pii + secrets (output)
        grounded_to=lambda: latest_context,   # block claims not in your sources
        banned=["politics", "legal advice"],  # semantic topic blocklist
        toxicity=True,                         # LLM toxicity scoring (0–1)
        llm=llm,
    ),
)
```

Every piece is also usable on its own — `grounded()` / `no_hallucination()`,
`no_secrets()` (masks the secret so it's never re-leaked), `banned_topics()` /
`allowed_topics()`, and `toxicity_check(mode="llm")`. Plus the `Reflect`
middleware, which self-critiques the final answer against non-negotiable `facts`
and revises it. Detection logic is shared with the eval scorers
(`faithfulness`, `pii_leakage`, `toxicity`, `prompt_injection`, `moderation`) —
one core detector, two surfaces — with an optional Presidio PII backend via
`pip install fastaiagent[safety]`.

See [docs/guardrails/responsible-ai.md](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/docs/guardrails/responsible-ai.md)
and the runnable [examples/73_responsible_ai.py](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/73_responsible_ai.py).

## Works with LangGraph, CrewAI, PydanticAI — universal harness

Don't rewrite your existing agents. **One line** and they get FastAIAgent's
full Local UI, eval framework, guardrails, prompt registry, and KB on top.

```python
# LangChain / LangGraph
from fastaiagent.integrations import langchain as lc
lc.enable()
result = compiled_graph.invoke({"messages": [HumanMessage("...")]})

# CrewAI
from fastaiagent.integrations import crewai as ca
ca.enable()
result = crew.kickoff(inputs={"input": "..."})

# PydanticAI
from fastaiagent.integrations import pydanticai as pa
pa.enable()
result = agent.run_sync("...")
```

After `enable()`, every LLM call, tool call, retrieval, and graph step
lands in `.fastaiagent/local.db` and renders in the Local UI side-by-side
with native FastAIAgent traces. The same Local UI shows them all — filter
by framework with the free-text input on the Traces page.

The harness also gives you the per-framework helpers `as_evaluable()`,
`with_guardrails()`, `prompt_from_registry()`, `kb_as_retriever()` /
`kb_as_tool()`, and `register_agent()` — see the
[universal harness overview](docs/integrations/overview.md) and the
[per-framework guides](docs/integrations/) for the full feature matrix.

What the harness *can't* give you (and why): Replay (fork-and-rerun),
durability (checkpoint-resumable runs), and suspending HITL all need
execution control of the framework's state machine. Build new
workflows that need those features natively in FastAIAgent.

### …and any other OpenTelemetry / OpenInference instrumentor

Not on the list above? If your stack emits OpenTelemetry spans through **any**
in-process instrumentor (OpenInference, OpenLLMetry / Traceloop, or your own),
one opt-in call captures and richly renders them in the same Local UI:

```python
import fastaiagent as fa
fa.enable_otel_capture()   # foreign spans now show model, tokens, cost, I/O
```

See [Capture any OTel / OpenInference framework](docs/tracing/third-party-otel.md)
for the convention-mapping table and a runnable OpenInference example.

## Build agents with guardrails and cyclic workflows

```python
from fastaiagent import Agent, Chain, LLMClient, Guardrail
from fastaiagent.guardrail import no_pii, json_valid

agent = Agent(
    name="support-bot",
    system_prompt="You are a helpful support agent...",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[search_tool, refund_tool],
    guardrails=[no_pii(), json_valid()]
)

# Chains with loops (retry until quality is good enough)
chain = Chain("support-pipeline", state_schema=SupportState)
chain.add_node("research", agent=researcher)
chain.add_node("evaluate", agent=evaluator)
chain.add_node("respond", agent=responder)
chain.connect("research", "evaluate")
chain.connect("evaluate", "research", max_iterations=3, exit_condition="quality >= 0.8")
chain.connect("evaluate", "respond", condition="quality >= 0.8")

result = chain.execute({"message": "My order is late"}, trace=True)
```

## Deploying

A fastaiagent agent is a plain Python object — wrap it in any web framework and ship it anywhere Python runs. [docs/deployment](docs/deployment/index.md) has copy-paste recipes for:

- **[FastAPI + Uvicorn](docs/deployment/fastapi.md)** — the baseline. Works on a laptop or any VM / container.
- **[Docker → Cloud Run / Fly / Render / Railway](docs/deployment/docker.md)** — one Dockerfile, four managed container platforms.
- **[Modal](docs/deployment/modal.md)** — serverless Python with no container work.
- **[Replicate (Cog)](docs/deployment/replicate.md)** — public inference endpoint.

Every recipe exposes the same `POST /run` + `POST /run/stream` contract so callers don't care where the agent lives. See the runnable starter: [examples/33_deploy_fastapi.py](examples/33_deploy_fastapi.py).

## Expose agents as MCP servers (Claude Desktop / Cursor / Continue / Zed)

Any `Agent` or `Chain` becomes an MCP server with one line:

```python
from fastaiagent import Agent, LLMClient

agent = Agent(name="research_assistant", llm=LLMClient(provider="openai", model="gpt-4o"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(agent.as_mcp_server(transport="stdio").run())
```

Register it in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "python",
      "args": ["/absolute/path/to/my_agent.py"]
    }
  }
}
```

Claude Desktop now treats your fastaiagent as a callable tool. Same config shape for Cursor / Continue / Zed. Or use the CLI: `fastaiagent mcp serve my_agent.py:agent`. See [docs/tools/mcp-server.md](docs/tools/mcp-server.md).

Install: `pip install 'fastaiagent[mcp-server]'`.

## Peer-to-peer swarms with handoffs

Beyond the central-coordinator Supervisor/Worker pattern, agents can hand off to each other directly:

```python
from fastaiagent import Agent, LLMClient, Swarm

llm = LLMClient(provider="openai", model="gpt-4o-mini")

triage = Agent(name="triage", llm=llm, system_prompt="Hand off to the right specialist.")
coder = Agent(name="coder", llm=llm, system_prompt="Answer Python questions.")
writer = Agent(name="writer", llm=llm, system_prompt="Help with prose.")

swarm = Swarm(
    name="help_desk",
    agents=[triage, coder, writer],
    entrypoint="triage",
    handoffs={"triage": ["coder", "writer"], "coder": [], "writer": []},
)
result = swarm.run("How do I reverse a list in Python?")
```

The currently active agent decides when to transfer control — no central LLM. See [docs/agents/swarm.md](docs/agents/swarm.md) for the full guide, and [Swarm vs Supervisor](docs/agents/swarm.md#swarm-vs-supervisor--when-to-use-which) for when to pick which.

## Memory — one object, multi-user, observable

`Memory` is the front door: tiered (global / user / session), multi-user safe, and pluggable — with progressive-disclosure keywords instead of hand-wiring blocks.

```python
from fastaiagent import Agent, LLMClient, Memory

llm = LLMClient(provider="openai", model="gpt-4o-mini")

agent = Agent(name="support", llm=llm, memory=Memory(
    user_id=lambda ctx: ctx.state.user_id,   # one agent, many users — resolved per run
    learn=llm,                               # extract + persist durable user facts
    summarize=llm,                           # compress old turns
    recall="auto",                           # semantic recall of past exchanges
))
```

- **Multi-user safe** — `user_id` routes to a per-user working memory, isolating both durable facts *and* the live window (no cross-session bleed). A missing id yields no personal facts (safe-by-default).
- **Tiers** — `Memory.persist("...", tier="global")` for shared truth; `tier="user", id=...` for personalization; the conversation window is the session tier.
- **Pluggable + semantic** — `Memory(location="postgres://…" | "redis://…")` for external backends; `Memory(semantic="auto")` enables `retrieve("query", tier=, id=)` by meaning.
- **Observable** — every read/write and persist/retrieve is a trace span (scores included), browsable in the Local UI's Memory page.

See [docs/agents/memory.md](docs/agents/memory.md) and [`examples/memory_simple/`](examples/memory_simple/).

<details>
<summary><b>Advanced: composable blocks</b> (the engine under <code>Memory</code>, for custom behaviours)</summary>

`Memory` composes these; reach for them directly to control ordering or write your own `MemoryBlock`:

```python
from fastaiagent import Agent, LLMClient, ComposableMemory, AgentMemory
from fastaiagent import StaticBlock, SummaryBlock, VectorBlock, FactExtractionBlock
from fastaiagent.kb.backends.faiss import FaissVectorStore

llm = LLMClient(provider="openai", model="gpt-4o-mini")

agent = Agent(name="assistant", llm=llm, memory=ComposableMemory(
    blocks=[
        StaticBlock("User is Upendra. Prefers terse answers."),
        SummaryBlock(llm=llm, keep_last=10, summarize_every=5),
        VectorBlock(store=FaissVectorStore(dimension=384)),
        FactExtractionBlock(llm=llm, max_facts=100),
    ],
    primary=AgentMemory(max_messages=20),
))
```

`VectorBlock` works with any `VectorStore` (Qdrant / Chroma / custom). Write your own block by subclassing `MemoryBlock` with two methods.
</details>

## Swap the KB storage layer

Default `LocalKB` ships with FAISS + BM25 + SQLite — zero setup. Point at Qdrant, Chroma, or your own backend with one kwarg:

```python
from fastaiagent.kb import LocalKB
from fastaiagent.kb.backends.qdrant import QdrantVectorStore

# Requires a running Qdrant at http://localhost:6333 — run with
# `docker run -p 6333:6333 qdrant/qdrant`. Wrapped in a try so the
# README snippet test passes when Qdrant isn't reachable.
try:
    kb = LocalKB(
        name="product-docs",
        search_type="vector",
        vector_store=QdrantVectorStore(
            url="http://localhost:6333",
            collection="product-docs",
            dimension=1536,
        ),
    )
    kb.add("docs/")
    results = kb.search("refund policy", top_k=5)
except Exception as e:
    print(f"Qdrant unavailable — start the server first: {e}")
```

Adapters shipped: **FAISS**, **BM25**, **SQLite** (defaults), **Qdrant** (`fastaiagent[qdrant]`), **Chroma** (`fastaiagent[chroma]`). Write your own against the `VectorStore` / `KeywordStore` / `MetadataStore` protocols — see [docs/knowledge-base/backends.md](docs/knowledge-base/backends.md).

**Platform-hosted KBs.** For KBs uploaded and managed on the FastAIAgent platform, use `fa.PlatformKB(kb_id=...)` — same `.search()` / `.as_tool()` surface, retrieval (hybrid + rerank + relevance gate) runs on the platform. See [docs/knowledge-base/platform-kb.md](docs/knowledge-base/platform-kb.md).

## Shape agent behavior with middleware

Compose pre/post model hooks and tool wrappers without subclassing `Agent`:

```python
from fastaiagent import Agent, LLMClient, TrimLongMessages, RedactPII, ToolBudget

agent = Agent(
    name="controlled",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[search_tool],
    middleware=[
        TrimLongMessages(keep_last=30),   # cap history size
        RedactPII(),                      # scrub emails/phones/SSNs both directions
        ToolBudget(max_calls=5),          # cooperatively stop after 5 tool calls
    ],
)
```

Write your own by subclassing `AgentMiddleware` and overriding `before_model`, `after_model`, or `wrap_tool`. See [docs/agents/middleware.md](docs/agents/middleware.md) for ordering, hook reference, and custom patterns.

## Multi-agent teams with context

```python
from fastaiagent import Agent, LLMClient, RunContext, Supervisor, Worker, tool

@tool(name="get_tickets")
def get_tickets(ctx: RunContext[AppState], status: str) -> str:
    """Get support tickets for the current user."""
    return ctx.state.db.query("tickets", user_id=ctx.state.user_id, status=status)

support = Agent(name="support", llm=llm, tools=[get_tickets], system_prompt="Handle tickets.")
billing = Agent(name="billing", llm=llm, tools=[get_billing], system_prompt="Handle billing.")

supervisor = Supervisor(
    name="customer-service",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    workers=[
        Worker(agent=support, role="support", description="Manages tickets"),
        Worker(agent=billing, role="billing", description="Handles billing"),
    ],
    system_prompt=lambda ctx: f"You lead support for {ctx.state.company}. Be helpful.",
)

# Context flows to all workers and their tools
ctx = RunContext(state=AppState(db=db, user_id="u-1", company="Acme"))
result = supervisor.run("Show my open tickets and billing", context=ctx)

# Stream the supervisor's response
import asyncio

async def stream_supervisor() -> None:
    async for event in supervisor.astream("Help me", context=ctx):
        if isinstance(event, TextDelta):
            print(event.text, end="")

asyncio.run(stream_supervisor())
```

## Connect to FastAIAgent Platform (optional)

```python
import fastaiagent as fa

fa.connect(api_key="fa-...", project="my-project")

# Traces automatically sent to platform dashboard
result = agent.run("Help me")

# Pull versioned prompts from platform
prompt = PromptRegistry().get("support-prompt")

# Publish eval results to platform
results = evaluate(agent, dataset=dataset)
results.publish()
```

**SDK works standalone. Platform adds: production observability, prompt management,
evaluation dashboards, team collaboration, HITL approval workflows.**

[Free tier available →](https://app.fastaiagent.net)

---

## Install

```bash
pip install fastaiagent
```

With optional integrations:
```bash
pip install "fastaiagent[openai]"       # OpenAI auto-tracing
pip install "fastaiagent[langchain]"    # LangChain auto-tracing
pip install "fastaiagent[kb]"           # Local knowledge base
pip install "fastaiagent[all]"          # Everything
```

## Documentation

- [Getting Started](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/getting-started)
- [Agent Replay Guide](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/replay)
- [Building Chains with Cycles](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/chains)
- [Guardrails](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/guardrails)
- [Evaluation](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/evaluation)
- [API Reference](https://github.com/fastaifoundry/fastaiagent-sdk/tree/main/docs/api-reference)

## Contributing

FastAIAgent is open source (Apache-2.0) but **does not accept external pull
requests** — please **file issues**, not PRs. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
