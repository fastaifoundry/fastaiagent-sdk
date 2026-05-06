# FastAIAgent SDK — Examples

Two flavors live in this directory:

- **Templates** (folders) — full project shape: README, `.env.example`, `requirements.txt`, multiple modules, eval suite, CLI entry point, Local-UI integration. Copy a template, edit your prompt + tools, ship it.
- **Snippets** (numbered `NN_*.py` files) — single-feature illustrations meant to be read top-to-bottom. Each one stands alone; run with `python examples/NN_<name>.py`.

If you're new, start with the templates, then dip into snippets when you need a specific feature.

---

## Templates

| Template | Use case | What it teaches | Read-first if you want |
|---|---|---|---|
| [`customer-support-agent/`](customer-support-agent/) | Single-agent KB-grounded support chatbot with HITL on ticket creation | `Agent` + tools + `LocalKB` + memory + middleware + guardrails + `interrupt()` / `aresume()` + `@idempotent` + `PromptRegistry` + `Replay` + `LLMJudge`/RAG eval + multimodal + FastAPI HITL deploy | the **single-agent** golden path |
| [`research-agent/`](research-agent/) | Perplexity / Deep-Research-style multi-agent investigation with verifier-driven revision loop | `Supervisor` + `Worker` + custom `Scorer` + pluggable web-search backends + `Supervisor.astream()` + replay across handoffs + per-case eval context | LLM-driven multi-agent orchestration |
| [`sales-sdr-agent/`](sales-sdr-agent/) | Clay / HubSpot Breeze-style outbound SDR pipeline with HITL approval before send | `Chain` DAG with conditional edge routing + `chain.aexecute(... context=ctx)` + LLM agents wrapped inside tool nodes + `fa.interrupt()` + `@idempotent` send + 3 pluggable backends (Clearbit / Salesforce / SendGrid) | a **deterministic-flow** workflow with HITL |
| [`meeting-notes-agent/`](meeting-notes-agent/) | Granola / Otter / Fireflies-style transcript → structured notes generator | `Chain` with **parallel LLM fan-out** via `asyncio.gather` + Pydantic `MeetingNotes` schema enforcement at merge + multimodal `fa.PDF.extract_text` + per-attendee personalization | parallel analyzer fan-out + schema enforcement |
| [`personal-assistant/`](personal-assistant/) | Long-lived REPL personal assistant with cross-session memory | **Every memory-block type** — `StaticBlock + SummaryBlock + VectorBlock + FactExtractionBlock` — composed via `ComposableMemory` with `FaissVectorStore` + on-disk persistence + `PromptRegistry`-backed system prompt | the canonical **memory** showcase |
| [`harness-migration/`](harness-migration/) | Wrap an existing **LangGraph / CrewAI / PydanticAI** agent with FastAIAgent's harness | `fastaiagent.integrations.{langchain,crewai,pydanticai}` — `enable()` auto-tracing + `with_guardrails()` + `kb_as_retriever()` / `kb_as_tool()` + `prompt_from_registry()` + `register_agent()` + cross-framework `fa.evaluate()` via `as_evaluable()` | gradual migration **from another framework** |

### Recommended onboarding path

1. Skim [`customer-support-agent/README.md`](customer-support-agent/README.md) to see how `Agent`, tools, KB, guardrails, memory, HITL, and eval fit together in one place.
2. Run `python customer-support-agent/agent.py` and `python customer-support-agent/eval_suite.py`.
3. Open `fastaiagent ui start` in another terminal — visit `/traces`, `/agents`, `/evals`. This is your debug surface from now on.
4. Pick a template that matches your use-case shape:
   - **LLM-driven multi-agent** with revisions → [`research-agent/`](research-agent/)
   - **Deterministic workflow with HITL gates** → [`sales-sdr-agent/`](sales-sdr-agent/)
   - **Parallel LLM analysis with structured output** → [`meeting-notes-agent/`](meeting-notes-agent/)
   - **Long-running session with rich memory** → [`personal-assistant/`](personal-assistant/)
   - **Already on LangChain / CrewAI / PydanticAI** → [`harness-migration/`](harness-migration/)
5. For specific features (RAG, OTel export, MCP, cyclic chains, etc.), grep the snippet table below.

### How to fork a template

```bash
cp -r examples/customer-support-agent ~/my-agent
cd ~/my-agent
cp .env.example .env       # add OPENAI_API_KEY
pip install -r requirements.txt   # installs fastaiagent>=1.6.0 from PyPI
# edit SYSTEM_PROMPT in agent.py, add/remove tools in tools.py, replace knowledge/
python agent.py
```

`fastaiagent>=1.6.0` is published on PyPI — no need to install from source unless you're contributing to the SDK.

---

## Snippets

Numbered scripts grouped by topic. Each one is ~50–150 lines and demonstrates exactly one feature.

### Agents core
- [`01_simple_agent.py`](01_simple_agent.py) — minimal Agent with one tool
- [`12_streaming.py`](12_streaming.py) — `agent.astream()` token-by-token
- [`14_context_di.py`](14_context_di.py) — `RunContext[Deps]` dependency injection
- [`15_context_di_anthropic.py`](15_context_di_anthropic.py) — same on Anthropic
- [`16_dynamic_instructions.py`](16_dynamic_instructions.py) — callable `system_prompt`
- [`17_dynamic_instructions_advanced.py`](17_dynamic_instructions_advanced.py) — fragment-driven prompts
- [`20_output_type.py`](20_output_type.py) — Pydantic structured output
- [`21_retry_backoff.py`](21_retry_backoff.py) — retry / backoff config
- [`22_llm_parameters.py`](22_llm_parameters.py) — provider-specific LLM params

### Tools & guardrails
- [`03_guardrails.py`](03_guardrails.py) — built-in PII / toxicity / JSON guardrails
- [`23_tool_guardrails.py`](23_tool_guardrails.py) — guardrails on tool calls / results
- [`27_middleware_tool_budget.py`](27_middleware_tool_budget.py) — `ToolBudget` middleware
- [`30_memory_blocks.py`](30_memory_blocks.py) — `ComposableMemory` block API
- [`32_mcp_expose_agent.py`](32_mcp_expose_agent.py) — expose Agent as MCP server
- [`41_agent_tools.py`](41_agent_tools.py) — tool decoration patterns

### Knowledge bases
- [`06_rag_agent.py`](06_rag_agent.py) — basic RAG with `LocalKB`
- [`28_kb_chroma.py`](28_kb_chroma.py) — Chroma backend
- [`29_kb_qdrant.py`](29_kb_qdrant.py) — Qdrant backend
- [`34_platform_kb.py`](34_platform_kb.py) — `PlatformKB` (remote, multi-tenant)
- [`37_kb_ui.py`](37_kb_ui.py) — KB management via Local UI

### Workflows & multi-agent
- [`02_chain_with_cycles.py`](02_chain_with_cycles.py) — `Chain` DAG with cycles
- [`18_supervisor_worker.py`](18_supervisor_worker.py) — `Supervisor` + `Worker` (also see `research-agent/`)
- [`31_swarm_research_team.py`](31_swarm_research_team.py) — `Swarm` peer-to-peer handoffs
- [`36_chain_workflow.py`](36_chain_workflow.py) — Chain end-to-end
- [`39_workflows_demo.py`](39_workflows_demo.py) — workflow comparison
- [`47_workflow_topology.py`](47_workflow_topology.py) — topology API for the UI

### Durability, HITL, idempotency
- [`42_durability_hitl.py`](42_durability_hitl.py) — `interrupt()` + `Resume` + `@idempotent` (also see `customer-support-agent/`)

### Evaluation
- [`07_eval_pipeline.py`](07_eval_pipeline.py) — basic `evaluate()`
- [`24_rag_eval.py`](24_rag_eval.py) — `Faithfulness` / `AnswerRelevancy` / `ContextPrecision`
- [`25_safety_eval.py`](25_safety_eval.py) — `Toxicity` / `Bias` / `PIILeakage`
- [`26_similarity_eval.py`](26_similarity_eval.py) — `SemanticSimilarity` / BLEU / ROUGE / Levenshtein
- [`40_evals_compare.py`](40_evals_compare.py) — A/B compare two runs

### Tracing & replay
- [`04_agent_replay.py`](04_agent_replay.py) — `Replay.fork_at(...).rerun()`
- [`08_trace_langchain.py`](08_trace_langchain.py) — auto-trace LangChain agents
- [`09_otel_export.py`](09_otel_export.py) — OpenTelemetry export
- [`10_trace_query.py`](10_trace_query.py) — query traces from the SDK
- [`38_replay_comparison.py`](38_replay_comparison.py) — diff two replays
- [`48_export_trace.py`](48_export_trace.py) — export trace as JSON
- [`52_trace_compare.py`](52_trace_compare.py) — Local UI trace comparison
- [`54_trace_filters.py`](54_trace_filters.py) — saved filter presets

### Multimodal
- [`43_multimodal_image.py`](43_multimodal_image.py) — `fa.Image` input
- [`44_multimodal_pdf.py`](44_multimodal_pdf.py) — `fa.PDF` input
- [`45_multimodal_chain.py`](45_multimodal_chain.py) — multimodal in Chain
- [`46_multimodal_swarm.py`](46_multimodal_swarm.py) — multimodal in Swarm

### Local UI features (sprint demos)
- [`35_local_ui.py`](35_local_ui.py) — start the UI from Python
- [`49_prompt_playground.py`](49_prompt_playground.py) — Playground API
- [`50_agent_dependencies.py`](50_agent_dependencies.py) — dependency-graph API
- [`51_guardrail_events.py`](51_guardrail_events.py) — guardrail-events drilldown
- [`53_dataset_editor.py`](53_dataset_editor.py) — eval dataset CRUD

### Platform & integrations
- [`05_prompt_fragments.py`](05_prompt_fragments.py) — `PromptRegistry` fragment composition
- [`10_platform_sync.py`](10_platform_sync.py) — `fa.connect()` end-to-end
- [`19_connect_e2e.py`](19_connect_e2e.py) — full platform round-trip
- [`33_deploy_fastapi.py`](33_deploy_fastapi.py) — wrap an Agent in FastAPI
- [`55_trace_crewai.py`](55_trace_crewai.py) — auto-trace CrewAI agents
- [`56_trace_pydanticai.py`](56_trace_pydanticai.py) — auto-trace PydanticAI
- [`57_eval_langchain.py`](57_eval_langchain.py) — eval a LangGraph agent
- [`58_guardrail_pydanticai.py`](58_guardrail_pydanticai.py) — `with_guardrails()` on PydanticAI
- [`59_register_external_agent.py`](59_register_external_agent.py) — register a non-fastaiagent agent

### Misc
- [`11_cli_usage.sh`](11_cli_usage.sh) — `fastaiagent` CLI commands
- [`13_structured_output.py`](13_structured_output.py) — JSON-schema constrained output

---

## Where to go next

- **Conceptual docs**: [`../docs/`](../docs/) — agent durability, memory, eval semantics, HITL design
- **API reference**: [`../docs/api-reference/`](../docs/api-reference/)
- **CHANGELOG**: [`../CHANGELOG.md`](../CHANGELOG.md) — see what shipped in each release

If you build something with these and want to contribute it back, PRs into this directory are welcome — keep them in the same shape (folder for templates, single-purpose `NN_*.py` for snippets).
