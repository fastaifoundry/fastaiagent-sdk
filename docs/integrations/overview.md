# FastAIAgent as a Universal Agent Harness

FastAIAgent is two things:

1. **A native agent framework** — `Agent`, `Chain`, `Swarm`,
   `Supervisor`, with full Replay and durability. For new builds.
2. **A universal agent harness** — observability, eval, guardrails,
   prompt registry, and knowledge bases that **wrap any framework**.
   For existing codebases that you don't want to rewrite.

Both surfaces use the same Local UI and the same trace store. As a
team builds more native FastAIAgent agents the balance shifts naturally
over time, but you don't have to choose up front.

## What "harness" means

You keep your existing **LangGraph**, **CrewAI**, or **PydanticAI**
agents. With one or two lines of glue you get:

| Feature | How it shows up |
|---|---|
| Auto-tracing | Every LLM call, tool call, retrieval, and graph step lands in `.fastaiagent/local.db` and renders in the Local UI |
| Token + cost capture | Pulled from each provider's response usage block; shown on the Analytics dashboard |
| Eval framework | `fa.evaluate(as_evaluable(your_agent), dataset=...)` |
| Guardrails | `with_guardrails(your_agent, input_guardrails=[...], output_guardrails=[...])` |
| Prompt registry | `prompt_from_registry("support-system", agent="my-agent")` returns the framework-native prompt object |
| Knowledge bases | `kb_as_retriever("support-kb")` (LangChain) / `kb_as_tool("support-kb")` (CrewAI / PydanticAI) |
| Dependency graph | `register_agent(your_agent, name="my-agent")` populates the Agent Detail page in the UI |

## Feature matrix

| Feature | Native FastAIAgent | LangChain / LangGraph | CrewAI | PydanticAI |
|---|---|---|---|---|
| Auto-tracing | ✅ | ✅ | ✅ | ✅ |
| Local UI (all surfaces) | ✅ | ✅ | ✅ | ✅ |
| Analytics & cost tracking | ✅ | ✅ | ✅ | ✅ |
| Eval framework | ✅ | ✅ via `as_evaluable()` | ✅ via `as_evaluable()` | ✅ via `as_evaluable()` |
| Guardrails | ✅ | ✅ via `with_guardrails()` | ✅ via `with_guardrails()` | ✅ via `with_guardrails()` |
| Prompt registry | ✅ | ✅ via `prompt_from_registry()` | ✅ via `prompt_from_registry()` | ✅ via `prompt_from_registry()` |
| Knowledge bases | ✅ | ✅ as retriever | ✅ as tool | ✅ as tool |
| Prompt playground | ✅ | ✅ | ✅ | ✅ |
| Trace comparison | ✅ | ✅ | ✅ | ✅ |
| Export trace | ✅ | ✅ | ✅ | ✅ |
| Dependency graph | ✅ | ✅ via `register_agent()` | ✅ via `register_agent()` | ✅ via `register_agent()` |
| Workflow visualisation | ✅ | ✅ via `register_agent()` | ✅ via `register_agent()` | ❌ (single-agent) |
| Agent Replay (fork-and-rerun) | ✅ | ❌ | ❌ | ❌ |
| Durability (checkpointing) | ✅ | ❌ | ❌ | ❌ |
| Suspending HITL | ✅ | ❌ | ❌ | ❌ |

The ❌ cells aren't gaps — they're **migration incentives**. Replay,
durability, and suspending HITL all require execution control of the
framework's state machine, which only the native FastAIAgent runtime
provides. When you're ready to build new workflows that need those
features, build them natively.

## Limitations

A few things diverge from what a casual reading of the spec might
suggest. They are deliberate.

- **Guardrails block, they don't redact.** `GuardrailResult` has
  `passed` / `score` / `message` / `metadata` — there's no
  `filtered_text`. A failing blocking guardrail logs an event row
  *and* raises `GuardrailBlocked`. To redact, write a custom
  guardrail that runs before the wrapped agent and rewrites the input
  itself.

- **External-agent registration is per-machine.** `register_agent()`
  writes to `.fastaiagent/local.db`; there is no cross-machine sync.
  When the Local UI is opened, it merges the in-memory `ctx.runners`
  registry with this on-disk one, so registrations from a separate
  process show up in the UI.

- **PydanticAI has no workflow visualisation.** PydanticAI agents are
  single-agent. The Agent Detail dependency graph still works, but
  there's no node-and-edge topology to render.

- **Prompt lineage in LangChain uses a thread-local stack.** LCEL
  isolates each step with `copy_context().run(...)`, so a `ContextVar`
  set inside a template's `format_messages` doesn't survive into the
  next step's `on_chat_model_start`. We use a per-thread LIFO stack
  instead. Concurrent chains in the *same thread* can race; in
  practice the LCEL pipeline is sequential per chain.

## Per-framework guides

- [LangChain / LangGraph](langchain.md)
- [CrewAI](crewai.md)
- [PydanticAI](pydanticai.md)
