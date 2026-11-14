---
name: fastaiagent-sdk
description: Use when writing, changing, testing or debugging Python code that imports `fastaiagent` (the FastAIAgent SDK) — agents, tools, guardrails, evals, tracing, multi-agent teams, or connecting to a FastAIAgent platform. Covers the real API (Agent, LLMClient, AgentConfig, @tool, RESTTool, MCPTool, Supervisor, Swarm, Chain, evaluate, connect) and the mistakes assistants make when they guess it from older versions. Read it before writing any fastaiagent code, and run scripts/check_imports.py on what you wrote.
---

# FastAIAgent SDK

`fastaiagent` is a Python SDK for building, testing and operating AI agents. You were probably
not trained on its current API. **Do not write fastaiagent code from memory** — read the
reference for the area you are touching, and check what you wrote before saying it is done.

Install: `pip install fastaiagent`. Extras are named in the reference that needs one
(for example `[mcp]` for MCP tools).

## The ten rules

Each is a mistake assistants make today.

1. **`connect()` needs a target.** Pass `target=` or set `FASTAIAGENT_TARGET`. There is no
   default since 1.100.0, so `connect(api_key=...)` alone fails. `connect()` is only for talking
   to a FastAIAgent platform; an agent runs fine without it.
2. **Never invent a platform address.** `app.fastaiagent.net` is dead. Take the address from the
   user or the environment. If you do not have one, ask.
3. **The framework adapters are gone** (1.96.0). `fastaiagent.integrations.langchain`,
   `.crewai` and `.pydanticai` do not exist. For an agent built in another framework use
   `enable_otel_capture()` with that framework's own OpenTelemetry instrumentation, a plain function
   passed to `evaluate`, `PromptRegistry().get(...).format(...)` for prompts, and `Agent.push()`
   for platform linkage. Details: `references/changed.md`.
4. **`discover_tools()` is gone.** Use `MCPTool.from_server(...)`, which needs the `[mcp]` extra.
5. **`AgentMiddleware` is the one extension model.** Hooks, redaction and budgets go through it.
   Run limits (`max_iterations`, `timeout_seconds`, `max_total_tokens`, `max_cost_usd`) live on
   `AgentConfig`, not on `Agent`.
6. **Give every tool a `replay_class`** (`"read_only"`, `"idempotent"` or `"side_effecting"`).
   An HTTP GET is not read-only by assumption. Unset means `side_effecting`, the safe default.
   The value is recorded on every tool span for a replay engine to choose between re-running a
   tool and reusing its recorded output; the SDK's own `rerun()` blocks tools you did not supply
   or allow (`references/tools.md`).
7. **Never put a key in code.** Read `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `FASTAIAGENT_API_KEY`
   and the rest from the environment. Never print or log one.
8. **Write an offline test next to every agent** with `fastaiagent.testing.TestModel`
   (see `references/testing.md`). A test that needs a live model key is marked and skipped
   without one.
9. **Start with a single agent.** Reach for `Supervisor`, `Swarm` or `Chain` only when the work
   genuinely splits (see `references/teams.md`).
10. **Check before you say done:** `python scripts/check_imports.py <your files>` from this skill's
    folder. It resolves every `fastaiagent` import and attribute in your files against the installed
    SDK and flags removed names. Then run your offline test.

## Where to read

| You are doing | Read |
|---|---|
| Creating or running an agent, structured output, streaming, run limits, middleware, memory | `references/agents.md` |
| Giving an agent tools: `@tool`, REST, MCP, deferred tools, `replay_class` | `references/tools.md` |
| More than one agent: `Supervisor` + `Worker`, `Swarm`, `Chain` | `references/teams.md` |
| Safety checks on input, output or tool calls | `references/guardrails.md` |
| Measuring quality: `evaluate`, scorers, the pytest plugin, gates | `references/evals.md` |
| Connecting to a platform, prompts, traces, push, the runner, the CLI | `references/platform.md` |
| Testing without a model | `references/testing.md` |
| Old code or old advice that no longer works | `references/changed.md` |

Read only the files you need. A tool that cannot read this folder can fetch the same content as one
page: <https://docs.fastaiagent.net/llms-full.txt> (index: <https://docs.fastaiagent.net/llms.txt>).

## A minimal agent and its offline test

```python
from fastaiagent import Agent, tool
from fastaiagent.testing import TestModel


@tool(description="Add two integers.", replay_class="read_only")
def add(a: int, b: int) -> int:
    return a + b


def build_agent(llm=None) -> Agent:
    return Agent(
        name="calculator",
        system_prompt="Use the add tool for arithmetic.",
        llm=llm,
        tools=[add],
    )


def test_calculator_answers_offline():
    agent = build_agent(llm=TestModel("The answer is 4."))
    result = agent.run("What is 2 + 2?")
    assert "4" in result.output
```

For a real run, pass `llm=LLMClient(provider=..., model=...)` with the key in the environment
(`references/agents.md` has the signature and the provider names).

## When something fails

- **`ImportError` / `AttributeError` on a `fastaiagent` name:** you guessed it. Run
  `scripts/check_imports.py`, then look the name up in the reference. Do not try variants.
- **`connect` raises about a target:** rule 1.
- **A feature needs an extra** (MCP, vector stores, Postgres): the error names it; install it
  and tell the user, do not catch and ignore it.
- **A guardrail reports it could not run:** that is not a pass. Fix its configuration
  (`references/guardrails.md`).

## Reporting

When you finish, say which files you wrote, what `check_imports.py` printed, and what your
offline test did. If you could not run something (no key, no network), say so; do not claim
it works.
