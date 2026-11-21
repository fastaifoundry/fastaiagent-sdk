---
name: fastaiagent-sdk
description: Use when writing, changing, testing, evaluating or debugging Python code that imports `fastaiagent` (the FastAIAgent SDK) — agents, tools, guardrails, evals, tracing, multi-agent teams, or connecting to a FastAIAgent platform, including evaluating an agent against a platform dataset and enforcing guardrails authored on the platform. Covers the real API (Agent, LLMClient, AgentConfig, @tool, RESTTool, MCPTool, Supervisor, Swarm, Chain, evaluate, connect) and the mistakes assistants make when they guess it from older versions. Read it before writing any fastaiagent code, and run scripts/check_imports.py on what you wrote.
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
   user or the environment. If you do not have one, ask. When the user gives you one, keep it in
   `FASTAIAGENT_TARGET`, not in code.
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
   and the rest from the environment. Never print or log one. If a key is not in the environment,
   ask the user how to load it; do not search their home directory or other projects for one.
8. **Write an offline test next to every agent** with `fastaiagent.testing.TestModel`
   (see `references/testing.md`). A test that needs a live model key is marked and skipped
   without one.
9. **Start with a single agent.** Reach for `Supervisor`, `Swarm` or `Chain` only when the work
   genuinely splits (see `references/teams.md`).
10. **Check, then run the real thing.** Run this skill's `scripts/check_imports.py <your files>`
    with the project's own interpreter (for example `.venv/bin/python
    .agents/skills/fastaiagent-sdk/scripts/check_imports.py desk/agent.py`); it works from the
    project root. Use that interpreter for every command, not a bare `python`, which can be a
    different Python without the SDK; never report "not installed" from a bare `python`. It resolves every `fastaiagent` import and attribute against the installed SDK
    and flags removed names. Then run your offline test. Then, if the user's model key is in the
    environment, run the real entry point once and read what it prints. Offline tests use scripted
    models, so they cannot see a wrong model id, a missing key or a dead server. **Never "fix" a
    model id or an address you have not tried:** a failing real run is evidence, your belief is
    not. If you have no key, say you did not run it. **Check before you say something is
    missing:** look at whether the key or setting is set (for example
    `python -c "import os; print(bool(os.environ.get('OPENAI_API_KEY')))"`) and report what the
    check printed; assistants have said "no key in the environment" without looking, when it was
    there. When the user asks to *see* a run and a key is set, show the real-model run: scripted
    models are for tests, and a `--real` switch you never use is not a run.

## Where to read

| You are doing | Read |
|---|---|
| Creating or running an agent, structured output, streaming, run limits, middleware, memory | `references/agents.md` |
| Giving an agent tools: `@tool`, REST, MCP, deferred tools, `replay_class` | `references/tools.md` |
| More than one agent: `Supervisor` + `Worker`, `Swarm`, `Chain` | `references/teams.md` |
| Safety checks on input, output or tool calls | `references/guardrails.md` |
| Measuring quality: `evaluate`, scorers, the pytest plugin, gates | `references/evals.md` |
| Connecting to a platform: what it supplies (guardrail rules, prompts, datasets, judges, knowledge bases) and receives (traces, eval verdicts), the connected-agent checklist, push, the runner, the CLI | `references/platform.md` |
| Guardrails authored on the platform | `references/guardrails.md`, "Plane-authored rules" |
| Eval datasets, judges and verdicts on the platform | `references/evals.md`, "With a platform" |
| Versioned prompts in a registry (local or on the platform), rolling out a new version | `references/platform.md`, "Prompts" |
| Testing without a model, and platform rules offline (`plane_rules`) | `references/testing.md` |
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

## Starting a new project

A project that only has agents and tests is not finished: tests must run with plain `pytest`, and
something must actually start the agent.

```text
my_app/
  agents.py        build_agent(llm=None) and the other builders
  main.py          the entry point (below)
  tests/           offline tests, TestModel or FunctionModel
  pytest.ini       [pytest]  then  pythonpath = .   (without it plain `pytest` cannot import my_app)
  .gitignore       .fastaiagent/   (local traces and the eval database)
```

An entry point loads the keys from the environment, connects only when a platform is configured,
and always ends with `disconnect()`. A helper such as `connect_platform()` that nothing calls
sends no traces, and a short script can exit before the background export finishes:

```python
import os
import sys

import fastaiagent as fa
from fastaiagent import LLMClient

from my_app.agents import build_agent


def main(question: str) -> None:
    if os.environ.get("FASTAIAGENT_API_KEY") and os.environ.get("FASTAIAGENT_TARGET"):
        fa.connect(
            api_key=os.environ["FASTAIAGENT_API_KEY"],
            target=os.environ["FASTAIAGENT_TARGET"],
        )
    try:
        agent = build_agent(llm=LLMClient(provider="openai", model="gpt-5-mini"))
        print(agent.run(question).output)
    finally:
        fa.disconnect()  # flushes pending traces first; a no-op when never connected


if __name__ == "__main__":
    main(sys.argv[1])
```

Build what was asked. Do not add tools, fake data, aliases, command-line flags or helper layers the
user did not ask for; say what you would add and let them decide. That includes handling for cases
that do not exist yet (rule types the platform does not have, providers the user does not use):
build for what is there. When a choice has a cost the user should hear
(for example `routing="decisions"` needs OpenAI's Decisions endpoint at run time, or a keyword topic
guardrail is coarse), say it in your summary.

## When something fails

- **`ImportError` / `AttributeError` on a `fastaiagent` name:** you guessed it. Run
  `scripts/check_imports.py`, then look the name up in the reference. Do not try variants.
- **`connect` raises about a target:** rule 1.
- **A feature needs an extra** (MCP, vector stores, Postgres): the error names it; install it
  and tell the user, do not catch and ignore it.
- **A guardrail reports it could not run:** that is not a pass. Fix its configuration
  (`references/guardrails.md`).

## Reporting

When you finish, say which files you wrote, what `check_imports.py` printed, what your offline
test did, and what you ran for real (the entry point, the model, the platform) with what it
printed. If you could not run something (no key, no network, no server), say so; do not claim it
works.
