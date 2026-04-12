# FastAIAgent SDK

**Build, debug, evaluate, and operate AI agents.**
The only SDK with **Agent Replay** — fork-and-rerun debugging for AI agents.

Works standalone or connected to the [FastAIAgent Platform](https://fastaiagent.net) for visual editing, production monitoring, and team collaboration.

[![PyPI](https://img.shields.io/pypi/v/fastaiagent?v=0.1.8)](https://pypi.org/project/fastaiagent/)
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
```

**No other SDK can do this.**

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

## Trace any agent — yours or LangChain/CrewAI

```python
import fastaiagent
fastaiagent.integrations.langchain.enable()

# Your existing LangChain agent, now with full tracing
result = langchain_agent.invoke({"input": "..."})
# → Traces stored locally or pushed to FastAIAgent Platform
```

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
async for event in supervisor.astream("Help me", context=ctx):
    if isinstance(event, TextDelta):
        print(event.text, end="")
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

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).
