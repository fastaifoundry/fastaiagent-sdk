# FastAIAgent SDK

**Build, debug, evaluate, and operate AI agents.**
The only SDK with **Agent Replay** — fork-and-rerun debugging for AI agents.

Works standalone or connected to the [FastAIAgent Platform](https://fastaiagent.net) for visual editing, production monitoring, and team collaboration.

[![PyPI](https://img.shields.io/pypi/v/fastaiagent)](https://pypi.org/project/fastaiagent/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://github.com/fastaifoundry/fastaiagent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/fastaifoundry/fastaiagent-sdk/actions)
[![Python](https://img.shields.io/pypi/pyversions/fastaiagent)](https://pypi.org/project/fastaiagent/)

---

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

## Connect to FastAIAgent Platform (optional)

```python
from fastaiagent import FastAI

fa = FastAI(api_key="sk-...", project="customer-support")

# Push your agent to the platform — see it in the visual editor
fa.push(chain)

# Traces appear in the platform dashboard
# Prompts sync between code and platform
# Eval results visible in the platform
```

**SDK works standalone. Platform adds: visual chain editor, production monitoring,
advanced KB intelligence, prompt optimization, team collaboration, HITL approval workflows.**

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
