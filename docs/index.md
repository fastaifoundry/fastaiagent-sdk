# FastAIAgent SDK

**Build, debug, evaluate, and operate AI agents.**

The only SDK with **Agent Replay** — fork-and-rerun debugging for AI agents.

Works standalone or connected to the [FastAIAgent Platform](https://fastaiagent.net) for visual editing, production monitoring, and team collaboration.

---

## What Makes FastAIAgent Different

| Feature | FastAIAgent | LangSmith | Langfuse |
|---------|-------------|-----------|----------|
| **Agent Replay (fork-and-rerun)** | Yes | No | No |
| **Build agents in code** | Yes | No | No |
| **Cyclic chain workflows** | Yes | LangGraph | No |
| **Built-in guardrails** | Yes | No | No |
| **OTel-native tracing** | Yes | Proprietary | Proprietary |
| **Fragment prompt composition** | Yes | No | No |
| **Visual editor sync** | Yes | No | No |

---

## Quick Start

```bash
pip install fastaiagent
```

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.tools import FunctionTool

agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[FunctionTool(name="greet", fn=lambda name: f"Hello, {name}!")]
)

result = agent.run("Say hello to World", trace=True)
print(result.output)
print(result.trace.summary())
```

---

## Core Features

- **[Agents](agents/index.md)** — Build agents with tools, memory, and multi-agent teams
- **[Chains](chains/index.md)** — Directed graph workflows with cycles, typed state, and checkpointing
- **[Guardrails](guardrails/index.md)** — Input/output/tool validation (code, regex, LLM judge)
- **[Tracing](tracing/index.md)** — OTel-native tracing with local SQLite storage
- **[Agent Replay](replay/index.md)** — Fork-and-rerun debugging at any execution step
- **[Evaluation](evaluation/index.md)** — Scorers, datasets, LLM-as-judge, trajectory eval
- **[Prompts](prompts/index.md)** — Registry with versioning and fragment composition
- **[Knowledge Base](knowledge-base/index.md)** — Local file ingestion with embedding search
- **[Platform Sync](platform/index.md)** — Push agents to the FastAIAgent visual editor
- **[Integrations](integrations/index.md)** — Auto-tracing for OpenAI, Anthropic, LangChain, CrewAI

---

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Build Your First Agent in 5 Minutes](tutorials/first-agent.md)
- [Trace a LangChain Agent in 2 Minutes](tutorials/trace-langchain.md)
- [Debug with Agent Replay](tutorials/debug-with-replay.md)
