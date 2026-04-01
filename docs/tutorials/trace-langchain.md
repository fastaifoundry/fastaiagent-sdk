# Trace Your LangChain Agent in 2 Minutes

Already using LangChain? Add full tracing in 2 lines.

## Install

```bash
pip install "fastaiagent[langchain]"
```

## Add Tracing to Your Existing Code

```python
# Your existing LangChain code - unchanged
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o")
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# Add these 2 lines
import fastaiagent
fastaiagent.integrations.langchain.enable()

# Run as usual - now with full tracing
result = executor.invoke({"input": "What's the weather?"})
```

Traces are stored locally in SQLite. View them:

```bash
fastaiagent traces list --last 24h
fastaiagent replay <trace_id>
```

## What Gets Traced

When LangChain auto-tracing is enabled, FastAIAgent captures:

- **LLM calls** — model, tokens, latency, prompt/completion
- **Tool calls** — tool name, arguments, output
- **Chain execution** — start/end, input/output
- **Retrieval** — queries and results (if using retrievers)

## Disable Tracing

```python
fastaiagent.integrations.langchain.disable()
```

## Push Traces to the Platform

```python
from fastaiagent import FastAI

fa = FastAI(api_key="sk-...", project="my-project")
# Traces are automatically pushed when platform is connected
```

## Next Steps

- [Push traces to the platform](../platform/index.md)
- [Evaluate your agent](../evaluation/index.md)
- [Migrate to native FastAIAgent](../migration-guides/from-langsmith.md) for the full experience
