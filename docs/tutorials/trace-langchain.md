# Trace Your LangChain Agent in 2 Minutes

Already using LangChain? Add tracing in 2 lines.

## Install

The `[langchain]` extra only pulls `langchain-core`. To run a full LangChain
agent you'll also need `langchain` itself plus a model adapter such as
`langchain-openai`:

```bash
pip install "fastaiagent[langchain]" langchain langchain-openai
```

## Add Tracing to Your Existing Code

```python
# Your existing LangChain code - unchanged
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4.1")

# Add these 2 lines
import fastaiagent
fastaiagent.integrations.langchain.enable()
handler = fastaiagent.integrations.langchain.get_callback_handler()

# Run as usual — pass the handler via callbacks
response = llm.invoke(
    [HumanMessage(content="What's the weather?")],
    config={"callbacks": [handler]},
)
```

!!! note "LangChain 1.x"
    Earlier versions of this tutorial used `langchain.agents.create_tool_calling_agent`
    and `AgentExecutor`. Those symbols were removed in LangChain 1.x —
    use `langchain.agents.create_agent` (or `langgraph`) instead.

Traces are stored locally in SQLite. View them:

```bash
fastaiagent traces list --last 24h
fastaiagent replay <trace_id>
```

## What the Handler Hooks

The callback handler subclasses `langchain_core.callbacks.BaseCallbackHandler`
and instruments:

- `on_llm_start` / `on_llm_end` — `langchain.llm.<model>` span
- `on_tool_start` / `on_tool_end` — `langchain.tool.<name>` span
- `on_llm_error` / `on_tool_error` — closes the matching open span on failure

Spans land in `.fastaiagent/local.db` and are visible through
`fastaiagent traces list` and the replay/export CLI commands.

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
