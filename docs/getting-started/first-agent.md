# Build Your First Agent

This guide walks you through creating an agent with a tool and full tracing in under 5 minutes.

## Prerequisites

- Python 3.10+
- An OpenAI API key (set as `OPENAI_API_KEY` environment variable)

## Step 1: Install

```bash
pip install "fastaiagent[openai]"
```

## Step 2: Create an Agent with a Tool

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.tool import FunctionTool

# Define a tool
weather_tool = FunctionTool(
    name="get-weather",
    description="Get current weather for a city",
    fn=lambda city: {"city": city, "temp": "72F", "condition": "sunny"}
)

# Create an agent
agent = Agent(
    name="assistant",
    system_prompt="You are a helpful assistant. Use tools when needed.",
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[weather_tool]
)

# Run it — every run is automatically traced
result = agent.run("What's the weather in San Francisco?")
print(result.output)
print(result.trace_id)  # e.g. "b6acf1ef2c2779bbc2fcf80802ae0534"
```

## Step 3: View and Replay the Trace

```python
from fastaiagent.trace import Replay

# Load the trace using the trace_id from the result
replay = Replay.load(result.trace_id)
print(replay.summary())
# Trace: b6acf1ef2c2779bbc2fcf80802ae0534
# Steps: 3 | Duration: 1.2s | Tokens: 245
# Step 1: LLM call (choose tool) - 400ms
# Step 2: Tool call (get-weather) - 5ms
# Step 3: LLM call (final response) - 750ms
```

## Step 4: Browse Past Traces

```python
from fastaiagent.trace import TraceStore

store = TraceStore()
for t in store.list_traces(last_hours=24):
    print(f"{t.trace_id[:12]}  {t.name}  {t.status}")
```

Or via CLI:

```bash
fastaiagent traces list --last-hours 24
```

That's it. You've built an agent with a tool and full tracing.

## Next Steps

- [Add guardrails](../guardrails/index.md) to validate inputs and outputs
- [Build a chain workflow](../chains/index.md) with loops and checkpointing
- [Debug with Agent Replay](../replay/index.md) using fork-and-rerun
- [Connect to the platform](../platform/index.md) for visual editing
- [Evaluate your agent](../evaluation/index.md) with scorers and datasets
