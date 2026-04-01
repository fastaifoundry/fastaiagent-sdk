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

# Run it with tracing
result = agent.run("What's the weather in San Francisco?", trace=True)
print(result.output)
```

## Step 3: See the Trace

```python
print(result.trace.summary())
# Steps: 3 | Duration: 1.2s | Tokens: 245 | Cost: $0.001
# Step 1: LLM call (choose tool) - 400ms
# Step 2: Tool call (get-weather) - 5ms
# Step 3: LLM call (final response) - 750ms
```

## Step 4: View Traces via CLI

```bash
fastaiagent traces list --last 24h
```

That's it. You've built an agent with a tool and full tracing.

## Next Steps

- [Add guardrails](../guardrails/index.md) to validate inputs and outputs
- [Build a chain workflow](../chains/index.md) with loops and checkpointing
- [Debug with Agent Replay](../replay/index.md) using fork-and-rerun
- [Connect to the platform](../platform/index.md) for visual editing
- [Evaluate your agent](../evaluation/index.md) with scorers and datasets
