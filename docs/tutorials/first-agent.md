# Build Your First Agent in 5 Minutes

## Prerequisites

- Python 3.10+
- An OpenAI API key

## Install

```bash
pip install "fastaiagent[openai]"
```

## Create an Agent with a Tool

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

# Run it
result = agent.run("What's the weather in San Francisco?", trace=True)
print(result.output)

# See the trace
print(result.trace.summary())
# Steps: 3 | Duration: 1.2s | Tokens: 245 | Cost: $0.001
# Step 1: LLM call (choose tool) - 400ms
# Step 2: Tool call (get-weather) - 5ms
# Step 3: LLM call (final response) - 750ms
```

That's it. You've built an agent with a tool and full tracing in 5 minutes.

## Next Steps

- [Add guardrails](../guardrails/index.md)
- [Build a chain workflow](../chains/index.md)
- [Debug with Agent Replay](../replay/index.md)
- [Connect to the platform](../platform/index.md)
