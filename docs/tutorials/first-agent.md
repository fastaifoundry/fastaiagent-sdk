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
from fastaiagent.trace import Replay

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
print(Replay.load(result.trace_id).summary())
# Trace: 75e1672ded9f2cba4811de44a190f8f2
# Name: agent.assistant
# Status: UNSET
# Spans: 4
# Duration: 2026-10-31T01:06:02.624746+00:00 → 2026-10-31T01:06:04.359789+00:00
#
# Steps:
#   [0] agent.assistant
#   [1] llm.openai.gpt-4o-mini
#   [2] tool.get-weather
#   [3] llm.openai.gpt-4o-mini
```

That's it. You've built an agent with a tool and full tracing in 5 minutes.

## Next Steps

- [Add guardrails](../guardrails/index.md)
- [Build a chain workflow](../chains/index.md)
- [Debug with Agent Replay](../replay/index.md)
- [Connect to the platform](../platform/index.md)
