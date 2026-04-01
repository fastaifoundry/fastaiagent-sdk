# Agent Memory

Memory lets agents remember previous conversations across multiple `run()` calls. This is essential for building chatbots, multi-turn assistants, and any agent that needs context from prior interactions.

## Basic Usage

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.agent import AgentMemory

memory = AgentMemory(max_messages=20)

agent = Agent(
    name="assistant",
    system_prompt="Remember what users tell you. Be brief.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    memory=memory,
)

agent.run("My name is Alice.")
result = agent.run("What's my name?")
print(result.output)  # "Your name is Alice."
```

## How It Works

When an agent has memory attached:

1. On each `run()` call, the agent prepends stored messages to the conversation
2. After the agent responds, the new user message and assistant response are added to memory
3. If `max_messages` is reached, the oldest messages are dropped (FIFO)

This gives the agent a sliding window of conversation history.

## Persistence

Save memory to disk and reload it in a new session:

```python
# Save memory to disk
memory.save("memory.json")

# Load in a new session
new_memory = AgentMemory()
new_memory.load("memory.json")

agent = Agent(
    name="assistant",
    system_prompt="Remember what users tell you.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    memory=new_memory,
)

# Agent still remembers Alice
result = agent.run("What's my name?")
print(result.output)  # "Your name is Alice."
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_messages` | `int` | `20` | Maximum number of messages to retain |

---

## Next Steps

- [Agents](index.md) — Core agent documentation
- [Multi-Agent Teams](teams.md) — Supervisor/worker patterns
- [Tracing](../tracing/index.md) — Debug agent execution with traces
