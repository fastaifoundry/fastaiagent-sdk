# Multi-Agent Teams

A supervisor agent delegates tasks to specialized worker agents. This pattern is useful when different parts of a task require different expertise, models, or tool sets.

## Supervisor / Worker Pattern

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.agent import Supervisor, Worker

researcher = Agent(
    name="researcher",
    system_prompt="Research topics thoroughly. Return facts only.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

writer = Agent(
    name="writer",
    system_prompt="Write clear, concise content from research.",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"),
)

supervisor = Supervisor(
    name="team-lead",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    workers=[
        Worker(agent=researcher, role="researcher", description="Finds facts"),
        Worker(agent=writer, role="writer", description="Writes content"),
    ],
)

result = supervisor.run("Write a summary of AI trends in 2025")
print(result.output)
```

## How It Works

1. The supervisor receives the user's request
2. It decides which worker(s) to delegate to, based on the task and worker descriptions
3. Each worker executes independently with its own tools, LLM, and guardrails
4. The supervisor combines worker outputs into a final response

## Worker Configuration

Each `Worker` wraps an agent with metadata that helps the supervisor decide when to use it:

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent` | `Agent` | The worker agent instance |
| `role` | `str` | A short label (e.g., "researcher", "writer") |
| `description` | `str` | What this worker does -- helps the supervisor route tasks |

## Mixed Providers

Workers can use different LLM providers. The supervisor picks the right worker regardless of backend:

```python
supervisor = Supervisor(
    name="team-lead",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    workers=[
        Worker(
            agent=Agent(name="fast-agent", llm=LLMClient(provider="openai", model="gpt-4.1-mini"), system_prompt="Quick answers."),
            role="quick-responder",
            description="Handles simple, fast questions",
        ),
        Worker(
            agent=Agent(name="deep-agent", llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"), system_prompt="Thorough analysis."),
            role="analyst",
            description="Handles complex analysis tasks",
        ),
    ],
)
```

---

## Next Steps

- [Agents](index.md) — Core agent documentation
- [Agent Memory](memory.md) — Give agents conversation memory
- [Chains](../chains/index.md) — For more complex multi-step workflows beyond supervisor/worker
- [Platform Sync](../platform/index.md) — Push teams to the platform
