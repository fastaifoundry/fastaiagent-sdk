# Multi-Agent Teams

FastAIAgent ships two multi-agent topologies:

- **Supervisor / Worker** (this page) — a centralized LLM delegates to specialist workers and synthesizes their outputs. Hub-and-spoke.
- **[Swarm](swarm.md)** — peer-to-peer mesh where each agent decides when to hand off control to another. No coordinator.

Use **Supervisor** when a central LLM should orchestrate and synthesize results. Use **Swarm** when the routing decision belongs to the specialist itself, or when you want a looping workflow like `writer ↔ critic` without a hub in the middle. See the [Swarm vs Supervisor comparison](swarm.md#swarm-vs-supervisor--when-to-use-which) for a full decision matrix.

## Supervisor / Worker Pattern

A supervisor agent delegates tasks to specialized worker agents. This pattern is useful when different parts of a task require different expertise, models, or tool sets.

## Supervisor / Worker Pattern

```python
from fastaiagent import Agent, LLMClient, Supervisor, Worker

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
| `role` | `str` | A short label (e.g., "researcher", "writer"). Used as tool name: `delegate_to_{role}` |
| `description` | `str` | What this worker does -- helps the supervisor route tasks. Defaults to first 200 chars of system prompt |

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

## Passing Context to Workers

`RunContext` flows from the supervisor through to all worker agents and their tools. This lets worker tools access shared runtime dependencies like database connections, user sessions, and configuration.

```python
from dataclasses import dataclass
from fastaiagent import Agent, LLMClient, RunContext, Supervisor, Worker, tool


@dataclass
class TeamState:
    db: DatabaseClient
    user_id: str
    company: str


@tool(name="get_user_tickets")
def get_user_tickets(ctx: RunContext[TeamState], status: str) -> str:
    """Get support tickets for the current user."""
    tickets = ctx.state.db.query("tickets", user_id=ctx.state.user_id, status=status)
    return str(tickets)


@tool(name="get_billing_info")
def get_billing_info(ctx: RunContext[TeamState], account_id: str) -> str:
    """Get billing details."""
    return ctx.state.db.query("billing", account_id=account_id)


support_agent = Agent(name="support", system_prompt="Handle support tickets.", llm=llm, tools=[get_user_tickets])
billing_agent = Agent(name="billing", system_prompt="Handle billing queries.", llm=llm, tools=[get_billing_info])

supervisor = Supervisor(
    name="customer-service",
    llm=llm,
    workers=[
        Worker(agent=support_agent, role="support", description="Manages support tickets"),
        Worker(agent=billing_agent, role="billing", description="Handles billing queries"),
    ],
)

# Context flows to both workers and their tools
ctx = RunContext(state=TeamState(db=get_db(), user_id="u-456", company="Acme"))
result = supervisor.run("Show my open tickets and latest invoice", context=ctx)
```

## Streaming

Stream the supervisor's output in real-time. Worker delegation appears as `ToolCallStart` / `ToolCallEnd` events, and the supervisor's synthesized response streams as `TextDelta` events.

### Async streaming

```python
from fastaiagent import TextDelta
from fastaiagent.llm.stream import ToolCallStart, ToolCallEnd

async for event in supervisor.astream("Help with my order", context=ctx):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStart):
        print(f"\n  [Delegating to {event.tool_name}...]", end="")
    elif isinstance(event, ToolCallEnd):
        print(" [done]", end="")
```

### Sync streaming

Collects the full stream into an `AgentResult`:

```python
result = supervisor.stream("Help with my order", context=ctx)
print(result.output)
```

## Dynamic Instructions

Customize the supervisor's behavior per request using callable prompts. The callable receives the `RunContext` (or `None` if no context is passed).

```python
supervisor = Supervisor(
    name="adaptive-lead",
    llm=llm,
    workers=[support_worker, billing_worker],
    system_prompt=lambda ctx: (
        f"You are the customer service lead for {ctx.state.company}. "
        f"The customer ({ctx.state.user_id}) has a {ctx.state.plan} plan.\n"
        + ("PRIORITY: This is an enterprise customer. Resolve quickly.\n"
           if ctx.state.plan == "enterprise" else "")
        + "Delegate to the appropriate worker and synthesize a helpful response."
    ),
)

ctx = RunContext(state=TeamState(company="Acme", user_id="u-1", plan="enterprise"))
result = supervisor.run("I need help with billing", context=ctx)
```

## API Reference

### `Supervisor`

```python
Supervisor(
    name: str,
    llm: LLMClient | None = None,
    workers: list[Worker] | None = None,
    system_prompt: str | Callable[[RunContext | None], str] = "",
    max_delegation_rounds: int = 3,
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Supervisor name |
| `llm` | `LLMClient \| None` | No | LLM for the supervisor (defaults to OpenAI gpt-4o-mini) |
| `workers` | `list[Worker] \| None` | No | Workers available for delegation |
| `system_prompt` | `str \| Callable` | No | Custom instructions. If omitted, auto-generates from worker descriptions |
| `max_delegation_rounds` | `int` | No | Max delegation rounds (default: 3, translates to `max_iterations * 2`) |

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `run()` | `(input, *, context=None) -> AgentResult` | Synchronous execution |
| `arun()` | `(input, *, context=None) -> AgentResult` | Async execution |
| `stream()` | `(input, *, context=None) -> AgentResult` | Sync streaming (collects result) |
| `astream()` | `(input, *, context=None) -> AsyncGenerator[StreamEvent]` | Async streaming |

All methods accept `context: RunContext | None` which is forwarded to all worker agents and their tools.

### `Worker`

```python
Worker(
    agent: Agent,
    role: str = "",
    description: str = "",
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent` | `Agent` | Yes | The worker agent |
| `role` | `str` | No | Role name (defaults to `agent.name`). Used as tool name: `delegate_to_{role}` |
| `description` | `str` | No | What this worker does (defaults to first 200 chars of system prompt) |

---

## Next Steps

- [Agents](index.md) -- Core agent documentation
- [Context & Dependency Injection](../tools/context.md) -- RunContext details
- [Streaming](../streaming/index.md) -- Streaming architecture
- [Dynamic Instructions](dynamic-instructions.md) -- Callable system prompts
- [Chains](../chains/index.md) -- For more complex multi-step workflows beyond supervisor/worker
