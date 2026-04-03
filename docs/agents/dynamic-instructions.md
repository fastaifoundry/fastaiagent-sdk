# Dynamic Instructions

Static system prompts treat every user and request the same. Dynamic Instructions let the system prompt adapt per request using a callable that receives the `RunContext`.

## Basic Usage

```python
from datetime import date
from fastaiagent import Agent, LLMClient, RunContext

agent = Agent(
    name="support",
    system_prompt=lambda ctx: (
        f"You are a support agent for {ctx.state.company_name}. "
        f"The customer's name is {ctx.state.user_name}. "
        f"Their subscription: {ctx.state.plan}. "
        f"Today is {date.today()}."
    ),
    llm=LLMClient(provider="openai", model="gpt-4o"),
)

ctx = RunContext(state=CustomerState(
    company_name="Acme Corp",
    user_name="Alice",
    plan="enterprise",
))
result = agent.run("I can't access the admin panel", context=ctx)
```

The callable is invoked fresh on every call to `run()`, `arun()`, or `astream()`. The agent instance is reusable across requests — only the context changes.

## Static Prompts Still Work

Dynamic Instructions is fully backward compatible. String prompts work exactly as before.

```python
# This is unchanged
agent = Agent(name="bot", system_prompt="You are helpful.")
result = agent.run("Hello")
```

## Handling Missing Context

If your agent might be called with or without context, handle `None` in the callable:

```python
agent = Agent(
    name="flexible",
    system_prompt=lambda ctx: (
        f"You help {ctx.state.user_name} with their {ctx.state.plan} plan."
        if ctx else
        "You are a general-purpose assistant."
    ),
    llm=LLMClient(provider="openai", model="gpt-4o"),
)

# With context — personalized prompt
ctx = RunContext(state=UserState(user_name="Alice", plan="pro"))
result = agent.run("Help me", context=ctx)

# Without context — fallback prompt
result = agent.run("Help me")
```

## Feature Flags and A/B Testing

Dynamic Instructions let you control agent behavior via feature flags without rebuilding the agent.

```python
agent = Agent(
    name="assistant",
    system_prompt=lambda ctx: (
        "You are a concise assistant. Keep responses under 2 sentences."
        if ctx and ctx.state.feature_flags.get("concise_mode")
        else "You are a thorough assistant. Provide detailed explanations."
    ),
    llm=LLMClient(provider="openai", model="gpt-4o"),
)
```

## Using Named Functions

Lambdas work for short prompts. For complex logic, use a named function:

```python
def build_support_prompt(ctx: RunContext | None) -> str:
    if ctx is None:
        return "You are a support agent."

    user = ctx.state
    lines = [
        f"You are a support agent for {user.company}.",
        f"Customer: {user.name} ({user.email})",
        f"Plan: {user.plan_tier}",
    ]

    if user.plan_tier == "enterprise":
        lines.append("This is a high-priority customer. Escalate unresolved issues.")

    if user.open_tickets > 3:
        lines.append(f"Note: customer has {user.open_tickets} open tickets. Be empathetic.")

    return "\n".join(lines)

agent = Agent(
    name="support",
    system_prompt=build_support_prompt,
    llm=LLMClient(provider="openai", model="gpt-4o"),
)
```

## Combining with Context Tools

Dynamic Instructions works naturally with [Context & Dependency Injection](../tools/context.md). Both the prompt and tools receive the same `RunContext`:

```python
from dataclasses import dataclass
from fastaiagent import Agent, LLMClient, RunContext, tool

@dataclass
class AppState:
    user_name: str
    plan_tier: str
    db: DatabaseClient

@tool(name="get_orders")
def get_orders(ctx: RunContext[AppState], status: str) -> str:
    """Get orders for the current user."""
    return ctx.state.db.query("orders", user=ctx.state.user_name, status=status)

agent = Agent(
    name="support",
    system_prompt=lambda ctx: f"You help {ctx.state.user_name} ({ctx.state.plan_tier} plan).",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[get_orders],
)

ctx = RunContext(state=AppState(user_name="Alice", plan_tier="pro", db=get_db()))
result = agent.run("Show my open orders", context=ctx)
```

## Streaming

Context flows through streaming execution identically:

```python
ctx = RunContext(state=AppState(user_name="Alice", plan_tier="pro", db=get_db()))

async for event in agent.astream("Show my open orders", context=ctx):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
```

## Push Limitation

Callable prompts are SDK-runtime-only. They cannot be pushed to the platform because Python functions aren't serializable.

```python
# This raises ValueError with a clear message
agent = Agent(name="bot", system_prompt=lambda ctx: "dynamic")
agent.to_dict()  # ValueError: callable system_prompt cannot be serialized

# To push, use a static string
agent_pushable = Agent(name="bot", system_prompt="You are helpful.")
fa.push(agent_pushable)  # Works
```

For dynamic prompts on the platform, use the **Prompt Registry** with `{{variables}}` template syntax and `{{@fragments}}` composition.

## Per-Request Isolation

Each call to `run()` resolves the callable independently with its own `RunContext`. This is safe for concurrent requests:

```python
from fastapi import FastAPI, Depends

app = FastAPI()

@app.post("/chat")
async def chat(message: str, user_id: str = Depends(get_current_user)):
    ctx = RunContext(state=AppState(
        user_name=get_user_name(user_id),
        plan_tier=get_plan(user_id),
        db=get_db_session(),
    ))
    result = await agent.arun(message, context=ctx)
    return {"response": result.output}
```

## API Reference

### `Agent` constructor — updated `system_prompt` parameter

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `system_prompt` | `str \| Callable[[RunContext \| None], str]` | No | Static string or callable that receives `RunContext` and returns a string |

The callable signature is:

```python
(context: RunContext[T] | None) -> str
```

The callable receives the `RunContext` passed to `agent.run()`. If `agent.run()` is called without context, the callable receives `None`. The callable must always return a `str`.

---

## Next Steps

- [Context & Dependency Injection](../tools/context.md) — Pass runtime dependencies to tools
- [Agents Overview](index.md) — Full agent documentation
- [Tools](tools.md) — Deep dive into using tools with agents
