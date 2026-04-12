# Context & Dependency Injection

Real-world tools need more than LLM-provided arguments — they need database connections, API clients, user sessions, and configuration. `RunContext` lets you pass runtime dependencies to your tools cleanly and type-safely.

## Basic Usage

```python
from dataclasses import dataclass
from fastaiagent import Agent, LLMClient, RunContext, tool

@dataclass
class Deps:
    db: DatabaseClient
    api_key: str

@tool(name="get_customer")
def get_customer(ctx: RunContext[Deps], customer_id: str) -> str:
    """Fetch customer details."""
    return ctx.state.db.get("customers", customer_id)

agent = Agent(
    name="support",
    system_prompt="You help customers with their accounts.",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[get_customer],
)

# Create context with your runtime dependencies
ctx = RunContext(state=Deps(db=DatabaseClient(), api_key="sk-..."))
result = agent.run("Look up customer C-100", context=ctx)
print(result.output)
```

**How it works:**

- Annotate any tool parameter with `RunContext[YourType]`
- The SDK detects this at tool creation time and excludes it from the LLM schema
- At runtime, pass `RunContext(state=your_object)` to `agent.run()`
- The SDK injects it into every tool that declares it
- Tools without `RunContext` work exactly as before

## Multiple Tools, Same Context

All tools in an agent share the same `RunContext` instance. This is useful for passing shared dependencies like a database connection or user session.

```python
@dataclass
class AppState:
    db: DatabaseClient
    user_id: str
    permissions: list[str]

@tool(name="get_orders")
def get_orders(ctx: RunContext[AppState], status: str) -> str:
    """Get orders filtered by status."""
    return ctx.state.db.query(
        "orders",
        user_id=ctx.state.user_id,
        status=status,
    )

@tool(name="cancel_order")
def cancel_order(ctx: RunContext[AppState], order_id: str) -> str:
    """Cancel an order."""
    if "order:cancel" not in ctx.state.permissions:
        return "Error: insufficient permissions"
    ctx.state.db.update("orders", order_id, status="cancelled")
    return f"Order {order_id} cancelled"

# Both tools receive the same context
agent = Agent(
    name="order-manager",
    system_prompt="You manage customer orders.",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[get_orders, cancel_order],
)

ctx = RunContext(state=AppState(
    db=DatabaseClient(),
    user_id="u-789",
    permissions=["order:read", "order:cancel"],
))
result = agent.run("Cancel order ORD-456", context=ctx)
```

## Mixing Context and Non-Context Tools

Tools with and without `RunContext` work together seamlessly. The SDK only injects context into tools that declare it.

```python
@tool(name="search_kb")
def search_kb(ctx: RunContext[Deps], query: str) -> str:
    """Search the knowledge base (needs DB connection)."""
    return ctx.state.db.search(query)

@tool(name="calculate_total")
def calculate_total(prices: list[float], tax_rate: float) -> float:
    """Calculate total with tax (pure computation, no context needed)."""
    subtotal = sum(prices)
    return round(subtotal * (1 + tax_rate), 2)

agent = Agent(
    name="assistant",
    tools=[search_kb, calculate_total],
    llm=LLMClient(provider="openai", model="gpt-4o"),
)

# search_kb gets ctx, calculate_total doesn't — both work
ctx = RunContext(state=Deps(db=DatabaseClient(), api_key="sk-..."))
result = agent.run("Search for pricing info and calculate total for $10, $20 with 8% tax", context=ctx)
```

## Async Tools with Context

Async tools receive context the same way.

```python
@tool(name="fetch_weather")
async def fetch_weather(ctx: RunContext[Deps], city: str) -> str:
    """Fetch current weather for a city."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.weather.com/v1/{city}",
            headers={"Authorization": f"Bearer {ctx.state.api_key}"},
        )
        return resp.text

# Works with both arun() and astream()
ctx = RunContext(state=Deps(db=db, api_key="weather-key-123"))
result = await agent.arun("What's the weather in Amsterdam?", context=ctx)
```

## Streaming with Context

Context flows through streaming execution identically.

```python
ctx = RunContext(state=Deps(db=DatabaseClient(), api_key="sk-..."))

async for event in agent.astream("Look up customer C-100", context=ctx):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
```

## Testing Tools with Context

`RunContext` makes tools easy to unit test — construct the context with mocks and call the tool directly.

```python
import pytest
from unittest.mock import MagicMock

def test_get_customer():
    mock_db = MagicMock()
    mock_db.get.return_value = {"id": "C-100", "name": "Alice"}

    ctx = RunContext(state=Deps(db=mock_db, api_key="test"))
    result = get_customer.execute({"customer_id": "C-100"}, context=ctx)

    assert result.success
    assert "Alice" in str(result.output)
    mock_db.get.assert_called_once_with("customers", "C-100")
```

## Per-Request Isolation

Each call to `agent.run()` gets its own `RunContext`. This is safe for concurrent requests — no shared mutable state between calls.

```python
# FastAPI example: each request gets isolated context
from fastapi import FastAPI, Depends

app = FastAPI()

@app.post("/chat")
async def chat(message: str, user_id: str = Depends(get_current_user)):
    ctx = RunContext(state=AppState(
        db=get_db_session(),
        user_id=user_id,
        permissions=get_permissions(user_id),
    ))
    result = await agent.arun(message, context=ctx)
    return {"response": result.output}
```

## What Context is NOT

- **Not serialized** — `RunContext` never appears in `agent.to_dict()` or `fa.push()`. It is runtime-only.
- **Not sent to the LLM** — Context parameters are excluded from the tool schema. The LLM never sees them.
- **Not available on the platform** — When you push an agent to FastAIAgent Platform, the platform uses its own dependency injection (ContextBuilder + FastAPI DI). The tool schema is the shared contract between SDK and platform.
- **Not a replacement for ChainState** — `ChainState` carries workflow state between chain nodes. `RunContext` carries runtime dependencies into tool functions. They serve different purposes.

---

## API Reference

### `RunContext`

```
class RunContext(Generic[T])
```

Typed dependency injection container for agent execution.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `state` | `T` | Yes | The runtime state object (dataclass, dict, or any type) |

| Property | Type | Description |
|----------|------|-------------|
| `state` | `T` | The wrapped state object (read-only) |

---

## Using RunContext with `from __future__ import annotations`

If your file uses `from __future__ import annotations` (common in modern Python for deferred type evaluation), there is an important requirement: **`RunContext` and any type referenced in the generic parameter must be imported at module level, not lazily inside a function or method.**

With `from __future__ import annotations`, all annotations become strings that are resolved lazily via `get_type_hints(fn)` against the function's `__globals__` (the defining module's top-level namespace). If `RunContext` is imported inside a function scope, resolution fails silently and the SDK falls back to treating the context parameter as a plain `str` type in the tool's JSON schema. The LLM then sees `ctx: string` and hallucinates a string value instead of the SDK injecting the real context.

```python
# WRONG — RunContext imported lazily, not visible in __globals__
from __future__ import annotations

def make_tools():
    from fastaiagent.agent.context import RunContext  # Too late!

    @tool(name="whoami")
    def whoami(ctx: RunContext[AppState]) -> str:
        return ctx.state.user_id  # ctx is actually a string here!

# CORRECT — RunContext and AppState imported at module level
from __future__ import annotations
from fastaiagent.agent.context import RunContext
from myapp import AppState

@tool(name="whoami")
def whoami(ctx: RunContext[AppState]) -> str:
    return ctx.state.user_id  # ctx is a real RunContext[AppState]
```

This also applies to the dataclass/type used as the generic parameter (`AppState` in the example). Both must be in the module's top-level namespace for `get_type_hints()` to resolve `RunContext[AppState]` correctly.

If you're not using `from __future__ import annotations`, this is not a concern — annotations are evaluated eagerly at class/function definition time and `RunContext` just needs to be importable at that point.

---

## Next Steps

- [FunctionTool](function-tools.md) — Wrap Python functions as tools
- [Using Tools with Agents](../agents/tools.md) — How to attach tools to agents
- [Tools Overview](index.md) — All tool types at a glance
