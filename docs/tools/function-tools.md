# FunctionTool

Wraps any Python function as a tool. JSON Schema is auto-generated from type hints.

## Basic Usage

```python
from fastaiagent import FunctionTool

def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°{units[0].upper()} in {city}"

tool = FunctionTool(name="get_weather", fn=get_weather)

# Auto-generated schema
print(tool.parameters)
# {
#   "type": "object",
#   "properties": {
#     "city": {"type": "string", "description": "city"},
#     "units": {"type": "string", "description": "units"}
#   },
#   "required": ["city"]
# }

# Execute directly
result = tool.execute({"city": "Paris"})
print(result.output)   # "Sunny, 22°C in Paris"
print(result.success)  # True
```

> **Replay safety.** Pass `replay_class="read_only" | "idempotent" | "side_effecting"` (also accepted by the `@tool` decorator) to mark how [Agent Replay](../replay/index.md) treats this tool — default `side_effecting`, never auto-inferred. See [Replay safety](index.md#replay-safety-replay_class).

## The @tool Decorator

Shorthand for creating a `FunctionTool`:

```python
from fastaiagent.tool import tool

@tool(name="calculate")
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# It's a FunctionTool — use it directly or pass to an agent
result = calculate.execute({"expression": "17 * 23"})
print(result.output)  # "391"
```

## Async Functions

Async functions work seamlessly:

```python
import httpx

async def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.text

tool = FunctionTool(name="fetch", fn=fetch_data)
result = await tool.aexecute({"url": "https://example.com"})
```

## Auto-Schema Generation

The SDK inspects type hints to generate JSON Schema:

| Python Type | JSON Schema Type |
|-------------|-----------------|
| `str` | `string` |
| `int` | `integer` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `list[str]` | `array` with `items: {type: string}` |
| `dict` | `object` |

Parameters without defaults are marked as `required`.

### Parameter descriptions from docstrings

As of v1.9.0, parameter descriptions are auto-extracted from any of three
docstring conventions. Detection order is **Google → NumPy → Sphinx**;
the first style with a `param: description` mapping wins.

```python
# Google style
def search(query: str, limit: int = 10) -> str:
    """Search the corpus.

    Args:
        query: The search query string.
        limit: Maximum number of results.
    """

# NumPy style
def search(query: str, limit: int = 10) -> str:
    """Search the corpus.

    Parameters
    ----------
    query : str
        The search query string.
    limit : int, optional
        Maximum number of results.
    """

# Sphinx / reST style
def search(query: str, limit: int = 10) -> str:
    """Search the corpus.

    :param query: The search query string.
    :type query: str
    :param limit: Maximum number of results.
    """
```

All three produce the same `description` field on the generated JSON
schema for `query` and `limit`. If no docstring or no recognised section
is present, descriptions default to the parameter name.

## Custom Schema

Override auto-generation when you need precise control:

```python
tool = FunctionTool(
    name="search",
    fn=search_fn,
    description="Search the knowledge base",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "filters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["docs", "faq", "guides"]},
                },
            },
        },
        "required": ["query"],
    },
)
```

## Context & Dependency Injection

Tools that need runtime dependencies (DB connections, API clients, user sessions) can declare a `RunContext` parameter. The SDK injects it automatically and hides it from the LLM.

```python
from fastaiagent import RunContext, tool

@tool(name="get_customer")
def get_customer(ctx: RunContext[MyDeps], customer_id: str) -> str:
    """Fetch customer details."""
    return ctx.state.db.get("customers", customer_id)
```

See [Context & Dependency Injection](context.md) for the full guide.

---

## Next Steps

- [Context & Dependency Injection](context.md) — Pass runtime dependencies to tools
- [RESTTool](rest-tools.md) — Call HTTP APIs as tools
- [MCPTool](mcp-tools.md) — Connect to MCP servers
- [Schema Drift Detection](schema-drift.md) — Detect when tool responses change
- [Tools Overview](index.md) — All tool types at a glance
