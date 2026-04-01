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

---

## Next Steps

- [RESTTool](rest-tools.md) — Call HTTP APIs as tools
- [MCPTool](mcp-tools.md) — Connect to MCP servers
- [Schema Drift Detection](schema-drift.md) — Detect when tool responses change
- [Tools Overview](index.md) — All tool types at a glance
