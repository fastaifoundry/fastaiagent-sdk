# Tools

Tools give agents the ability to take actions — call APIs, query databases, run code, or interact with external systems. The SDK provides three tool types and a schema validation system.

## FunctionTool

Wraps any Python function as a tool. JSON Schema is auto-generated from type hints.

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

### The @tool Decorator

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

### Async Functions

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

### Auto-Schema Generation

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

### Custom Schema

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

## RESTTool

Calls an HTTP API endpoint. No Python function needed — just configure the URL and method.

```python
from fastaiagent import RESTTool

weather_api = RESTTool(
    name="weather",
    description="Get weather forecast for a city",
    url="https://api.weather.example.com/v1/forecast",
    method="GET",
    headers={"X-API-Key": "my-key"},
    body_mapping="query_params",  # Send arguments as URL query parameters
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
    },
)

# Executes: GET https://api.weather.example.com/v1/forecast?city=Paris&days=3
result = await weather_api.aexecute({"city": "Paris", "days": 3})
```

### Body Mapping Options

| Mode | Behavior | Use Case |
|------|----------|----------|
| `query_params` | Arguments sent as URL query parameters | GET requests |
| `json_body` | Arguments sent as JSON request body | POST/PUT requests |
| `path_params` | Arguments replace `{placeholders}` in URL | RESTful paths |

**Path parameters example:**

```python
order_api = RESTTool(
    name="get_order",
    url="https://api.example.com/orders/{order_id}",
    method="GET",
    body_mapping="path_params",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
)
# Executes: GET https://api.example.com/orders/ORD-123
```

## MCPTool

Connects to a Model Context Protocol (MCP) server via JSON-RPC 2.0.

```python
from fastaiagent import MCPTool

file_search = MCPTool(
    name="file_search",
    description="Search files in the codebase",
    server_url="http://localhost:3000/mcp",
    tool_name="search_files",
    auth_token="my-token",  # Optional Bearer token
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "file_pattern": {"type": "string"},
        },
        "required": ["query"],
    },
)

result = await file_search.aexecute({"query": "authentication", "file_pattern": "*.py"})
```

### Discovering MCP Tools

List available tools on an MCP server:

```python
tools = await file_search.discover_tools()
for t in tools:
    print(f"{t['name']}: {t.get('description', '')}")
```

## Using Tools with Agents

Pass tools to an agent — the agent handles the entire tool-calling loop:

```python
from fastaiagent import Agent, FunctionTool, RESTTool, LLMClient

agent = Agent(
    name="assistant",
    system_prompt="Use tools to answer questions.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[
        FunctionTool(name="calculate", fn=lambda expr: str(eval(expr))),
        RESTTool(name="weather", url="https://api.weather.com/v1", method="GET"),
    ],
)

result = agent.run("What is 15% of 230, and what's the weather in Tokyo?")
# Agent calls both tools and combines results
```

## ToolResult

Every tool execution returns a `ToolResult`:

| Field | Type | Description |
|-------|------|-------------|
| `output` | `Any` | The tool's return value |
| `error` | `str \| None` | Error message if execution failed |
| `success` | `bool` | `True` if no error |
| `metadata` | `dict` | Extra info (e.g., HTTP status code for REST tools) |

```python
result = tool.execute({"query": "test"})
if result.success:
    print(result.output)
else:
    print(f"Error: {result.error}")
```

## Schema Drift Detection

Detect when tool responses no longer match their declared schema — catches API changes before they break your agents.

```python
from fastaiagent.tool.schema import validate_schema, detect_drift

# Validate a single response
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
    },
    "required": ["name", "price"],
}

violations = validate_schema(schema, {"name": "Widget", "price": "not-a-number"})
for v in violations:
    print(f"{v.field}: {v.message}")
    # price: Expected number, got string

# Detect drift across multiple responses
report = detect_drift("product_api", schema, [
    {"name": "A", "price": 10.0},
    {"name": "B", "price": 20.0},
    {"name": "C", "price": "free"},  # drift!
])

print(report.drift_detected)   # True
print(report.violations)        # 1 violation
print(report.summary)           # "Drift detected for 'product_api': 1 violations..."
```

## OpenAI Format

Tools are internally converted to OpenAI function-calling format for LLM communication:

```python
fmt = tool.to_openai_format()
# {
#   "type": "function",
#   "function": {
#     "name": "get_weather",
#     "description": "Get current weather for a city.",
#     "parameters": { ... }
#   }
# }
```

This format is automatically converted to Anthropic's `input_schema` format when using the Anthropic provider.

## Serialization

All tool types support roundtrip serialization:

```python
# Serialize
data = tool.to_dict()
# {
#   "name": "weather",
#   "description": "Get weather",
#   "tool_type": "rest_api",
#   "parameters": { ... },
#   "config": {"url": "...", "method": "GET", ...}
# }

# Restore (auto-dispatches to correct class)
restored = Tool.from_dict(data)
# Returns RESTTool if tool_type is "rest_api", FunctionTool if "function", etc.
```

> **Note:** `FunctionTool` serialization stores the schema but NOT the Python function itself. After `from_dict()`, the restored tool will have the correct schema but no executable function. This is by design — functions can't be serialized to JSON.

## Error Handling

```python
from fastaiagent._internal.errors import (
    ToolError,           # Base tool error
    ToolExecutionError,  # Tool failed during execution
    ToolSchemaError,     # Invalid tool schema
    SchemaDriftError,    # Response schema has drifted
)

try:
    result = tool.execute({"bad": "args"})
except ToolExecutionError as e:
    print(f"Tool failed: {e}")
```

## Sync vs Async

```python
# Sync
result = tool.execute({"query": "test"})

# Async
result = await tool.aexecute({"query": "test"})
```

Both work correctly whether called from sync or async contexts (including Jupyter notebooks).
