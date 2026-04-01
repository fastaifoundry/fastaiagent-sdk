# Tools

Tools give agents the ability to take actions — call APIs, query databases, run code, or interact with external systems. The SDK provides three tool types and a schema validation system.

## Tool Types

| Type | Use Case | Deep Dive |
|------|----------|-----------|
| [FunctionTool](function-tools.md) | Wrap any Python function | Auto-generated JSON Schema from type hints |
| [RESTTool](rest-tools.md) | Call an HTTP API endpoint | No Python function needed -- just configure URL and method |
| [MCPTool](mcp-tools.md) | Connect to an MCP server | JSON-RPC 2.0 communication |

## Quick Example

```python
from fastaiagent import Agent, FunctionTool, RESTTool, LLMClient

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = Agent(
    name="assistant",
    system_prompt="Use tools to answer questions.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[
        FunctionTool(name="calculate", fn=calculate),
        RESTTool(name="weather", url="https://api.weather.com/v1", method="GET"),
    ],
)

result = agent.run("What is 15% of 230, and what's the weather in Tokyo?")
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

## Sync vs Async

```python
# Sync
result = tool.execute({"query": "test"})

# Async
result = await tool.aexecute({"query": "test"})
```

Both work correctly whether called from sync or async contexts (including Jupyter notebooks).

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

---

## Next Steps

- [FunctionTool](function-tools.md) — Wrap Python functions as tools
- [RESTTool](rest-tools.md) — Call HTTP APIs as tools
- [MCPTool](mcp-tools.md) — Connect to MCP servers
- [Schema Drift Detection](schema-drift.md) — Detect when tool responses change
- [Using Tools with Agents](../agents/tools.md) — How to attach tools to agents
