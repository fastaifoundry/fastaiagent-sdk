# Using Tools with Agents

Tools let your agent take actions — call APIs, search databases, run calculations. This page covers how to attach tools to agents and how the tool-calling loop works. For deep dives on each tool type, see the [Tools](../tools/index.md) section.

## Passing Tools to an Agent

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

## How the Tool-Calling Loop Works

1. Agent sends messages + tool schemas to the LLM
2. LLM decides to call one or more tools (or respond directly)
3. SDK executes the tools and sends results back to the LLM
4. LLM generates a final response using the tool results
5. This loop repeats up to `max_iterations` times

## The @tool Decorator

For quick tool creation:

```python
from fastaiagent.tool import tool

@tool(name="calculate")
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

# Use directly — it's a FunctionTool
result = calculate.execute({"expression": "2 + 2"})
```

## Tool Types Overview

| Type | Use Case | Example |
|------|----------|---------|
| `FunctionTool` | Wrap any Python function | `FunctionTool(name="calc", fn=my_func)` |
| `RESTTool` | Call an HTTP API | `RESTTool(name="weather", url="https://api.weather.com/v1", method="GET")` |
| `MCPTool` | Connect to MCP server | `MCPTool(name="search", server_url="http://localhost:3000")` |

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

## Controlling Tool Usage

Configure how the agent uses tools via `AgentConfig`:

```python
from fastaiagent import Agent, AgentConfig

agent = Agent(
    name="configured-agent",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[my_tool],
    config=AgentConfig(
        max_iterations=5,     # Max tool-calling loop iterations (default: 10)
        tool_choice="auto",   # "auto", "required", "none"
    ),
)
```

| tool_choice | Behavior |
|-------------|----------|
| `"auto"` | LLM decides whether to use tools (default) |
| `"required"` | LLM must call at least one tool |
| `"none"` | LLM cannot call tools |

## Context & Dependency Injection

Tools that need runtime dependencies (DB connections, API clients, user sessions) can use `RunContext` for clean, type-safe dependency injection. See [Context & Dependency Injection](../tools/context.md) for the full guide.

---

## Next Steps

- [Tools Overview](../tools/index.md) — Full tool type reference
- [FunctionTool](../tools/function-tools.md) — Deep dive into Python function tools
- [RESTTool](../tools/rest-tools.md) — Deep dive into HTTP API tools
- [MCPTool](../tools/mcp-tools.md) — Deep dive into MCP server tools
- [Schema Drift Detection](../tools/schema-drift.md) — Detect when tool responses change
- [Agents](index.md) — Core agent documentation
