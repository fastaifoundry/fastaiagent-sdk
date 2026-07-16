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

## Replay safety (`replay_class`)

Every tool type and the `@tool` decorator accept an optional `replay_class` that
tells [Agent Replay](../replay/index.md) (and the central Replay engine) whether
a recorded tool call may be **re-executed** during a rerun or must have its
**recorded output injected** instead:

| `replay_class` | Meaning | Replay behavior |
|----------------|---------|-----------------|
| `"read_only"` | No side effects; safe to call again (e.g. a GET, a pure lookup). | May be re-executed during replay. |
| `"idempotent"` | Repeating the call with the same args is safe / converges. | Recorded output is injected (not re-executed). |
| `"side_effecting"` | Has observable side effects (writes, payments, emails). | Recorded output is injected; never re-executed. |

```python
from fastaiagent import RESTTool
from fastaiagent.tool import tool

# Marked explicitly by the developer — a GET is NOT auto-classified read_only.
weather = RESTTool(
    name="weather", url="https://api.weather.com/v1", method="GET",
    replay_class="read_only",
)

@tool(name="upsert_user", replay_class="idempotent")
def upsert_user(user_id: str, name: str) -> str:
    ...

@tool(name="charge_card")   # unmarked → resolves to "side_effecting"
def charge_card(amount: int) -> str:
    ...
```

- **Default is `"side_effecting"`** — the safe choice. An unmarked tool is never
  re-executed during replay.
- **Never auto-inferred.** A `GET` `RESTTool` is *not* automatically `read_only`;
  only an explicit mark makes a tool re-executable. Auto-inferring would violate
  the replay-safety invariant.
- **Validated strictly.** A value outside `read_only` / `idempotent` /
  `side_effecting` raises `ValueError` at construction.
- The resolved value is emitted on every tool-call span as
  `fastaiagent.tool.replay_class` (see [Tracing](../tracing/index.md)).

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

> **FunctionTool callables and the ToolRegistry.** Python function objects can't be serialized to JSON, so `FunctionTool.to_dict()` only stores the schema. To make `from_dict()` usable for [Agent Replay](../replay/index.md), every `FunctionTool` that is constructed with a live `fn=...` is automatically registered in the process-wide [`ToolRegistry`](#toolregistry). When `Tool.from_dict()` sees a tool name that's been registered, it returns the live tool (with the callable) instead of a schema-only skeleton. Replay reconstructions in the same process that created the tools "just work".

## ToolRegistry

A process-wide, name-keyed registry that holds live tool callables so replay can rebind them after reconstruction from a trace.

```python
from fastaiagent import FunctionTool, ToolRegistry, tool

# Creating a FunctionTool with a callable auto-registers it
def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: shipped"

t = FunctionTool(name="lookup_order", fn=lookup_order)
assert ToolRegistry.get("lookup_order") is t

# The @tool decorator also auto-registers
@tool(name="echo")
def echo(msg: str) -> str:
    return msg

assert ToolRegistry.get("echo") is not None
```

### API

| Method | Behavior |
|--------|----------|
| `ToolRegistry.register(tool)` | Store a tool by `tool.name`. Last-write-wins — re-registering the same name replaces. Returns the tool. |
| `ToolRegistry.get(name)` | Return the registered tool, or `None`. |
| `ToolRegistry.all()` | Return a copy of the full registry (name → tool). |
| `ToolRegistry.unregister(name)` | Remove by name, returning the removed tool or `None`. |
| `ToolRegistry.clear()` | Drop all entries. Intended for tests. |

### When you need it

You generally do not need to touch `ToolRegistry` directly — auto-registration at `FunctionTool.__init__` covers the common case. You need it explicitly when:

- **Replaying a trace in a different process than the one that created the tools.** Import your tool module in the replay process so the tools get registered at import time, or re-register manually.
- **Unit tests that want to start from a clean slate** — call `ToolRegistry.clear()` in setup.
- **Distinct tools with the same name** — registration is last-write-wins, so give tools unique names if you care about isolation.

### What happens when a tool isn't registered?

When `Tool.from_dict()` reconstructs a `FunctionTool` whose name isn't in the registry, it logs a warning and returns a schema-only skeleton. Calling `tool.aexecute()` on that skeleton returns a `ToolResult(error="No function attached...")`. Replay reruns that invoke the tool will surface the error to the agent (as the tool message content), not crash — the agent can then react to the "tool missing" signal.

## Execution Policy

Every tool type accepts optional keyword args that govern how a call runs — all
off by default, so unset means no change in behavior:

| Option | Effect |
|--------|--------|
| `timeout` | Per-call wall-clock timeout in seconds; the call is cancelled and reported as an error when exceeded. |
| `max_retries` / `retry_delay` | Retry transient failures (exceptions or timeouts) with exponential backoff (`retry_delay * 2**attempt`). |
| `output_type` | Validate/coerce the return value against a Pydantic-compatible type; a mismatch is returned to the model as an error. |
| `validate_args` | *(`FunctionTool` only, default `True`)* Validate/coerce the model's arguments against the function's type hints. |

```python
@tool(timeout=2.0, max_retries=2, output_type=float)
def fx_rate(base: str, quote: str) -> str:
    ...
```

See [FunctionTool → Timeout, Retry & Output Validation](function-tools.md#timeout-retry-output-validation) for details.

## Sync vs Async

```python
# Sync — applies the execution policy above
result = tool.execute({"query": "test"})

# Async — applies the execution policy above
result = await tool.ainvoke({"query": "test"})

# Async, raw — bypasses the policy (no timeout / retry / output validation)
result = await tool.aexecute({"query": "test"})
```

`execute()` and `ainvoke()` are the policy-aware entry points (the agent loop
uses `ainvoke`). Call `aexecute()` directly only to deliberately bypass the
policy. All work from sync or async contexts (including Jupyter notebooks).

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
- [Context & Dependency Injection](context.md) — Pass runtime dependencies to tools
- [RESTTool](rest-tools.md) — Call HTTP APIs as tools
- [MCPTool](mcp-tools.md) — Connect to MCP servers
- [Schema Drift Detection](schema-drift.md) — Detect when tool responses change
- [Using Tools with Agents](../agents/tools.md) — How to attach tools to agents
