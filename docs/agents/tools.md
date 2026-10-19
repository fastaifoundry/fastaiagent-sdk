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

By default the tool calls in a single turn run **sequentially**. See
[Parallel tool execution](#parallel-tool-execution) to opt into concurrency.

## Parallel Tool Execution

When the LLM emits several tool calls in one turn, you can run them concurrently
instead of one after another. It's **opt-in** (default off) via `AgentConfig`:

```python
from fastaiagent import Agent, AgentConfig, LLMClient

agent = Agent(
    name="assistant",
    llm=LLMClient(provider="openai", model="gpt-4o"),
    tools=[get_weather, get_flights],
    config=AgentConfig(
        parallel_tools=True,    # run a turn's tool calls concurrently (default: False)
        max_parallel_tools=4,   # cap concurrency with a semaphore (default: 4)
    ),
)
```

A turn's wall-clock becomes the *slowest* tool instead of the *sum* of all of
them. Results are re-ordered by call index, so message history and
`result.tool_calls` stay deterministic regardless of which tool finishes first.

!!! note "When it stays sequential"
    For correctness, the parallel path is used only when none of these
    order/identity-sensitive features are engaged; otherwise the agent
    transparently falls back to sequential execution:

    - a **checkpointer** is configured (per-tool checkpoints / crash recovery / HITL),
    - **middleware** is attached (`wrap_tool` sees calls in order) — including
      the built-in one behind [deferred tools](#tools-loaded-when-needed),
    - **managed governance** is enrolled (`agent_id` is set for policy/approvals).

    Since 1.90.0 the fallback logs one warning per agent per reason, naming it.
    Before, it was silent — and a `Swarm` always attaches middleware to its
    agents, so their `parallel_tools=True` never took effect.

    Tools that share mutable `RunContext` state are your responsibility to make
    concurrency-safe when you enable this.

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
| [Provider-run](../tools/provider-tools.md) | Web search, code execution, file search run by the model's provider (1.91.0) | `web_search()`, `code_execution()`, `file_search(["vs_1"])` from `fastaiagent.tool` |

The first three (and `@tool`) accept an optional `replay_class` (`read_only` / `idempotent` / `side_effecting`, default `side_effecting`) that controls inject-vs-execute during [Agent Replay](../replay/index.md). See [Tools → Replay safety](../tools/index.md#replay-safety-replay_class).

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
        max_iterations=5,       # Max tool-calling loop iterations (default: 10)
        tool_choice="auto",     # "auto", "required", "none"
        parallel_tools=False,   # Run a turn's tool calls concurrently (default: False)
        max_parallel_tools=4,   # Concurrency cap when parallel_tools is on (default: 4)
    ),
)
```

| tool_choice | Behavior |
|-------------|----------|
| `"auto"` | LLM decides whether to use tools (default) |
| `"required"` | LLM must call at least one tool — on the run's **first** turn, so it can still answer |
| `"none"` | LLM cannot call tools, on every turn |

`"auto"` sends nothing (the provider's default). Before 1.90.0 none of the three
reached the model. See [Agent Configuration](index.md#agent-configuration).

## Tools loaded when needed

Every tool definition goes into every request. With a large toolset — an MCP
server's forty tools — that costs context on every turn whether or not the tools
are used. Mark a tool `deferred=True` (since 1.91.0) and its definition is not
sent:

```python
from fastaiagent import Agent, FunctionTool, LLMClient, tool


@tool(deferred=True)
def get_invoice(invoice_id: str) -> str:
    """Look up an invoice: its amount and due date."""
    ...


agent = Agent(
    name="billing",
    llm=LLMClient(provider="gemini", model="gemini-3.8-flash"),
    tools=[
        get_invoice,
        FunctionTool(name="track_shipment", fn=track_shipment, deferred=True),
    ],
)
```

Instead the model is offered one built-in tool, **`load_tools(names)`**, whose
description lists each deferred tool by name and the first line of its
description. When the model loads a tool, its definition is sent from the next
step on, for the rest of the run:

```
turn 0   offered: load_tools          → model calls load_tools(["get_invoice"])
turn 1   offered: get_invoice, load_tools → model calls get_invoice("INV-9")
turn 2   offered: get_invoice, load_tools → model answers
```

- `deferred=` is accepted by `FunctionTool`, `@tool`, `MCPTool` and `RESTTool`.
- A loaded tool is governed and guarded like any other tool.
- A deferred tool the model calls *without* loading it is not run; the model is
  told to call `load_tools` first.
- The loaded set is checkpointed, so a resumed run keeps it.
- `load_tools` is no longer offered once every deferred tool is loaded. It is a
  reserved name while any tool is deferred.
- **Local only.** `deferred` is not part of `Tool.to_dict()`, so a
  plane-distributed copy of the agent sends every tool, as before.
- MCP `expose_tools` clients call deferred tools directly — deferral hides
  definitions from a model, not from a client with its own tool list.
- Deferral uses the middleware machinery, so it turns off
  [parallel tool calls](#parallel-tool-execution) for the agent.

Live-checked on `gpt-5-mini`, `gemini-3.8-flash` and `claude-haiku-4-5`: each
found the tool from the `load_tools` description, loaded it and called it
(`tests/e2e/test_middleware_bundles_live.py`).

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
