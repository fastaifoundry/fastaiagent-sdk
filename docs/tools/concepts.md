# Concepts & Mental Model

This page is the mental model for tools — *why* they exist, *how* the
tool-calling cycle works end to end (the schema round-trip that ties it all
together), *which* tool type to reach for, and how tools compose with agents,
guardrails, and replay. Read it first, then use the reference pages
([Function Tools](function-tools.md), [REST Tools](rest-tools.md),
[MCP Tools](mcp-tools.md), [Context](context.md)) for depth.

## Why tools exist

An LLM can only produce text. On its own it can't look up an order, query a
database, call an API, or do arithmetic reliably — it can only *describe* doing
those things. Almost everything useful an agent does is an **external call to
get context or take an action**, and the model has no way to make those calls
itself.

A **tool** is the bridge. You expose a capability — a Python function, an HTTP
endpoint, an MCP server — and the SDK advertises it to the model as something it
can *request*. The model asks; your code runs; the result flows back into the
conversation. Tools are how a text generator reaches the real world.

## The tool-calling cycle

This is the one mental model to internalize. A tool call is a **round-trip
through a JSON schema**, and every stage has a job:

1. **Define** — you write a typed function (or REST/MCP tool).
2. **Generate schema** — the SDK turns your type hints + docstring into a JSON
   Schema. Primitive-only signatures use a fast hand-rolled path; rich types
   (Pydantic models, `Enum`, `Literal`, `Optional`, nested generics) go through
   Pydantic. Docstring parameter descriptions (Google/NumPy/Sphinx) become field
   descriptions the model reads.
3. **Advertise** — `to_openai_format()` renders each tool as
   `{"type": "function", "function": {...}}`, passed to the LLM in `tools=`.
4. **Model requests** — the LLM replies not with prose but with a `ToolCall`:
   the tool name and JSON arguments.
5. **Validate & coerce** — before your code runs, the JSON arguments are
   validated and coerced back to your Python types via Pydantic. A `tool_call`
   guardrail can inspect the arguments here.
6. **Execute** — your function runs, with `timeout`/`retry` policy applied. If
   it declares a `RunContext` parameter, dependencies are injected (that
   parameter is *excluded* from the advertised schema — the model never sees it).
7. **Validate output** — if you set `output_type`, the return value is coerced
   to it. A `tool_result` guardrail can inspect the output here.
8. **Feed back** — the result becomes a tool message the model reads, then it
   continues the loop (another tool call) or writes the final answer.

```
your typed fn ─▶ JSON Schema ─▶ [wire] ─▶ LLM picks tool + emits JSON args
                                                     │
   final answer ◀─ model reads result ◀─ execute ◀─ validate/coerce back to Python
```

!!! info "Verified against a live run"
    A `@tool` wrapping `open_ticket(ticket: Ticket)` (a Pydantic model with an
    `Enum` field) auto-generated a JSON Schema with `$defs`, `required`, and the
    docstring as the description. `add(a: int, b: int)` called with
    `{"a": "2", "b": "3"}` coerced the strings and returned `5`. And an agent
    asked for two cities' weather emitted **two tool calls in one turn** that
    ran in parallel.

### Errors don't crash the loop

The critical resilience property: a bad tool call is **reported to the model,
not raised to you**. When the model sends arguments that fail validation, the
tool returns `ToolResult(error="Invalid arguments for tool '...': ...")` — the
error text goes back to the model as the tool result so it can correct itself
and try again, rather than blowing up your program.

!!! info "Verified against a live run"
    Calling `add` with `{"a": "not-a-number"}` returned
    `ToolResult(success=False, error="Invalid arguments ...")` — no exception.

## Which tool type

All types share the same lifecycle, validation, execution policy, and
`replay_class`. Pick by where the capability lives:

| Type | Reach for it when | Origin |
|------|-------------------|--------|
| **FunctionTool** (`@tool` / `FunctionTool`) | You have (or can write) a Python function — schema auto-generated from its type hints | `function` |
| **RESTTool** | You want to call an HTTP API directly, no Python wrapper — map args to query/body/path | `rest` |
| **MCPTool** | You're connecting to an MCP server (JSON-RPC), e.g. an existing extension | `mcp` |
| **Knowledge base** (`LocalKB.as_tool()`) | You want retrieval/search as a callable tool | `kb` |
| **Custom `Tool` subclass** | You need special execution or validation the built-ins don't cover | `custom` |
| **Agent/Chain as a tool** (`as_mcp_server()`) | You want to expose a whole agent or chain as a tool for another agent | — |

The `@tool` decorator is the shorthand for the common case; it accepts
`name`, `description`, `replay_class`, `validate_args`, `timeout`,
`max_retries`, `retry_delay`, and `output_type`.

## How tools compose

- **In agents** — pass `Agent(tools=[...])`; the model decides when to call
  them. `tool_choice` (`auto` / `required` / `none`) constrains that decision —
  see [Using Tools with Agents](../agents/tools.md).
- **In parallel** — with `AgentConfig(parallel_tools=True)`, multiple tool calls
  the model requests in one turn run concurrently. The SDK falls back to
  **sequential** execution when order/identity matter — a checkpointer,
  middleware, or managed governance is active — so durable and governed runs
  stay deterministic. Results are re-ordered by index so message history is
  stable regardless.
- **With guardrails** — the two tool positions are gates in the cycle above:
  `tool_call` validates arguments *before* execution (block a destructive call),
  `tool_result` validates output *after* (catch leaked data). See
  [Guardrails](../guardrails/concepts.md).
- **With context** — a `RunContext[Deps]` parameter injects db handles, API
  clients, tenant IDs, etc., isolated per request and hidden from the model.
  See [Context](context.md).
- **In chains** — a chain `tool` node runs a tool as a deterministic step in a
  graph rather than at the model's discretion. See [Chains](../chains/concepts.md).

## Determinism & replay

Every tool carries a `replay_class` — `read_only`, `idempotent`, or
`side_effecting` (the safe default). It's **never auto-inferred**: a GET-style
REST tool is not assumed read-only, because guessing wrong would re-fire a call
during replay. The mark tells the central Replay engine whether it may
re-execute the tool or must inject the recorded output. See
[Agent Replay](../replay/concepts.md) and
[Fidelity Guarantees](../replay/guarantees.md).

## A guided path

1. [`examples/41_agent_tools.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/41_agent_tools.py) — `@tool`, a custom `Tool` subclass, and a KB-as-tool; the three origins in the UI.
2. [`examples/67_tool_docstrings.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/67_tool_docstrings.py) — how Google/NumPy/Sphinx docstrings become parameter descriptions.
3. [`examples/90_parallel_and_pydantic_tools.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/90_parallel_and_pydantic_tools.py) — Pydantic-model args, enums, parallel execution, execution policy.
4. [`examples/23_tool_guardrails.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/23_tool_guardrails.py) — guardrails at `tool_call` and `tool_result`.
5. [`examples/50_agent_dependencies.py`](https://github.com/fastaiagent/fastaiagent-sdk/blob/main/examples/50_agent_dependencies.py) — `RunContext` dependency injection.

## Next steps

- [Tools reference](index.md) — types, `ToolResult`, execution policy, serialization
- [Function Tools](function-tools.md) — schema generation, rich types, docstrings, validation
- [REST Tools](rest-tools.md) · [MCP Tools](mcp-tools.md) · [MCP Server](mcp-server.md)
- [Context](context.md) — dependency injection with `RunContext`
- [Using Tools with Agents](../agents/tools.md) — `tool_choice`, parallel tools, the loop in agent terms
