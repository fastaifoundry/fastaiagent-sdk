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

### Rich types: Pydantic models, Enum, Literal, Optional

As of v1.41.0, parameters typed with anything richer than the primitives above
get a full, self-contained JSON Schema generated via Pydantic — so the model can
fill **structured** arguments, not just flat strings. This covers Pydantic
`BaseModel` subclasses (including nested ones), `Enum`, `Literal[...]`,
`Optional[...]` / `X | None`, and typed collections like `dict[str, Model]`.

```python
import enum
from pydantic import BaseModel

class Priority(str, enum.Enum):
    low = "low"
    high = "high"

class Ticket(BaseModel):
    title: str
    body: str
    priority: Priority

def create_ticket(ticket: Ticket) -> str:
    """File a support ticket.

    Args:
        ticket: the ticket to create
    """
    ...

tool = FunctionTool(name="create_ticket", fn=create_ticket)
# tool.parameters is now:
# {
#   "type": "object",
#   "properties": {
#     "ticket": {"$ref": "#/$defs/Ticket", "description": "the ticket to create"}
#   },
#   "required": ["ticket"],
#   "$defs": {
#     "Priority": {"type": "string", "enum": ["low", "high"]},
#     "Ticket": {
#       "type": "object",
#       "properties": {
#         "title": {"type": "string"},
#         "body": {"type": "string"},
#         "priority": {"$ref": "#/$defs/Priority"}
#       },
#       "required": ["title", "body", "priority"]
#     }
#   }
# }
```

The generated schema (with `$defs`/`$ref`) is passed through verbatim to both
OpenAI (strict function calling) and Anthropic (`input_schema`).

!!! note "Non-breaking"
    Signatures whose parameters are **all** primitives (the table above) still
    produce byte-identical schemas — this path only engages when a rich type is
    present. The tool receives the model's JSON arguments as a plain `dict`;
    validate/coerce inside your function if you want a `Ticket` instance
    (e.g. `Ticket(**ticket)`).

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

## Argument Validation & Coercion

As of v1.41.0, `FunctionTool` **validates and coerces** the model's arguments
against your function's type hints before calling it — on by default, no config.
So a parameter typed `n: int` receives a real `int` (not the string `"5"` the
wire may carry), and a Pydantic-model parameter arrives as a validated instance:

```python
def file_ticket(ticket: Ticket) -> str:
    # `ticket` is already a validated Ticket — no `Ticket(**ticket)` boilerplate.
    return f"Filed {ticket.title!r} at {ticket.priority.value} priority"
```

If the model sends malformed arguments (wrong type, bad enum value, missing
required field), the tool returns a clear error **back to the model** instead of
raising — so the agent can correct itself and retry. Opt out with
`FunctionTool(..., validate_args=False)` (or `@tool(validate_args=False)`) to
receive the raw JSON `dict`.

## Timeout, Retry & Output Validation

Every tool (any type) accepts an optional execution policy — plain keyword args,
all off by default:

```python
@tool(
    name="fx_rate",
    timeout=2.0,      # per-call wall-clock timeout (seconds)
    max_retries=2,    # retry on failure, exponential backoff (retry_delay * 2**n)
    retry_delay=0.5,  # base backoff in seconds
)
def fx_rate(base: str, quote: str) -> float:
    ...
```

- **`timeout`** — the call is cancelled and reported as an error after N seconds.
- **`max_retries` / `retry_delay`** — transient failures (exceptions or timeouts)
  are retried with exponential backoff before the error surfaces.

**`output_type`** is a separate, optional knob that validates/coerces the tool's
*return value*. It's only useful when the raw return isn't already the clean type
you want to hand the model — for example, parsing a `dict` into a validated model:

```python
@tool(output_type=Ticket)      # the dict return is parsed + validated into a Ticket
def draft_ticket(subject: str) -> dict:
    return {"title": subject, "body": "...", "priority": "high"}
```

`output_type` accepts any Pydantic-compatible type (`int`, `list[str]`, a
`BaseModel`, …); a value that can't be coerced is returned to the model as an
error instead of passed downstream. If your function already returns the right
type (annotate it, e.g. `-> float`), you don't need `output_type` at all.

These all work on `RESTTool(...)` and `MCPTool(...)` via the same keyword args.

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
