# Agent Middleware

Middleware lets you intercept and transform the inputs and outputs of an Agent's run without subclassing `Agent` or wrapping `LLMClient`. Use it for message trimming, budget enforcement, PII redaction, response-rewriting, caching, and any other cross-cutting concern that should compose cleanly.

## Middleware vs Guardrails

Both plug into the Agent pipeline, but they serve different purposes.

| | Middleware | Guardrail |
|---|---|---|
| Purpose | **Transform** messages or responses | **Assert** on content |
| Failure mode | Can raise `StopAgent` to end cooperatively | Raises `GuardrailBlockedError` (blocking) or logs (non-blocking) |
| Composition | Onion — each middleware wraps the next | Flat list — each guardrail runs independently |
| When to use | "Trim history", "redact PII", "cap tool calls", "rewrite refusals" | "Is this PII?", "Is the JSON valid?", "Is this toxic?" |

Input guardrails run **before** middleware's `before_model`; output guardrails run **after** middleware's `after_model` on the final iteration.

## The Three Hooks

```python
from fastaiagent import AgentMiddleware, MiddlewareContext
from fastaiagent.llm.client import LLMResponse
from fastaiagent.llm.message import Message
from fastaiagent.tool.base import Tool, ToolResult


class MyMiddleware(AgentMiddleware):
    name = "my_middleware"

    async def before_model(
        self, ctx: MiddlewareContext, messages: list[Message]
    ) -> list[Message]:
        """Transform messages just before the LLM call."""
        return messages

    async def after_model(
        self, ctx: MiddlewareContext, response: LLMResponse
    ) -> LLMResponse:
        """Inspect or rewrite the LLM response before tool dispatch."""
        return response

    async def wrap_tool(
        self,
        ctx: MiddlewareContext,
        tool: Tool,
        args: dict,
        call_next,
    ) -> ToolResult:
        """Wrap each tool invocation. MUST ``await call_next(tool, args)``
        to proceed, or return a ToolResult to short-circuit."""
        return await call_next(tool, args)
```

All three hooks are optional — override only what you need.

## Hook Ordering

- `before_model` — declaration order (first middleware first)
- `after_model` — **reverse** declaration order (last middleware first)
- `wrap_tool` — onion (first middleware is outermost, calls into inner middleware via `call_next`)

```
Agent.arun()
 ├─ input guardrails
 ├─ middleware[0].before_model
 ├─ middleware[1].before_model
 ├─ middleware[2].before_model
 ├─ LLM.acomplete()
 ├─ middleware[2].after_model
 ├─ middleware[1].after_model
 ├─ middleware[0].after_model
 ├─ for each tool call:
 │   ├─ middleware[0].wrap_tool  (pre)
 │   │   └─ middleware[1].wrap_tool  (pre)
 │   │       └─ middleware[2].wrap_tool  (pre)
 │   │           └─ real tool
 │   │       ← middleware[2].wrap_tool  (post)
 │   │   ← middleware[1].wrap_tool  (post)
 │   ← middleware[0].wrap_tool  (post)
 ├─ (repeat LLM + tool calls per iteration)
 └─ output guardrails
```

## The MiddlewareContext

Each run creates one `MiddlewareContext` that every hook sees:

| Attribute | Description |
|---|---|
| `run_context` | The user-supplied `RunContext`, if any |
| `turn` | Zero-indexed LLM iteration within the run |
| `tool_call_index` | Zero-indexed tool call within the current turn |
| `scratch` | Mutable `dict` shared across all hooks — use to pass data between `before_model` and `after_model`, or to accumulate state across tool calls |
| `agent_name` | Name of the agent running |

Scratch is **per-run**; a fresh dict on every `agent.arun()`.

## Cooperative Stop

To end a run from inside middleware, raise `StopAgent`:

```python
from fastaiagent import StopAgent

class BudgetMiddleware(AgentMiddleware):
    async def before_model(self, ctx, messages):
        if ctx.turn >= 3:
            raise StopAgent("Turn budget exhausted.")
        return messages
```

The agent returns an `AgentResult` whose `output` is the `StopAgent` message. No `GuardrailBlockedError`, no unwinding through the caller — it's a cooperative signal.

Use `StopAgent` for budgets, completion signals, or feature flags. Use `Guardrail` for policy rejections that should surface as errors.

## Built-in Middleware

### `TrimLongMessages(keep_last=20)`

Keeps only the most recent `keep_last` messages plus any leading `SystemMessage`. Cheap alternative to summarization for long-running agents.

```python
from fastaiagent import Agent, TrimLongMessages

agent = Agent(
    name="chatty",
    middleware=[TrimLongMessages(keep_last=30)],
    ...,
)
```

### `ToolBudget(max_calls=10, message="...")`

Raises `StopAgent` once `max_calls` tool invocations have occurred in a single run.

```python
from fastaiagent import Agent, ToolBudget

agent = Agent(
    name="budgeted",
    middleware=[ToolBudget(max_calls=5)],
    ...,
)
```

### `RedactPII(patterns=..., placeholder="[REDACTED]")`

Redacts common PII patterns (email, US phone, SSN, long digit runs) from outbound messages **and** inbound LLM responses. Pass your own regex list for domain-specific patterns.

```python
from fastaiagent import Agent, RedactPII

agent = Agent(
    name="safe",
    middleware=[RedactPII()],
    ...,
)
```

## Writing Your Own Middleware

A `TokenCounter` middleware that tallies tokens across a run:

```python
from fastaiagent import AgentMiddleware

class TokenCounter(AgentMiddleware):
    name = "token_counter"

    async def after_model(self, ctx, response):
        used = response.usage.get("total_tokens", 0)
        ctx.scratch["tokens_total"] = ctx.scratch.get("tokens_total", 0) + used
        return response

agent = Agent(name="counter", llm=..., middleware=[TokenCounter()])
result = await agent.arun("hello")
# Access via a second middleware, or log from after_model.
```

A `PromptCache` that returns a cached response when the message list has been seen before:

```python
import hashlib
import json
from fastaiagent import AgentMiddleware
from fastaiagent.llm.client import LLMResponse


class PromptCache(AgentMiddleware):
    name = "prompt_cache"

    def __init__(self):
        self._cache: dict[str, LLMResponse] = {}

    def _key(self, messages):
        payload = [m.to_openai_format() for m in messages]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    async def before_model(self, ctx, messages):
        ctx.scratch["cache_key"] = self._key(messages)
        return messages

    async def after_model(self, ctx, response):
        self._cache[ctx.scratch["cache_key"]] = response
        return response
```

## Short-Circuiting Tools

A middleware can return a `ToolResult` without calling `call_next`:

```python
from fastaiagent.tool.base import ToolResult

class CacheTool(AgentMiddleware):
    async def wrap_tool(self, ctx, tool, args, call_next):
        cache_key = (tool.name, tuple(sorted(args.items())))
        if cache_key in ctx.scratch.get("tool_cache", {}):
            return ToolResult(output=ctx.scratch["tool_cache"][cache_key])
        result = await call_next(tool, args)
        ctx.scratch.setdefault("tool_cache", {})[cache_key] = result.output
        return result
```

## Interaction with Tracing

Middleware hooks are called inside the agent's root span. Custom spans created inside a middleware hook nest under it — see [Tracing](../tracing/index.md) for the `trace_context` helper.

---

## Next Steps

- [Agent Memory](memory.md) — combine middleware with persistent memory
- [Guardrails](../guardrails/index.md) — assertion layer that runs alongside middleware
- [Dynamic Instructions](dynamic-instructions.md) — system-prompt callables that complement middleware
