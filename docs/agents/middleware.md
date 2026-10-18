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

Input guardrails run **before** middleware's `before_model` — and **after** `before_run`, so they judge the input as middleware rewrote it; output guardrails run **after** middleware's `after_model` on the final iteration.

## The Hooks

Eight hooks, all optional. Three act on one model call or one tool call; five
(since 1.91.0) act on the run as a whole or on every model call:

| Hook | When | Can do |
|---|---|---|
| `before_run(ctx, input)` | once per run (`run`, `astream`, a resume), before the input guardrails | rewrite the input; `raise StopAgent` to end the run before any model call |
| `before_model(ctx, messages)` | before each model call of the tool loop | transform the messages; change `ctx.llm` / `ctx.tools` for this call |
| `wrap_model(ctx, messages, call_next)` | around **every** model call — the structured-output and guardrail re-asks too | retry, time, cache (answer without calling the model), swap `ctx.llm` |
| `after_model(ctx, response)` | after each model call, before tools run | inspect or rewrite the response |
| `wrap_tool(ctx, tool, args, call_next)` | around each tool call | short-circuit, retry, budget |
| `after_run(ctx, result)` | once, when the run finishes (not on a pause or a failure) | inspect or replace the `AgentResult` |
| `on_error(ctx, error)` | once, when the run fails | observe — log, count, alert. It cannot swallow the error |
| `on_stream_event(ctx, event)` | each event `astream` yields | rewrite it, or return `None` to drop it |

The first three hooks, in full:

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

Override only what you need. The run-level hooks have the same shape:

```python
from fastaiagent import AgentMiddleware


class RunHooks(AgentMiddleware):
    async def before_run(self, ctx, input):
        return input                      # or a rewrite; raise StopAgent to end

    async def wrap_model(self, ctx, messages, call_next):
        return await call_next(messages)  # MUST call it to reach the model

    async def after_run(self, ctx, result):
        return result                     # the AgentResult, or a replacement

    async def on_error(self, ctx, error):
        ...                               # observe; the run still fails

    async def on_stream_event(self, ctx, event):
        return event                      # or a replacement, or None to drop
```

## Hook Ordering

- `before_run`, `before_model` — declaration order (first middleware first)
- `after_model`, `after_run`, `on_error`, `on_stream_event` — **reverse** declaration order (last middleware first)
- `wrap_model`, `wrap_tool` — onion (first middleware is outermost, calls into inner middleware via `call_next`)

```
Agent.arun()
 ├─ middleware[0..2].before_run
 ├─ input guardrails
 ├─ middleware[0].before_model
 ├─ middleware[1].before_model
 ├─ middleware[2].before_model
 ├─ middleware[0].wrap_model → [1] → [2] → LLM.acomplete()
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
 ├─ output guardrails (a reask goes through wrap_model too)
 └─ middleware[2..0].after_run
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
| `execution_id` | The run's id — what its checkpoints and resumes use *(since 1.91.0)* |
| `llm` | The client **this** model call uses. Set it in `before_model` or `wrap_model` to send one call to another model *(since 1.91.0)* |
| `tools` | The tools **this** model call offers. Filter or extend the list; a tool added here can be called on this turn *(since 1.91.0)* |
| `usage` | Read-only `RunUsage(tokens, cost, cost_known, calls)`: the run's spend so far — the same running total `AgentResult.cost` and `cost_limit` read *(since 1.91.0)* |
| `state` | **This** middleware's own JSON-safe dict, saved in checkpoints and restored on resume and fork — see [A middleware as a bundle](#a-middleware-as-a-bundle) *(since 1.91.0)* |

Scratch is **per-run**; a fresh dict on every `agent.arun()`. A **resume** keeps
one context from start to finish — the tool call it re-runs and the loop it
continues see the same `scratch` (before 1.91.0 they saw two different ones).

`ctx.llm` and `ctx.tools` are reset to the agent's own before every turn, so a
change lasts one call. A structured-output or guardrail re-ask offers no tools.

## Wrapping a model call

`wrap_model` is to the model what `wrap_tool` is to a tool. It sees **every**
model call — each turn of the tool loop, and the structured-output and guardrail
re-asks, which before 1.91.0 bypassed middleware entirely.

```python
import time
from fastaiagent import AgentMiddleware
from fastaiagent.llm.client import LLMResponse


class TimeModelCalls(AgentMiddleware):
    async def wrap_model(self, ctx, messages, call_next):
        started = time.monotonic()
        try:
            return await call_next(messages)
        finally:
            ctx.scratch.setdefault("model_ms", []).append(
                int((time.monotonic() - started) * 1000)
            )


class RetryOnce(AgentMiddleware):
    async def wrap_model(self, ctx, messages, call_next):
        try:
            return await call_next(messages)
        except Exception:
            return await call_next(messages)


class AnswerFromCache(AgentMiddleware):
    def __init__(self):
        self.cache: dict[str, LLMResponse] = {}

    async def wrap_model(self, ctx, messages, call_next):
        key = "\n".join(str(m.content) for m in messages)
        if key in self.cache:
            return self.cache[key]        # the model is never called
        response = await call_next(messages)
        self.cache[key] = response
        return response
```

**One call on another model.** Set `ctx.llm` — for example, let a cheaper model
write the final answer once the tools have run:

```python
from fastaiagent import LLMClient

cheap = LLMClient(provider="gemini", model="gemini-3.5-flash-lite")


class FinalAnswerOnCheapModel(AgentMiddleware):
    async def before_model(self, ctx, messages):
        if ctx.turn >= 1:
            ctx.llm = cheap
        return messages
```

**One call with other tools.** Set `ctx.tools` to offer a subset (or extra
tools) on this call only.

**On `astream`** the model's deltas stream to the caller as they arrive while
your `wrap_model` waits on `call_next`. A response you return without calling
the model is streamed as one `TextDelta`; a retry after deltas streams the second
attempt too. Rewriting the response after `call_next` changes what the run keeps
and what `stream()` returns, not the deltas already sent — the same rule as
`after_model`.

## Around the run

`before_run` and `after_run` bracket the run once — a tool loop of any length.

```python
class TagTicket(AgentMiddleware):
    async def before_run(self, ctx, input):
        return f"[ticket {ctx.run_context.state.ticket}] {input}"

    async def after_run(self, ctx, result):
        result.output = result.output.strip()
        return result
```

- `before_run` runs **before the input guardrails**, so they judge the rewritten
  input. Raising `StopAgent` there ends the run before any model call; the result
  carries the stop message and `stop_reason`.
- `before_run` also fires when a run is **resumed**, with the run's original
  input; there its return value is not used — the saved history is what the run
  continues from.
- `after_run` fires once on success — not when the run pauses (a resumed run
  fires it when it finishes) and not when it fails. On `astream` the reply has
  already been streamed; what `after_run` returns is what `stream()` returns.
  Memory keeps the agent's own reply, not the rewrite.

## When a run fails

Middleware is **fail-closed**: a hook that raises anything other than `StopAgent`
fails the run, exactly as a failing tool loop does. Since 1.91.0, on every path
(`run`, `astream`, a resume):

- `on_error(ctx, error)` fires once, in reverse order, with the exception the run
  is failing with. It observes; the run still raises `error`. An exception raised
  inside `on_error` is logged and never replaces `error`.
- A run with a checkpointer gets a `failed` run-end marker (see
  [Durability concepts](../durability/concepts.md)).
- A `StopAgent` and a pause are not failures — `on_error` does not fire for them.

```python
import logging

class AlertOnFailure(AgentMiddleware):
    async def on_error(self, ctx, error):
        logging.getLogger("agents").error(
            "run %s failed: %s", ctx.execution_id, type(error).__name__
        )
```

`Reflect` is the one built-in that **fails open** by design — a review that
errors keeps the draft. Since 1.91.0 it says so: a warning, a count in
`ctx.scratch["reflect_skipped"]` and a `reflect.skipped` span event, so an
unreviewed answer is never mistaken for an approved one.

## Filtering the stream

`on_stream_event` sees each event `astream` is about to yield:

```python
from fastaiagent.llm.stream import TextDelta, Usage


class HideUsage(AgentMiddleware):
    async def on_stream_event(self, ctx, event):
        if isinstance(event, Usage):
            return None                   # dropped
        return event
```

`StreamDone` and `Paused` are always delivered — a consumer waits on them, so a
`None` for those is ignored.

All of these together, on a support agent, run live or `--offline`:
[`examples/108_middleware_run_hooks.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/108_middleware_run_hooks.py).

## A middleware as a bundle

A policy is usually three things: the tools that act, the text that tells the
model when to use them, and a rule that holds across the run. A middleware can
carry all three (since 1.91.0):

```python
from fastaiagent import AgentMiddleware, FunctionTool, StopAgent


def issue_refund(order_id: str) -> str:
    """Issue a refund for an order."""
    ...


class Refunds(AgentMiddleware):
    name = "refunds"
    tools = [FunctionTool(name="issue_refund", fn=issue_refund)]
    instructions = "Refunds: orders under 30 days old only. Use issue_refund."

    def __init__(self, max_refunds: int = 2):
        self.max_refunds = max_refunds

    async def wrap_tool(self, ctx, tool, args, call_next):
        if tool.name == "issue_refund":
            done = ctx.state.get("refunds", 0)
            if done >= self.max_refunds:
                raise StopAgent("Refund limit reached for this conversation.", reason="refund_limit")
            ctx.state["refunds"] = done + 1
        return await call_next(tool, args)
```

- **`tools`** are added to the run's tools — offered to the model, dispatched,
  governed and guarded like the agent's own. They are not added to `agent.tools`
  (so not to `Agent.to_dict()`). A name that clashes with the agent's tools or
  another middleware's raises `ValueError` when the `Agent` is built.
- **`instructions`** is a string or a `callable(ctx) -> str` (a method works).
  All middleware's instructions form **one** system message right after the
  system prompt and before memory, in declaration order — a leading system
  message, so `TrimLongMessages` keeps it.
- **`ctx.state`** is this middleware's own dict — each middleware sees its own.
  It is **saved in checkpoints** under `state_snapshot["middleware_state"]` and
  restored when the run is resumed or forked, so a limit or a counter survives a
  pause. It must be JSON-safe: anything else raises `TypeError` at the next
  checkpoint, naming the middleware, rather than being lost on resume. Keep
  objects that should not be saved in `ctx.scratch`. State is keyed by the
  middleware's `name` (class name if unset; `#2`, `#3` for repeats), so resume
  with the same middleware list. A checkpoint carries `middleware_state` only
  when some middleware keeps state.

## Middleware on teams and chains

`Chain`, `Swarm` and `Supervisor` take `middleware=` too (since 1.91.0), for a
policy that should hold across the whole team without copying it onto every
agent:

```python
from fastaiagent import Supervisor, Swarm, ToolBudget, Worker
from fastaiagent.chain import Chain

swarm = Swarm(name="desk", agents=[triage, billing], entrypoint="triage",
              middleware=[ToolBudget(max_calls=8), Audit()])
sup = Supervisor(name="lead", workers=[Worker(agent=billing)], middleware=[Audit()])
chain = Chain("flow", middleware=[Audit()])
```

| Container | Where its middleware runs |
|---|---|
| `Swarm` | every hop, **outside** each agent's own middleware |
| `Supervisor` | the supervisor's own loop — its `delegate_to_*` calls; workers keep theirs. With `routing="decisions"` there is no supervisor loop, so it runs on the routed worker's run |
| `Chain` | outermost around every **agent node** (outside the node agent's own, without changing that agent), and `wrap_tool` around every **tool node** |

A container's middleware keeps **one** `ctx.state` for the container's run:
a swarm-level `ToolBudget(max_calls=8)` counts the tools of every hop together,
and a Swarm resume picks the count up again from the checkpoints. (On a stream a
handoff is detected from the model's tool call and not run as a tool, so it is
not counted there.) A Chain's middleware state lasts one execution; `resume`
starts it afresh. A tool name a container's middleware brings that clashes with
an agent's raises `ValueError` when the container is built.

Every run's root span names its middleware in order in one attribute,
`fastaiagent.middleware` — see [Tracing](../tracing/index.md).

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

The agent returns an `AgentResult` whose `output` is the `StopAgent` message and whose `stop_reason` is the stop's `reason` (`StopAgent("…", reason="budget")`; `"stopped"` when none was given). No `GuardrailBlockedError`, no unwinding through the caller — it's a cooperative signal.

**A stopped run is not re-driven.** With `output_type` set, the stop message does not parse, and an output guardrail may reject it; neither calls the model again. Output guardrails still judge the stop message, and a failing `reask` rule blocks instead of re-asking. **`astream` yields the stop message** as one `TextDelta` before `StreamDone`, so a streaming caller sees the same output `run` returns.

!!! warning "Changed in 1.90.0"
    Before 1.90.0 the reason was dropped, and with `output_type` set the
    structured re-ask answered the stop message: it called the model again —
    outside middleware, so the `ToolBudget` or cost stopper never saw the call —
    and the model's answer **replaced** the stop. A streaming caller got no text.

Use `StopAgent` for budgets, completion signals, or feature flags. Use `Guardrail` for policy rejections that should surface as errors.

### What the history looks like after a stop

`StopAgent` raised from `before_model` or `after_model` ends the turn on a
boundary: nothing is half-done.

`StopAgent` raised from `wrap_tool` does not. The assistant message at the top of
that turn already declares **every** tool call the model asked for, and the stop
lands partway through dispatching them. Both OpenAI and Anthropic reject a
history where an assistant `tool_calls` message is not answered call-for-call, so
the agent closes the turn before returning (since 1.67.0):

* the call that was in flight is answered with whatever it actually produced — a
  stopper that fires *before* `call_next` (`ToolBudget` does) produces nothing;
* every call the stop skipped is answered with a short note saying it was not
  run.

The skipped calls are **not dispatched**. Their side effects never happen, and
the note says so in terms the model can act on, rather than leaving it to assume
a silent success. If your middleware needs the remaining calls to run, return a
`ToolResult` from `wrap_tool` instead of raising — a short-circuit ends one call,
`StopAgent` ends the run.

## Built-in Middleware

### `TrimLongMessages(keep_last=20)`

Keeps only the most recent `keep_last` messages plus **every leading system message** — the system prompt and the context memory blocks render right after it (a pinned `StaticBlock`, a summary, known facts). Cheap alternative to summarization for long-running agents.

!!! warning "Changed in 1.90.0"
    Before 1.90.0 only the first message was kept, so on a long run the trim
    dropped the user's pinned memory. And because the trimmer returned a new
    list, a structured or guardrail re-ask after a trim re-sent the **untrimmed**
    history without its latest turns. Both are fixed; a `before_model` hook may
    return a new list or edit the one it was given, and either is what the rest
    of the run sees.

```python
from fastaiagent import Agent, LLMClient, TrimLongMessages

agent = Agent(
    name="chatty",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    middleware=[TrimLongMessages(keep_last=30)],
)
```

**Trimming preserves tool-call pairing.** `keep_last` is a message count, so the
window can land between an assistant message carrying `tool_calls` and the tool
results that answer it. Since 1.67.0 the tail is re-balanced before it is
returned:

* a tool result whose parent call was trimmed away is **dropped**;
* a tool call whose results were trimmed away is **answered** with a note saying
  the result is no longer in the window (the tool did run — only its result fell
  out of the window).

That can leave slightly more than `keep_last` messages. A provider 400 is the
more expensive of the two.

### `ToolBudget(max_calls=10, message="...")`

Raises `StopAgent` once `max_calls` tool invocations have occurred in a single run.

```python
from fastaiagent import Agent, LLMClient, ToolBudget

agent = Agent(
    name="budgeted",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    middleware=[ToolBudget(max_calls=5)],
)
```

The budget is checked *before* `call_next`, so the call that trips it never runs
— see [What the history looks like after a stop](#what-the-history-looks-like-after-a-stop).

**It counts across a resume.** The count is kept in `ctx.state`, which is saved
in every checkpoint, so a run that pauses for approval (or crashes) and is resumed
keeps counting from where it was. Before 1.91.0 it lived in `ctx.scratch` and
every resume started a fresh budget.

**With `output_type`.** The stop message becomes the run's output and the run's
`stop_reason` is `"tool_budget"`. It is not re-asked to fit your schema — since
1.90.0 a stopped run is never re-driven — so `parsed` is `None`.

### `RedactPII(patterns=..., placeholder="[REDACTED]", entities=...)`

Redacts PII from outbound messages **and** inbound LLM responses.

```python
from fastaiagent import Agent, LLMClient, RedactPII

agent = Agent(
    name="safe",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    middleware=[RedactPII()],
)
```

By default it delegates to the **same detector** behind [`no_pii()`](../guardrails/index.md#no_pii),
the `pii` guardrail type and the `PIILeakage` scorer — email, US phone, SSN and
credit cards, with cards **Luhn-validated**. Narrow it with
`entities=("email", "ssn")`.

!!! warning "Behaviour change in 1.63.0 — it redacts *less*, and that is the fix"
    Until 1.63.0 this middleware carried its own private copy of the PII regexes,
    and the copy had drifted: its card pattern was a bare
    `\b(?:\d[ \-]?){13,19}\b` with **no Luhn check**. Any 13–19 digit run —
    an order number, an invoice id, an IMEI — was redacted as a credit card.

    That corruption was what the model saw and what was replayed in a guardrail
    re-ask. If you were relying on long digit runs being masked, pass an
    explicit `patterns=` list.

**What it redacts, and what it doesn't.** `RedactPII` redacts what the model is
sent and what it returns. In conversation memory:

- **The user's messages are stored as they said them.** Since 1.81.0
  `before_model` redacts copies, so history already stored is never rewritten.
  (Before, it edited the stored messages in place, and turn 1 turned into
  `[REDACTED]` in memory once turn 2 ran.)
- **The model's reply is stored redacted.** Memory records the agent's answer
  after all middleware — the same text `run` returns — with `run` and `astream`
  alike. With `astream` the caller has already received the raw deltas as they
  streamed; memory still gets the redacted reply. (Before 1.82.0, `astream`
  stored the raw text.)

To keep PII out of memory entirely, redact the input before it reaches the agent.

`patterns=` is unchanged: your regexes, applied verbatim, with no Luhn opinion and
without consulting the shared detector. Non-string content (a multimodal message's
`list[ContentPart]`) is now passed through untouched rather than raising.

## Writing Your Own Middleware

A `TokenCounter` middleware that tallies tokens across a run:

```python
from fastaiagent import Agent, AgentMiddleware, LLMClient

class TokenCounter(AgentMiddleware):
    name = "token_counter"

    async def after_model(self, ctx, response):
        used = response.usage.get("total_tokens", 0)
        ctx.scratch["tokens_total"] = ctx.scratch.get("tokens_total", 0) + used
        return response

agent = Agent(
    name="counter",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    middleware=[TokenCounter()],
)
result = await agent.arun("hello")
# Access via a second middleware, or log from after_model.
```

`response.usage` carries `prompt_tokens`, `completion_tokens` and `total_tokens`
on `arun` and `astream` alike. Before 1.91.0 a streamed turn's response had no
`total_tokens`, so this counter read `0` on every `astream` turn.

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
