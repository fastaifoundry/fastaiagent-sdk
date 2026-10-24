# Agents

An Agent is the core building block of the FastAIAgent SDK. It wraps an LLM with tools, guardrails, and memory to create an autonomous assistant that can reason, take actions, and validate its own output.

## Creating an Agent

```python
from fastaiagent import Agent, LLMClient

agent = Agent(
    name="support-bot",
    system_prompt="You are a helpful customer support agent.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

result = agent.run("How do I reset my password?")
print(result.output)
```

**Supported LLM providers:**

| Provider | Example |
|----------|---------|
| Gemini | `LLMClient(provider="gemini", model="gemini-3.8-flash")` |
| OpenAI | `LLMClient(provider="openai", model="gpt-5-mini")` |
| Anthropic | `LLMClient(provider="anthropic", model="claude-sonnet-5-5")` |
| Ollama | `LLMClient(provider="ollama", model="llama3")` |
| Azure | `LLMClient(provider="azure", model="my-gpt-5-deployment", base_url="https://myendpoint.openai.azure.com/openai/v1")` |
| AWS Bedrock | `LLMClient(provider="bedrock", model="us.anthropic.claude-haiku-4-5-20251001-v1:0", region="us-east-1")` |
| Custom | `LLMClient(provider="custom", model="my-model", base_url="https://my-api.com/v1")` |

Groq, OpenRouter, DeepSeek, Mistral and the other presets are on
[Providers](../llm/providers.md).

## Agent with Tools

Tools let your agent take actions — call APIs, search databases, run calculations.

```python
from fastaiagent import Agent, FunctionTool, LLMClient

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

def search_orders(order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: Shipped, arriving tomorrow."

agent = Agent(
    name="assistant",
    system_prompt="You help customers with weather and order questions. Use tools.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    tools=[
        FunctionTool(name="get_weather", fn=get_weather),
        FunctionTool(name="search_orders", fn=search_orders),
    ],
)

result = agent.run("What's the weather in Paris and where is order ORD-123?")
print(result.output)       # LLM's final text response
print(result.tool_calls)   # List of tool calls made
print(result.tokens_used)  # Tokens across EVERY turn of the run, not just the last
print(result.latency_ms)   # Total execution time
```

**How tool calling works:**
1. Agent sends messages + tool schemas to the LLM
2. LLM decides to call one or more tools (or respond directly)
3. SDK executes the tools and sends results back to the LLM
4. LLM generates a final response using the tool results
5. This loop repeats up to `max_iterations` times

### The @tool Decorator

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

### Tool Types

| Type | Use Case | Example |
|------|----------|---------|
| `FunctionTool` | Wrap any Python function | `FunctionTool(name="calc", fn=my_func)` |
| `RESTTool` | Call an HTTP API | `RESTTool(name="weather", url="https://api.weather.com/v1", method="GET")` |
| `MCPTool` | Use an MCP server's tools | `MCPTool.from_server("http://localhost:3000/mcp")` |
| [Provider-run](../tools/provider-tools.md) | Web search, code execution, file search run by the model's provider (1.91.0) | `web_search()`, `code_execution()`, `file_search(["vs_1"])` from `fastaiagent.tool` |

## Agent with Guardrails

Guardrails validate input/output at every step, blocking unsafe content automatically.

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.guardrail import no_pii, toxicity_check, json_valid

agent = Agent(
    name="safe-bot",
    system_prompt="You are a helpful assistant.",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6"),
    guardrails=[
        no_pii(),           # Blocks SSN, email, phone, credit cards in output
        toxicity_check(),   # Blocks toxic language
    ],
)

result = agent.run("What are the benefits of eating healthy?")
print(result.output)  # Clean output passes both guardrails
```

> **Important:** `no_pii()` is an **output guardrail** — it checks the LLM's response, not the user's input. If the LLM happens to include a real SSN, email, or phone number in its response, the guardrail blocks it and raises `GuardrailBlockedError`. Most LLMs will refuse to output real PII on their own, so this guardrail acts as a safety net for edge cases, tool results that leak PII, or less-guarded models.

To guard against **input** containing PII, set the position explicitly:

```python
from fastaiagent import Agent, LLMClient
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail import no_pii, GuardrailPosition

agent = Agent(
    name="input-safe-bot",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    guardrails=[
        no_pii(position=GuardrailPosition.input),   # Blocks PII in user input
        no_pii(position=GuardrailPosition.output),   # Blocks PII in LLM output
    ],
)

# User input containing an SSN is blocked before reaching the LLM
try:
    result = agent.run("My SSN is 123-45-6789, can you store it?")
except GuardrailBlockedError as e:
    print(f"Blocked: {e}")  # "PII detected: SSN"
```

**Built-in guardrail factories:**

All thirteen are exported from `fastaiagent.guardrail`. The default position is
in the signature — `no_pii()` is an **output** guardrail, `no_prompt_injection()`
an **input** one, `allowed_domains()` a **tool_call** one — and every factory
takes `position=` to move it.

| Factory | What it checks | Default position |
|---------|---------------|---|
| `no_pii(entities=…, backend=…)` | Email, US phone, SSN, credit cards (Luhn-validated) | `output` |
| `no_secrets()` | Leaked credentials, API keys and tokens | `output` |
| `no_prompt_injection(mode="heuristic"｜"llm")` | Prompt-injection / jailbreak attempts. `on_error="allow"` by default | `input` |
| `json_valid()` | Output is valid JSON | `output` |
| `toxicity_check(mode="keyword"｜"llm")` | Toxic language. `on_error="allow"` by default | `output` |
| `openai_moderation(model="omni-moderation-latest")` | Content flagged by the OpenAI moderation endpoint. `on_error="block"` | `output` |
| `grounded(reference, threshold=0.7)` | Whether the answer is supported by `reference`. `on_error="block"` | `output` |
| `no_hallucination(reference, threshold=0.7)` | Alias of `grounded()` — same check, the name you reach for | `output` |
| `banned_topics([...])` | Content falling under any banned topic (blocklist). `on_error="allow"` | `output` |
| `allowed_topics([...])` | Content *outside* the given topics (allowlist gate). `on_error="block"` | `output` |
| `cost_limit(max_usd=0.10)` | The run's accumulated LLM spend so far. Blocks over budget; **raises** (so `on_error` decides) when the model has no rate in the pricing table, because an unpriced run is not a free one. Until 1.67.0 this always passed. | `output` |
| `allowed_domains(["api.example.com"])` | URL domains in tool calls | `tool_call` |
| `responsible_ai(...)` | Not one guardrail — returns a **list** to spread into `guardrails=[...]`. See [Responsible AI](../guardrails/responsible-ai.md) | (several) |

See the [Guardrails reference](../guardrails/index.md) for each factory's full
signature and [`on_error`](../guardrails/concepts.md#when-the-check-itself-fails-on_error)
for what a *degraded* check costs.

**Custom guardrails:**

```python
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType

# Inline function
guardrail = Guardrail(
    name="max_length",
    position=GuardrailPosition.output,
    blocking=True,
    fn=lambda text: len(text) < 500,
)

# Regex-based
guardrail = Guardrail(
    name="no_urls",
    guardrail_type=GuardrailType.regex,
    position=GuardrailPosition.output,
    config={"pattern": r"https?://", "should_match": False},
)
```

**Guardrail positions:** `input`, `output`, `tool_call`, `tool_result`

**Blocking modes:**
- `blocking=True` — raises `GuardrailBlockedError` if validation fails
- `blocking=False` — logs the failure but continues execution

## Agent Configuration

```python
from fastaiagent import Agent, AgentConfig

agent = Agent(
    name="configured-agent",
    llm=LLMClient(provider="gemini", model="gemini-3.8-flash"),
    config=AgentConfig(
        max_iterations=5,     # Max tool-calling loop iterations (default: 10)
        tool_choice="auto",   # "auto", "required", "none"
        temperature=0.7,      # LLM temperature override
        max_tokens=1000,      # Max response tokens
    ),
)
```

Every field:

| Field | Default | What it does |
|---|---|---|
| `max_iterations` | `10` | Model calls the tool loop may make before `MaxIterationsError` |
| `tool_choice` | `"auto"` | `"required"` forces a tool on the first turn; `"none"` on every turn (below) |
| `temperature` / `max_tokens` | `None` | Sent on every model call of the run, over the `LLMClient`'s own |
| `parallel_tools` / `max_parallel_tools` | `False` / `4` | Run one turn's tool calls concurrently (sequential with middleware, a checkpointer or managed governance) |
| `output_retries` | `2` | Re-asks when the reply does not parse into `output_type` |
| `guardrail_retries` | `1` | Re-asks for an output guardrail with `action="reask"` |
| `strict_output` | `False` | OpenAI strict JSON schema for `output_type` |
| `output_mode` | `"auto"` | How `output_type` is asked for: `"native"`, `"tool"`, `"prompt"` (1.91.0, see [Structured Output](../structured-output/index.md#choosing-how-it-is-asked-for-output_mode-1910)) |
| `timeout_seconds` / `max_total_tokens` / `max_cost_usd` | `None` | Run limits (1.91.0, [below](#run-limits-and-cancel-1910)) |

`temperature` and `max_tokens` go on **every** model call of the run — each turn
of the tool loop, the structured-output re-ask and a guardrail re-ask — in place
of the `LLMClient`'s own values. A value passed to the call itself
(`agent.run(..., temperature=0.2)`) wins over both. Nothing is dropped for a
model's sampling rules: a model that only accepts the default temperature (the
OpenAI reasoning models, the newest Claude models) returns its own error, the
same as when the value is set on the `LLMClient`.

`tool_choice` is sent only when it is not `"auto"`, so an agent on the default
sends exactly the request it always did. `"required"` forces a tool call on the
**first** turn only — a model forced on every turn could never answer — and
`"none"` applies to every turn. OpenAI, Anthropic and Gemini each get their own
form of it, and Bedrock Converse takes `"required"`. Ollama has none, and Bedrock
has no `"none"` (it also needs its tool list once a history holds tool calls), so
there it logs one warning and is not sent. Any other value raises `ValueError`.

`"none"` is the provider's own switch, and a prompt that demands a tool under it
can leave the model with nothing to say: Claude then returns an empty reply and
Gemini ends with `finish_reason="unexpected_tool_call"`, so the run's `output` is
`""`. OpenAI answers in text.

!!! warning "Changed in 1.90.0"
    Before 1.90.0 all three fields were accepted and **ignored**: nothing read
    `temperature` or `max_tokens`, and `tool_choice` never reached the model.
    Agents that set them — including ones distributed from the control plane,
    whose config carries them — now send them.

### Run limits and cancel (1.91.0)

```python
agent = Agent(
    name="research",
    llm=LLMClient(provider="gemini", model="gemini-3.8-flash"),
    tools=[search],
    config=AgentConfig(
        timeout_seconds=60,      # the run's wall clock
        max_total_tokens=50_000, # checked before every model call
        max_cost_usd=0.25,       # checked before every model call
    ),
)
```

| Limit | What happens when it is reached |
|---|---|
| `timeout_seconds` | `arun` / `run` / `aresume` stop at once, inside a model or tool call, and raise `AgentTimeoutError`. `astream` stops at its next event. A synchronous tool that blocks the event loop (`time.sleep`, a blocking client) cannot be interrupted, so the stop lands when it returns. An `async` tool stops at once. |
| `max_total_tokens` | Before the next model call, the run ends normally: `status="completed"`, `stop_reason="max_total_tokens"`. No exception. |
| `max_cost_usd` | The same, with `stop_reason="max_cost_usd"`. A run whose spend cannot be priced (a model with no rate) cannot be held to it: the check raises `RunLimitError` rather than letting the run go unbudgeted. |

The caps read the run's running total, the same one `AgentResult.cost` and the
`cost_limit` guardrail read, so they never disagree. They are checked before every
model call, the structured-output and guardrail re-asks included. A re-ask the
budget no longer allows is not made: a structured run keeps its last reply (and
`stop_reason` says why), and a guardrail re-ask blocks, as when its retries run
out. `max_cost_usd` and `cost_limit` do different jobs:

- **`max_cost_usd`** is a run budget. It ends the run cleanly before the next call.
- **`cost_limit`** is a policy check at a guardrail position. It blocks with
  `GuardrailBlockedError`, and the control plane can author it.

**Cancel** a run in progress from anywhere in the process, any thread included:

```python
task = asyncio.create_task(agent.arun("long research task", execution_id="job-7"))
...
agent.cancel("job-7")      # True if the run was in progress
result = await task        # status="cancelled", stop_reason="cancelled"
```

- **A non-streamed run** stops at once and returns `status="cancelled"`, with
  `tokens_used`, `cost` and `messages` as far as it got.
- **A stream** stops at its next event and ends with
  `StreamDone(finish_reason="cancelled")`. `agent.stream()` returns
  `status="cancelled"`.
- **The checkpoint** of a cancelled or timed-out run ends `failed`, with
  `state_snapshot["run_reason"]` set to `"cancelled"` or `"timeout"`.
- **Your own task:** cancelling it (`task.cancel()`) is not `agent.cancel`. It
  cancels the run and propagates as `CancelledError`, as before.

The limits and the stop belong to the **top-level** run. An agent run inside
another one (a Supervisor's worker, an agent called from a tool, a Swarm hop)
runs under its parent's limits and stops with it. A `Supervisor` has no `cancel()`
of its own: call `cancel(<the supervisor run's execution_id>)` on any `Agent`, as
the registry of running runs is per process. A `Swarm` or `Chain` run cannot be
cancelled in 1.91.0.

The three limits are written into `Agent.to_dict()` only when set, so a
plane-dispatched or replayed agent keeps them, and a default agent's payload is
unchanged.

Limits, cancel, `result.messages` and a typed result together:
[`examples/113_run_limits_and_cancel.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/113_run_limits_and_cancel.py).

## Streaming

Stream tokens from the agent as they are generated, rather than waiting for the full response:

```python
from fastaiagent.llm.stream import TextDelta, ToolCallStart

async for event in agent.astream("What's the weather in Paris?"):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStart):
        print(f"\n[Calling {event.tool_name}...]")
```

A sync wrapper is also available:

```python
result = agent.stream("Hello")  # returns AgentResult
print(result.output)
```

Streaming runs input guardrails before streaming begins and output guardrails after streaming completes. Memory is updated at the end.

See [Streaming](../streaming/index.md) for full details, event types, and chat UI patterns.

## Sync vs Async

Every method has both sync and async versions:

```python
# Sync
result = agent.run("Hello")

# Async
result = await agent.arun("Hello")
```

Streaming also has both forms:

```python
# Async — yields events in real time
async for event in agent.astream("Hello"):
    ...

# Sync — collects into AgentResult
result = agent.stream("Hello")
```

The sync `run()` and `stream()` safely handle being called from within an async context (e.g., Jupyter notebooks, async frameworks).

## Multi-turn with `messages=`

`run`, `arun`, and `astream` accept an optional keyword-only `messages=` — a
list of prior conversation turns inserted **after** the system prompt + memory
context and **before** the current `input`. Default `None` reproduces the
single-input behavior exactly.

```python
from fastaiagent import Agent
from fastaiagent.llm.message import AssistantMessage, UserMessage

history = [
    UserMessage("My name is Alice."),
    AssistantMessage("Nice to meet you, Alice!"),
]
result = agent.run("What's my name?", messages=history)
# The model sees the prior turns and can answer "Alice".
```

`result.messages` (1.91.0) is the run's conversation in that shape — the prior
`messages=`, the input, the tool turns and the answer — so a chat continues with:

```python
first = agent.run("My name is Alice.")
second = agent.run("What's my name?", messages=first.messages)
```

It leaves out the system prompt, middleware instructions and memory context; each
run builds those again. Claude's thinking blocks stay on the assistant turns that
carry them.

This is the building block for [Agent Simulation](../simulation/index.md), which
drives multi-turn conversations against an agent and judges the transcript. For
durable, server-side conversation history across requests, see
[Memory](memory.md).

## Dynamic Instructions

System prompts can be a callable that receives the `RunContext`, enabling per-request personalization:

```python
agent = Agent(
    name="personalized",
    system_prompt=lambda ctx: (
        f"You help {ctx.state.user_name} with their {ctx.state.plan} plan."
        if ctx else
        "You are a general-purpose assistant."
    ),
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

ctx = RunContext(state=UserState(user_name="Alice", plan="pro"))
result = agent.run("Help me", context=ctx)
```

See [Dynamic Instructions](dynamic-instructions.md) for full details.

## Serialization

Agents can be serialized to JSON and restored:

```python
# Serialize
data = agent.to_dict()

# Restore
restored = Agent.from_dict(data)
```

`to_dict()` emits `{name, agent_type, system_prompt, llm_endpoint, tools, guardrails,
config}`, plus two governed fields **only when configured** (so an agent with neither is
byte-identical to earlier versions):

- `prompt_slug` — set via `Agent(prompt_slug=...)`; references a control-plane registry
  prompt. When set, `system_prompt` is emitted as `""` (the slug is the source of truth).
- `memory_enabled` — `true` when `memory=` is configured.

This is the canonical payload for pushing an agent to a connected control plane — `POST` it
to `/public/v1/sdk/agents`. See [Pushing agent definitions](../platform/index.md#registering-agent-definitions).

## AgentResult

Every agent execution returns an `AgentResult`. It has fifteen fields, and the
table below is all of them:

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | The agent's final text response |
| `parsed` | `T \| None` | The response parsed into your `output_type`, when the agent declares one. `None` otherwise — read this, not `output`, for a structured run. Typed since 1.91.0: see below. |
| `tool_calls` | `list[dict]` | All tool calls made during execution |
| `messages` | `list[Message]` | The run's conversation (1.91.0): prior `messages=`, the input, tool turns and the answer, without the system prompt, instructions or memory context. Pass it back as `messages=` to continue. |
| `tokens_used` | `int` | Tokens across **every** LLM call the run made — each turn of the tool loop, a structured re-ask, a guardrail re-ask. Same run-scoped accumulator `cost` is built from, so the two always agree. |
| `cost` | `float` | Estimated USD spend, summed over **every** LLM call the run made — the tool loop's turns, a structured re-ask, a guardrail re-ask. Priced from the model id and the provider's token counts against the built-in list-price table (override it with `set_rate_overrides()`). `0.0` for a model with no rate — see `cost_known`. |
| `cost_known` | `bool` | Whether every call in the run could be priced. **`cost == 0.0` does not always mean the run was free**: a private fine-tune and a bedrock/azure deployment id have no rate, and reporting `$0.00` for them would be a guess, so this reads `False`. A self-hosted provider (`ollama`, `lmstudio`, `vllm`) is the exception — it runs on hardware you already pay for, so it is a known zero and this reads `True`. Check this flag before trusting the number. |
| `latency_ms` | `int` | Total execution time in milliseconds |
| `trace_id` | `str \| None` | Trace ID for debugging. Populated on every path since 1.67.0 — `run`, `arun`, `stream`, and the same fields on `Swarm`, `Supervisor` and `Chain`. |
| `execution_id` | `str` | The durable execution's id. Always populated when a `checkpointer=` is configured; `""` otherwise. It is what you pass to `agent.aresume(...)`. |
| `status` | `str` | `"completed"` for a normal run, `"paused"` when a tool called [`interrupt()`](../durability/index.md), `"cancelled"` after `agent.cancel()` (1.91.0). A paused or cancelled run **returns** — it does not raise — so branch on this. |
| `pending_interrupt` | `dict \| None` | Set when `status == "paused"`: `{reason, context, node_id, agent_path}` — the same payload the `/approvals` UI reads from the `pending_interrupts` table. |
| `stop_reason` | `str \| None` | Why the run ended early: middleware's `StopAgent` reason (1.90.0), or `"stopped"` when it gave none; `"max_total_tokens"` / `"max_cost_usd"` for a run limit, `"cancelled"` for a cancel (1.91.0). `output` is the stop message. `None` for a run that was not stopped. |
| `route` | `SupervisorRoute \| None` | How a `Supervisor(routing="decisions")` routed the request: the worker, the choice's confidence, and any fallback. `None` otherwise. |
| `guardrails` | `list[GuardrailFiring]` | Every guardrail that executed, in order, across all four positions. The **only** way to observe a non-halting outcome: `warn`, `mask` and `override` all let the run finish, so before 1.64.0 a run with the Local UI off and no plane attached reported a clean string whether or not a rule had fired. A blocking failure still raises `GuardrailBlockedError` rather than returning. |

**Typed `parsed` (1.91.0).** `AgentResult` is generic in `parsed`. Pass
`output_type=` to a single run, or declare it on the agent, and your type checker
knows what `parsed` holds:

```python
result = agent.run("Extract the invoice", output_type=Invoice)  # AgentResult[Invoice]
if result.parsed is not None:                                  # Invoice | None
    print(result.parsed.total)                                 # typed as float
```

Plain `AgentResult` is `AgentResult[Any]`, so annotations written before 1.91.0
keep working.

```python
result = agent.run("my ssn is 123-45-6789")

if result.status == "paused":
    print("waiting on", result.pending_interrupt["reason"], result.execution_id)

for g in result.guardrails:
    if g.fired():
        print(g.name, g.position, g.action_taken)
```

!!! warning "`tokens_used` counts the whole run as of 1.68.0 — the number goes up"
    It used to be read off the **last** LLM response the tool loop returned, so a
    three-call run reported one call's tokens and a re-ask reported none of the
    turns that preceded it. It now comes from the run-scoped accumulator
    `LLMClient` writes to at the single point every completion passes through —
    the same hook `cost` uses, which is why the two now agree.

    This is a behaviour change with no deprecation window: for any multi-turn run
    the reported number **increases**, and a budget, alert or assertion calibrated
    against the old value will see a step change at 1.68.0. Single-call runs are
    unaffected. A custom client that implements `acomplete` without extending
    `LLMClient` never reaches the accumulator and keeps the legacy last-response
    sum, because that is a better answer for it than zero.

!!! note "Where cost comes from, and where it lands"
    It is computed at the provider call inside `LLMClient` and also written to
    the `llm.*` span as `fastaiagent.cost.total_usd` — the same attribute the
    LangChain, CrewAI and Pydantic-AI integrations have always emitted, so the
    control plane, the Local UI and trace export read one key for every
    framework. Before 1.67.0 nothing populated `AgentResult.cost` and the SDK's
    own runs were the only framework the plane received no cost for; the UI hid
    it behind a read-time estimate from token counts.

## Error Handling

```python
from fastaiagent._internal.errors import (
    AgentError,              # Base agent error
    AgentTimeoutError,       # The run passed AgentConfig.timeout_seconds (1.91.0)
    RunLimitError,           # max_cost_usd set, but the run cannot be priced (1.91.0)
    MaxIterationsError,      # Tool loop exceeded max_iterations
    GuardrailBlockedError,   # Guardrail rejected input/output
    LLMProviderError,        # LLM API error
    LLMRateLimitError,       # a 429, with retry_after in seconds (1.91.0)
)

try:
    result = agent.run("Do something complex")
except MaxIterationsError:
    print("Agent couldn't complete in time")
except GuardrailBlockedError as e:
    print(f"Blocked by {e.guardrail_name}: {e}")
except LLMRateLimitError as e:   # before LLMProviderError: it is a subclass
    print(f"Rate limited; the provider asked to wait {e.retry_after}s")
except LLMProviderError as e:
    print(f"LLM error: {e}")
```

`LLMProviderError` carries `status_code`, plus, since 1.89.0, `provider`,
`provider_message` (the provider's own explanation), `provider_code` (for
example `invalid_api_key`) and `request_id`. Each is `None` when the provider
didn't send it. Its message is scrubbed of credentials, because a provider can
echo a rejected key back (see
[Security posture](../security.md#credentials-in-traces-and-pushed-definitions)).

---

## Middleware

Middleware intercepts and transforms messages, responses, and tool calls without subclassing `Agent`. Use it for message trimming, PII redaction, tool-call budgets, caching, and other cross-cutting concerns. Since 1.91.0 it can also wrap the whole run (`before_run`, `after_run`, `on_error`) and each model call (`wrap_model`), bring its own tools and instructions, and be set once on a `Chain`, `Swarm` or `Supervisor`.

```python
from fastaiagent import Agent, TrimLongMessages, ToolBudget, RedactPII

agent = Agent(
    name="controlled",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    middleware=[
        TrimLongMessages(keep_last=20),
        RedactPII(),
        ToolBudget(max_calls=10),
    ],
)
```

See [Middleware](middleware.md) for the full reference, ordering semantics, and how to write your own.

## Next Steps

- [Middleware](middleware.md) — Run, model-call and tool hooks; policies as one bundle; team-wide middleware
- [Dynamic Instructions](dynamic-instructions.md) — Personalize system prompts per-request
- [Agent Memory](memory.md) — Give agents conversation memory across turns
- [Multi-Agent Teams](teams.md) — Supervisor / Worker (centralized delegation)
- [Swarm](swarm.md) — Peer-to-peer handoff topology (no coordinator)
- [Tools](tools.md) — Deep dive into using tools with agents
- [Guardrails](../guardrails/index.md) — Full guardrail reference
- [Chains](../chains/index.md) — Compose agents into multi-step workflows
