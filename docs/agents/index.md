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
| OpenAI | `LLMClient(provider="openai", model="gpt-4.1")` |
| Anthropic | `LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")` |
| Ollama | `LLMClient(provider="ollama", model="llama3")` |
| Azure | `LLMClient(provider="azure", model="gpt-4", base_url="https://myendpoint.openai.azure.com/openai/deployments/gpt-4/")` |
| AWS Bedrock | `LLMClient(provider="bedrock", model="anthropic.claude-3-sonnet-20240229-v1:0")` |
| Custom | `LLMClient(provider="custom", model="my-model", base_url="https://my-api.com/v1")` |

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
print(result.tokens_used)  # Total tokens consumed
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
| `MCPTool` | Connect to MCP server | `MCPTool(name="search", server_url="http://localhost:3000")` |

## Agent with Guardrails

Guardrails validate input/output at every step, blocking unsafe content automatically.

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.guardrail import no_pii, toxicity_check, json_valid

agent = Agent(
    name="safe-bot",
    system_prompt="You are a helpful assistant.",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"),
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

| Factory | What it checks |
|---------|---------------|
| `no_pii()` | SSN, email, phone numbers, credit card numbers |
| `json_valid()` | Output is valid JSON |
| `toxicity_check()` | Toxic keywords |
| `cost_limit(max_usd=0.10)` | Accumulated cost |
| `allowed_domains(["api.example.com"])` | URL domains in tool calls |

**Custom guardrails:**

```python
from fastaiagent.guardrail import Guardrail, GuardrailPosition

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
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    config=AgentConfig(
        max_iterations=5,     # Max tool-calling loop iterations (default: 10)
        tool_choice="auto",   # "auto", "required", "none"
        temperature=0.7,      # LLM temperature override
        max_tokens=1000,      # Max response tokens
    ),
)
```

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

## Serialization

Agents can be serialized to JSON and restored:

```python
# Serialize
data = agent.to_dict()

# Restore
restored = Agent.from_dict(data)
```

This is the format used when pushing to the platform with `fa.push(agent)`.

## AgentResult

Every agent execution returns an `AgentResult`:

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | The agent's final text response |
| `tool_calls` | `list[dict]` | All tool calls made during execution |
| `tokens_used` | `int` | Total tokens consumed |
| `cost` | `float` | Estimated cost in USD |
| `latency_ms` | `int` | Total execution time in milliseconds |
| `trace_id` | `str \| None` | Trace ID for debugging |

## Error Handling

```python
from fastaiagent._internal.errors import (
    AgentError,              # Base agent error
    AgentTimeoutError,       # Execution timeout
    MaxIterationsError,      # Tool loop exceeded max_iterations
    GuardrailBlockedError,   # Guardrail rejected input/output
    LLMProviderError,        # LLM API error
)

try:
    result = agent.run("Do something complex")
except MaxIterationsError:
    print("Agent couldn't complete in time")
except GuardrailBlockedError as e:
    print(f"Blocked by {e.guardrail_name}: {e}")
except LLMProviderError as e:
    print(f"LLM error: {e}")
```

---

## Next Steps

- [Agent Memory](memory.md) — Give agents conversation memory across turns
- [Multi-Agent Teams](teams.md) — Build supervisor/worker agent teams
- [Tools](tools.md) — Deep dive into using tools with agents
- [Guardrails](../guardrails/index.md) — Full guardrail reference
- [Chains](../chains/index.md) — Compose agents into multi-step workflows
