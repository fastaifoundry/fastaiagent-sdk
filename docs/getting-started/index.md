# Getting Started

Welcome to the FastAIAgent SDK. This guide walks you through the foundational component of the SDK -- the `LLMClient` -- which powers every agent, chain, guardrail judge, and eval scorer. Once you understand the LLM client, you can build on top of it with agents, tools, chains, and more.

---

# LLM Client

The `LLMClient` is the most fundamental component of the SDK. Every agent, chain, guardrail judge, and eval scorer depends on it. It provides a unified interface to call any LLM provider — OpenAI, Anthropic, Ollama, Azure, Bedrock, or any OpenAI-compatible endpoint — through a single API.

## Quick Start

```python
from fastaiagent import LLMClient
from fastaiagent.llm import SystemMessage, UserMessage

llm = LLMClient(provider="openai", model="gpt-4.1")
response = llm.complete([UserMessage("What is 2+2?")])

print(response.content)       # "4"
print(response.usage)          # {"prompt_tokens": 12, "completion_tokens": 3, ...}
print(response.finish_reason)  # "stop"
print(response.latency_ms)     # 450
```

## Supported Providers

| Provider | Setup | API Key Env Var |
|----------|-------|-----------------|
| **OpenAI** | `LLMClient(provider="openai", model="gpt-4.1")` | `OPENAI_API_KEY` |
| **Anthropic** | `LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")` | `ANTHROPIC_API_KEY` |
| **Ollama** | `LLMClient(provider="ollama", model="llama3")` | None (local) |
| **Azure OpenAI** | `LLMClient(provider="azure", model="gpt-4", base_url="https://<endpoint>.openai.azure.com/openai/deployments/gpt-4/")` | `OPENAI_API_KEY` |
| **AWS Bedrock** | `LLMClient(provider="bedrock", model="anthropic.claude-3-sonnet-20240229-v1:0", region="us-east-1")` | AWS credentials |
| **Custom** | `LLMClient(provider="custom", model="my-model", base_url="https://my-api.com/v1")` | `OPENAI_API_KEY` |

## API Key Handling

Keys are resolved in this order:
1. `api_key` parameter passed to constructor
2. Environment variable (`OPENAI_API_KEY` for OpenAI, `ANTHROPIC_API_KEY` for Anthropic)

```python
# Explicit key
llm = LLMClient(provider="openai", model="gpt-4.1", api_key="sk-...")

# From environment (recommended)
# export OPENAI_API_KEY=sk-...
llm = LLMClient(provider="openai", model="gpt-4.1")
```

If no key is found, a clear error is raised:
```
LLMProviderError: No API key provided. Set api_key parameter or OPENAI_API_KEY env var.
```

## Messages

The SDK uses typed message objects that are automatically converted to each provider's format.

```python
from fastaiagent.llm import (
    SystemMessage,      # Sets agent behavior/persona
    UserMessage,        # User's input
    AssistantMessage,   # LLM's response (used in multi-turn)
    ToolMessage,        # Tool execution result
)

messages = [
    SystemMessage("You are a helpful assistant."),
    UserMessage("What is Python?"),
]
response = llm.complete(messages)
```

### Multi-Turn Conversations

```python
messages = [
    SystemMessage("You are a math tutor. Be concise."),
    UserMessage("What is 2+2?"),
]
r1 = llm.complete(messages)
print(r1.content)  # "4"

# Continue the conversation
messages.append(AssistantMessage(r1.content))
messages.append(UserMessage("And 3+3?"))
r2 = llm.complete(messages)
print(r2.content)  # "6"
```

### Message Format Conversion

The SDK automatically handles provider-specific message formats:

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System message | Inline `{"role": "system"}` | Separate `system` field |
| Tool calls | `tool_calls` in assistant msg | `tool_use` content blocks |
| Tool results | `{"role": "tool"}` | `{"role": "user", "type": "tool_result"}` |

You don't need to worry about these differences — write messages once, they work everywhere.

## Tool Calling

When tools are provided, the LLM can request tool executions:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = llm.complete([UserMessage("Weather in Paris?")], tools=tools)

if response.tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}, Args: {tc.arguments}")
        # tc.id, tc.name, tc.arguments
```

After executing tools, send results back:

```python
from fastaiagent.llm import AssistantMessage, ToolMessage

messages.append(AssistantMessage(content=None, tool_calls=response.tool_calls))
messages.append(ToolMessage(content="Sunny, 22°C", tool_call_id=response.tool_calls[0].id))

final = llm.complete(messages, tools=tools)
print(final.content)  # "The weather in Paris is sunny at 22°C."
```

> **Note:** You rarely need to manage this loop manually. The `Agent` class handles the entire tool-calling loop automatically. Use `LLMClient` directly only when you need low-level control.

## Configuration

```python
llm = LLMClient(
    provider="openai",
    model="gpt-4.1",
    api_key="sk-...",            # Optional if env var is set
    base_url="https://...",      # Override default endpoint
    temperature=0.7,             # Sampling temperature
    max_tokens=1000,             # Max response tokens
)
```

### Per-Request Overrides

```python
response = llm.complete(messages, max_tokens=500)
```

## LLMResponse

Every completion returns an `LLMResponse`:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str \| None` | Text response (None if only tool calls) |
| `tool_calls` | `list[ToolCall]` | Tool calls requested by the LLM |
| `usage` | `dict` | Token counts: `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `model` | `str` | Model used |
| `finish_reason` | `str` | `"stop"` (complete), `"tool_calls"` (needs tool execution) |
| `latency_ms` | `int` | Request latency in milliseconds |

## Sync vs Async

```python
# Sync (works everywhere, including Jupyter)
response = llm.complete(messages)

# Async (for async frameworks, better performance)
response = await llm.acomplete(messages)
```

### Streaming

Stream tokens as they are generated, rather than waiting for the full response:

```python
from fastaiagent.llm.stream import TextDelta, Usage

# Async streaming — yields StreamEvent objects
async for event in llm.astream([UserMessage("Hello!")]):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, Usage):
        print(f"\nTokens: {event.prompt_tokens} in, {event.completion_tokens} out")

# Sync streaming — collects into LLMResponse
response = llm.stream([UserMessage("Hello!")])
print(response.content)
```

Streaming is supported for OpenAI, Anthropic, Ollama, Azure, and Custom providers. See [Streaming](../streaming/index.md) for full details.

### Structured Output

Force the LLM to respond with valid JSON matching a specific schema:

```python
response = llm.complete(
    [UserMessage("Describe Paris")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "city_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "country": {"type": "string"},
                },
                "required": ["name", "country"],
            },
        },
    },
)

import json
data = json.loads(response.content)
print(data["name"])  # "Paris"
```

Three modes: `"text"` (default), `"json_object"` (any JSON), `"json_schema"` (schema-validated JSON). Works across OpenAI, Anthropic, and Ollama with automatic provider adaptation. See [Structured Output](../structured-output/index.md) for full details.

## Serialization

```python
# Serialize (for platform push, config storage)
config = llm.to_dict()
# {"provider": "openai", "model": "gpt-4.1", "temperature": 0.7}

# Restore
llm = LLMClient.from_dict(config)
```

Note: `api_key` is NOT included in serialization for security.

## Provider-Specific Details

### OpenAI

- Uses `max_completion_tokens` for newer models (gpt-4.1, gpt-4o)
- Usage includes `prompt_tokens_details` and `completion_tokens_details`
- All OpenAI models supported including o-series reasoning models

### Anthropic

- System messages extracted to separate `system` field automatically
- Tool schemas converted from OpenAI format to Anthropic `input_schema` format
- Tool results converted to `tool_result` content blocks automatically
- `finish_reason` normalized: `end_turn` → `stop`, `tool_use` → `tool_calls`
- Usage fields normalized: `input_tokens` → `prompt_tokens`, `output_tokens` → `completion_tokens`

### Ollama

- Runs locally at `http://localhost:11434` by default
- `max_tokens` mapped to `num_predict`
- Tool call IDs are synthetic (`call_0`, `call_1`, ...)
- No API key required

### Custom Endpoints

Any OpenAI-compatible API (vLLM, LiteLLM, etc.):

```python
llm = LLMClient(
    provider="custom",
    model="my-model",
    base_url="https://my-api.com/v1",
    api_key="my-key",
)
```

## Error Handling

```python
from fastaiagent._internal.errors import LLMError, LLMProviderError

try:
    response = llm.complete(messages)
except LLMProviderError as e:
    print(f"API error: {e}")  # 401, 429, 500, etc.
except LLMError as e:
    print(f"LLM error: {e}")  # Unsupported provider, missing key, etc.
```

---

## Next Steps

- [Agents](../agents/index.md) — Build autonomous agents on top of the LLM client
- [Streaming](../streaming/index.md) — Real-time token delivery from LLM to your app
- [Tools](../tools/index.md) — Give agents the ability to take actions
- [Chains](../chains/index.md) — Compose agents into multi-step workflows
