# Structured Output

Structured output forces the LLM to respond with valid JSON matching a specific format. This eliminates manual parsing and validation, giving you reliable typed data from any provider.

## Quick Start

```python
from fastaiagent import LLMClient
from fastaiagent.llm import UserMessage

llm = LLMClient(provider="openai", model="gpt-4.1")

response = llm.complete(
    [UserMessage("What is the capital of France?")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "country_info",
            "schema": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "capital": {"type": "string"},
                    "population": {"type": "integer"},
                },
                "required": ["country", "capital"],
            },
        },
    },
)

import json
data = json.loads(response.content)
print(data["capital"])  # "Paris"
```

## Response Format Types

The `response_format` parameter accepts a dict with a `type` field:

| Type | Description | Use Case |
|------|------------|----------|
| `"text"` | Plain text (default) | Normal conversations |
| `"json_object"` | Any valid JSON | When you need JSON but the schema is flexible |
| `"json_schema"` | JSON conforming to a specific schema | When you need structured, typed data |

### JSON Object Mode

Forces the LLM to respond with valid JSON, without specifying a schema:

```python
response = llm.complete(
    [UserMessage("List 3 colors as JSON")],
    response_format={"type": "json_object"},
)
# '{"colors": ["red", "blue", "green"]}'
```

### JSON Schema Mode

Forces the LLM to respond with JSON matching an explicit schema:

```python
response = llm.complete(
    [UserMessage("Describe a person named Alice who is 30")],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "hobbies": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "age"],
            },
            "strict": True,  # OpenAI only — enforces strict schema adherence
        },
    },
)
```

**Schema fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Identifier for the schema (e.g. `"person"`) |
| `schema` | `dict` | The JSON Schema object |
| `strict` | `bool \| None` | Enable strict schema adherence (OpenAI only) |

## Provider Behavior

Structured output works across all providers, with automatic adaptation:

| Provider | Native Support | How It Works |
|----------|:-:|-------------|
| **OpenAI** | Yes | `response_format` passed directly to API |
| **Azure** | Yes | Same as OpenAI (OpenAI-compatible) |
| **Custom** | Yes | Same as OpenAI (OpenAI-compatible) |
| **Anthropic** | No | Schema injected into system prompt; code fences stripped from response |
| **Ollama** | Partial | `json_object` → `format: "json"`, `json_schema` → `format: {schema}` |
| **Bedrock** | No | Not supported |

> **Note:** For Anthropic, the SDK automatically augments the system prompt with JSON instructions and strips any markdown code fences from the response. This means your code works identically across providers — no per-provider handling needed.

## Streaming with Structured Output

Structured output works with streaming. The tokens arrive as normal `TextDelta` events; the final concatenated text is valid JSON:

```python
from fastaiagent.llm.stream import TextDelta

content = ""
async for event in llm.astream(
    [UserMessage("Describe Paris")],
    response_format={"type": "json_object"},
):
    if isinstance(event, TextDelta):
        content += event.text
        print(event.text, end="", flush=True)

import json
data = json.loads(content)
```

## Using with Agents

Pass `response_format` through the agent's `run()` or `arun()` kwargs. The format is forwarded to the underlying LLM:

```python
from fastaiagent import Agent, LLMClient

agent = Agent(
    name="data-extractor",
    system_prompt="Extract structured data from user messages.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

result = agent.run(
    "Alice is 30 years old and lives in Paris",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "city": {"type": "string"},
                },
                "required": ["name", "age", "city"],
            },
        },
    },
)

import json
person = json.loads(result.output)
print(person)  # {"name": "Alice", "age": 30, "city": "Paris"}
```

## Platform Compatibility

The `response_format` structure matches the FastAIAgent Platform's `ResponseFormat` schema. When pushing agents to the platform, the same format works in both SDK and platform API invocations.

**Platform request format:**
```json
{
  "message": "Describe Paris",
  "options": {
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "city_info",
        "schema": { ... },
        "strict": true
      }
    }
  }
}
```

## Error Handling

```python
from fastaiagent._internal.errors import LLMProviderError

try:
    response = llm.complete(
        [UserMessage("Give me JSON")],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.content)
except LLMProviderError as e:
    print(f"LLM error: {e}")
except json.JSONDecodeError:
    print("LLM returned invalid JSON despite response_format")
```

> **Note:** With OpenAI's `strict: true` mode, invalid JSON should not occur. With Anthropic or Ollama, the LLM may occasionally return imperfect JSON. Always include a `json.JSONDecodeError` handler as a safety net.

---

## Next Steps

- [Streaming](../streaming/index.md) — Stream structured output tokens in real time
- [Agents](../agents/index.md) — Build agents that return structured data
- [Guardrails](../guardrails/index.md) — Validate structured output with `json_valid()` guardrail
- [Evaluation](../evaluation/index.md) — Score structured output with `JSONValid` scorer
