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

The `response_format` structure matches the FastAIAgent Platform's `ResponseFormat` schema, ensuring compatibility between SDK and platform API invocations.

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

## Output Type (Pydantic Models)

Instead of manually constructing `response_format` dicts and parsing JSON, use `output_type` on Agent to get automatic Pydantic model parsing:

```python
from pydantic import BaseModel
from fastaiagent import Agent, LLMClient

class Person(BaseModel):
    name: str
    age: int
    city: str

agent = Agent(
    name="extractor",
    system_prompt="Extract person info from the message.",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
    output_type=Person,
)

result = agent.run("Alice is 30 and lives in Tokyo.")
print(result.parsed.name)   # "Alice"
print(result.parsed.age)    # 30
print(result.parsed.city)   # "Tokyo"
print(result.output)         # Raw JSON string
```

### How it works

1. The SDK generates a `response_format` from `output_type.model_json_schema()`
2. The format is passed to the LLM as a kwarg (works with all providers)
3. The JSON response is automatically parsed into a Pydantic model on `result.parsed`
4. If parsing fails, `result.parsed` is `None` and `result.output` contains the raw text

### Nested models

```python
class Address(BaseModel):
    street: str
    city: str

class Customer(BaseModel):
    name: str
    address: Address

agent = Agent(name="extractor", output_type=Customer, ...)
result = agent.run("John at 123 Main St, SF")
print(result.parsed.address.city)  # "SF"
```

### Streaming

`stream()` collects all tokens and parses at the end:

```python
result = agent.stream("Alice is 30 from Tokyo.")
print(result.parsed)  # Person(name='Alice', age=30, city='Tokyo')
```

### Serialization

`to_dict()` includes the JSON schema in `config.response_format`. The `output_type` Python class cannot be restored from `from_dict()` — the schema is informational.

## LLM Parameters

`LLMClient` supports additional sampling parameters with automatic per-provider mapping:

```python
llm = LLMClient(
    provider="openai",
    model="gpt-4.1",
    temperature=0.7,
    top_p=0.9,
    seed=42,
    stop=["END", "\n\n"],
    frequency_penalty=0.5,
    presence_penalty=0.3,
    parallel_tool_calls=False,
)
```

**Per-call override:**

```python
response = llm.complete(messages, top_p=0.5)  # overrides 0.9 for this call
```

**Provider compatibility:**

| Parameter | OpenAI | Anthropic | Ollama | Bedrock |
|-----------|:------:|:---------:|:------:|:-------:|
| `top_p` | Yes | Yes | Yes | Yes |
| `stop` | Yes | Yes (as `stop_sequences`) | Yes | Yes (as `stopSequences`) |
| `seed` | Yes | -- | Yes | -- |
| `frequency_penalty` | Yes | -- | Yes | -- |
| `presence_penalty` | Yes | -- | Yes | -- |
| `parallel_tool_calls` | Yes | -- | -- | -- |

Unsupported parameters are silently skipped for each provider.

## Retry with Backoff

`LLMClient` supports automatic retries on transient errors (HTTP 429 rate limits and 5xx server errors):

```python
llm = LLMClient(
    provider="openai",
    model="gpt-4.1",
    max_retries=3,  # Retry up to 3 times
)
```

**Behavior:**
- Retries on: 429 (rate limit), 500+ (server errors)
- No retry on: 400, 401, 403, 404 (client errors)
- Backoff: exponential (1s, 2s, 4s, 8s, ... capped at 30s)
- `LLMProviderError.status_code` gives the HTTP status code

```python
from fastaiagent._internal.errors import LLMProviderError

try:
    response = llm.complete(messages)
except LLMProviderError as e:
    print(f"Status: {e.status_code}")  # e.g., 429
```

---

## Next Steps

- [Streaming](../streaming/index.md) — Stream structured output tokens in real time
- [Agents](../agents/index.md) — Build agents that return structured data
- [Guardrails](../guardrails/index.md) — Validate structured output with `json_valid()` guardrail
- [Evaluation](../evaluation/index.md) — Score structured output with `JSONValid` scorer
