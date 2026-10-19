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
| **Gemini** | Yes, without tools | `responseMimeType` + `responseSchema` on a call with no tools. While function tools are in the request, the schema is asked for in the prompt and the fenced reply is parsed: Gemini refuses JSON mode with function calling (2.5: HTTP 400) or calls tools forever (3.5, 3.8). Fixed in 1.91.0 (X20); before, an `output_type` agent with tools never worked on Gemini |
| **Ollama** | Partial | `json_object` → `format: "json"`, `json_schema` → `format: {schema}` |
| **Bedrock** | Opt-in | Schema in the system prompt; `outputConfig.textFormat` with `output_mode="native"`. Before 1.91.0 Bedrock sent neither, so a structured agent got no schema (X21) |

> **Note:** For Anthropic, the SDK automatically augments the system prompt with JSON instructions and strips any markdown code fences from the response. This means your code works identically across providers — no per-provider handling needed.

## Choosing how it is asked for: `output_mode` (1.91.0)

An agent with `output_type` asks the model for that shape. `AgentConfig(output_mode=...)`
picks how:

| Mode | What is sent | When to use it |
|---|---|---|
| `"auto"` (default) | Each provider's usual way: OpenAI/Azure native `response_format`, Gemini's native schema (prompted on tool turns), Ollama `format`, the schema in the prompt for Anthropic and Bedrock | Almost always |
| `"native"` | The provider's own schema-constrained output, including **Anthropic `output_config.format`** and **Bedrock `outputConfig.textFormat`** | You want the provider to guarantee the shape. On Anthropic the schema is adapted to its rules: objects get `additionalProperties: false`, and `minimum` / `maxLength` and the like move into the field description. The reply is still validated against your full type. |
| `"tool"` | A `final_result` tool whose arguments are your schema. Calling it ends the run, and its arguments are `parsed` | Agents that use tools and should keep using them until they answer. The tool is offered, not forced, so it works on models that refuse forced tool use (Opus 5.5, Sonnet 5.5). A text answer that parses is accepted too. |
| `"prompt"` | The schema in the system prompt only, no provider parameter | A gateway or model that rejects the native fields |

```python
from fastaiagent import Agent, AgentConfig, LLMClient

agent = Agent(
    name="forecaster",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-5-5"),
    tools=[get_weather],
    output_type=Forecast,
    config=AgentConfig(output_mode="tool"),
)
result = agent.run("What's the weather in Rome?")
result.parsed  # Forecast(city='Rome', celsius=24.0, ...)
```

A tool named `final_result` on the agent is refused in `"tool"` mode. `output_mode`
is part of `Agent.to_dict()` when it is not `"auto"`.

### Streaming the answer as it forms: `PartialOutput`

While a structured run streams, `agent.astream()` also yields `PartialOutput(data=...)`:
the reply so far, parsed as incomplete JSON. Keys and values appear as they arrive,
and a string can be partway through. `data` changes with each event and is not
validated; the finished reply is, on the result.

```python
from fastaiagent import PartialOutput

async for event in agent.astream("What's the weather in Rome?"):
    if isinstance(event, PartialOutput):
        render(event.data)   # {"city": "Ro"} … {"city": "Rome", "celsius": 24, …}
```

Text the model says before a tool call ("Let me check.") is not JSON, so it yields
none. In `"tool"` mode the answer arrives as a tool call, which providers stream
whole, so it comes as one `PartialOutput`.

Every mode, and `PartialOutput` while the answer streams, on Gemini 3.8 Flash,
Claude Sonnet 5.5 and gpt-5-mini:
[`examples/114_output_modes_and_partial_output.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/114_output_modes_and_partial_output.py).

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
3. The JSON response is automatically parsed into the output type on `result.parsed`
4. If parsing fails, the agent re-asks the model with the error (up to `output_retries`, default 2); if it still fails, `result.parsed` is `None` and `result.output` holds the raw text

### Nested models

```python
class Address(BaseModel):
    street: str
    city: str

class Customer(BaseModel):
    name: str
    address: Address

agent = Agent(name="extractor", output_type=Customer, llm=llm)
result = agent.run("John at 123 Main St, SF")
print(result.parsed.address.city)  # "SF"
```

### Lists, primitives, and other types

As of v1.42.0, `output_type` accepts any Pydantic-compatible type — not just a
`BaseModel`. Lists, primitives, enums, and unions all work; the SDK wraps
non-object types internally and unwraps them on parse, so you just read
`result.parsed`:

```python
agent = Agent(name="geo", output_type=list[Country], llm=llm)
result = agent.run("List France, Japan, and Egypt with their capitals.")
result.parsed          # [Country(name='France', capital='Paris'), ...]

agent = Agent(name="counter", output_type=int, llm=llm)
agent.run("How many sides does a hexagon have?").parsed   # 6
```

### Retry on validation failure

If the model returns malformed JSON or output that doesn't match the schema, the
agent re-asks it with the validation error and tries again — up to
`AgentConfig.output_retries` times (**default 2**). Retries only fire on failure,
so they add no cost on the happy path. Set `output_retries=0` to restore the old
behavior (a failure yields `parsed=None`):

```python
agent = Agent(
    name="extractor",
    output_type=Invoice,
    config=AgentConfig(output_retries=3),   # 0 disables
    llm=llm,
)
```

### Strict Structured Outputs (OpenAI/Azure)

For a hard schema guarantee on OpenAI/Azure, enable native strict Structured
Outputs. The SDK adapts the schema (all-required + `additionalProperties:false`)
automatically. It's opt-in and ignored by non-OpenAI providers:

```python
agent = Agent(
    name="extractor",
    output_type=Company,
    config=AgentConfig(strict_output=True),
    llm=LLMClient(provider="openai", model="gpt-4o"),
)
```

### Streaming

`stream()` collects all tokens and parses at the end (retry applies to
`run()`/`arun()`, not streaming):

```python
result = agent.stream("Alice is 30 from Tokyo.")
print(result.parsed)  # Person(name='Alice', age=30, city='Tokyo')
```

### Serialization

`to_dict()` includes the JSON schema in `config.response_format`. The `output_type` Python class cannot be restored from `from_dict()` — the schema is informational.

### At the `LLMClient` level (no Agent)

`output_type` also works directly on `LLMClient.complete()` / `acomplete()`, which
populate `LLMResponse.parsed`:

```python
from fastaiagent import LLMClient
from fastaiagent.llm.message import UserMessage

client = LLMClient(provider="openai", model="gpt-4o")
resp = client.complete([UserMessage("France and its capital?")], output_type=Country)
resp.parsed   # Country(name='France', capital='Paris')
```

The same flexible types apply. The retry and strict-mode features are Agent-level;
at the client level you get a single structured call (build `response_format`
yourself if you need strict mode).

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

`LLMClient` supports automatic retries on transient errors:

```python
llm = LLMClient(
    provider="openai",
    model="gpt-5-mini",
    max_retries=3,  # Retry up to 3 times
)
```

**Behavior:**
- Retries on: 429 (rate limit), 500+ (server errors), and (*since 1.91.0*) a
  refused or reset connection and a timeout
- No retry on: 400, 401, 403, 404 (client errors)
- Backoff: what the provider asked for in `Retry-After` (capped at 60s), else
  exponential (1s, 2s, 4s, 8s, ... capped at 30s)
- `LLMProviderError.status_code` gives the HTTP status code; a 429 is
  `LLMRateLimitError`, a subclass, with `retry_after` in seconds (1.91.0)
- When the retries are spent, `fallbacks=[LLMClient(...)]` tries another model
  (1.91.0) — see [When a model fails](../llm/providers.md#when-a-model-fails-retries-rate-limits-fallbacks)

```python
from fastaiagent._internal.errors import LLMProviderError, LLMRateLimitError

try:
    response = llm.complete(messages)
except LLMRateLimitError as e:
    print(f"Rate limited; retry after {e.retry_after}s")
except LLMProviderError as e:
    print(f"Status: {e.status_code}")
```

---

## Next Steps

- [Streaming](../streaming/index.md) — Stream structured output tokens in real time
- [Agents](../agents/index.md) — Build agents that return structured data
- [Guardrails](../guardrails/index.md) — Validate structured output with `json_valid()` guardrail
- [Evaluation](../evaluation/index.md) — Score structured output with `JSONValid` scorer
