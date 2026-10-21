# OpenAI Responses API

*New in 1.91.0.*

`LLMClient(provider="openai", api="responses")` sends calls to OpenAI's Responses
API (`POST /v1/responses`) instead of Chat Completions. Nothing else changes: the
agent loop, tools, guardrails, middleware, streaming, checkpoints, retries,
fallbacks, tracing and pricing work the same. Chat Completions stays the default.

```python
from fastaiagent import Agent, FunctionTool, LLMClient

llm = LLMClient(provider="openai", model="gpt-6-luna", api="responses", reasoning="medium")
agent = Agent(name="weather", llm=llm, tools=[FunctionTool(name="get_weather", fn=get_weather)])
result = agent.run("What's the weather in Paris?")
result.output          # the answer
llm.complete([...]).reasoning   # the summary of the model's reasoning, when it shows one
```

Runnable version: [`115_responses_api_and_provider_tools.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/115_responses_api_and_provider_tools.py).

## When you need it

| You want | Chat Completions (`api="chat"`) | Responses (`api="responses"`) |
|---|---|---|
| Tools on GPT-6 Astra or GPT-6.1 Sol | Refused | ✓ |
| Tools with reasoning on `gpt-6-luna` and GPT-5.4+ | Refused unless `reasoning="none"` | ✓ |
| A summary of the model's reasoning | Never returned | `LLMResponse.reasoning` and `ReasoningDelta` |
| OpenAI's own `web_search()`, `code_execution()`, `file_search()` | Refused | ✓ — see [Provider-run tools](../tools/provider-tools.md) |
| Any other OpenAI-compatible provider (Groq, OpenRouter, vLLM …) | ✓ | Not served |

OpenAI's migration guide says GPT-6 Astra and GPT-6.1 Sol "require the Responses
API for tool calling", and Chat Completions refuses tools with any
`reasoning_effort` other than `"none"` on GPT-5.4 and later. Measured on
2026-10-19, `gpt-6-luna` on Chat Completions answers a request with tools and
`reasoning="medium"` with:

```
HTTP 400: Function tools with reasoning_effort are not supported for gpt-6-luna in
/v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.
```

The SDK adds `— with fastaiagent: LLMClient(provider='openai', api='responses')`
to that error, so the fix is in the message.

## What is sent

The SDK translates its own settings to the Responses shape:

| `LLMClient` / call setting | Responses field |
|---|---|
| messages | `input` items (`system`, `user`, `assistant`, `function_call`, `function_call_output`) |
| tools | flat function tools (`{"type": "function", "name", "parameters"}`) |
| `tool_choice` | `tool_choice` |
| `reasoning=` | `reasoning.effort`, with `summary: "auto"` unless it is `"none"` |
| `output_type` / `response_format` | `text.format` |
| `max_tokens` | `max_output_tokens` |
| `temperature`, `top_p`, `parallel_tool_calls` | the same names |
| `extra_body` / `extra_headers` | merged in, as on Chat Completions |

`stop`, `seed`, `frequency_penalty` and `presence_penalty` have no Responses
field. They are not sent, and the SDK logs a warning once for each.

## Stateless, and what goes back

Every request is `store: false` and carries the whole history, as Chat
Completions does. The SDK's messages and checkpoints are the state. Nothing is
kept at OpenAI, and `previous_response_id` is not used.

The Responses API requires a reasoning model's **reasoning items** to be sent
back with the function calls they led to. They come back with an encrypted
copy of the reasoning, and the SDK carries them on the assistant message as
`ReasoningBlock(provider="openai-responses")`. They are sent back unchanged
through the tool loop, `astream`, checkpoints and resume, and only to the
Responses API. The readable part is the summary, in `LLMResponse.reasoning`.

A turn in which an OpenAI-run tool ran goes back verbatim. See
[Provider-run tools](../tools/provider-tools.md#why-the-turn-goes-back-unchanged).

## Endpoints and clients

- **`provider="openai"`**: `https://api.openai.com/v1/responses`.
- **`provider="azure"` or `"custom"`**: `{base_url}/responses`, for a deployment
  or gateway that serves it.
- **`openai_client=`**: `client.responses.create(...)` on the injected OpenAI SDK
  client, which keeps its own auth (Azure AD, managed identity).
- **Any other provider:** `api="responses"` raises `ValueError` when the client
  is built.

`api` is part of `LLMClient.to_dict()` when it is `"responses"`, so a replayed
or plane-dispatched agent calls the same endpoint (1.91.0).

## Traces and cost

A Responses call is an `llm.openai.<model>` span with the same attributes as a
Chat Completions call: tokens, cached and reasoning tokens, the reasoning summary
(payload-gated), finish reason and cost. `incomplete` with `max_output_tokens`
is `finish_reason="length"`, and a `failed` response raises `LLMProviderError`.
Tokens are priced at the model's rates, cached input at the cached rate, plus any
[provider-run tool fees](../tools/provider-tools.md#cost).
