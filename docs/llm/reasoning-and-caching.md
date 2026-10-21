# Reasoning and prompt caching

Two `LLMClient` settings, added in 1.91.0, that every provider spells differently:

```python
from fastaiagent import Agent, LLMClient

llm = LLMClient(provider="gemini", model="gemini-3.8-flash", reasoning="low", cache=True)
agent = Agent(name="support", system_prompt=LONG_POLICY, llm=llm, tools=[...])
```

- `reasoning=` sets how hard the model thinks before answering.
- `cache=` turns on prompt caching on the providers that need to be asked.

Both are off by default. An unset client sends exactly the request it sent before 1.91.0.

## Reasoning

```python
llm = LLMClient(provider="anthropic", model="claude-haiku-5-5", reasoning="medium")
response = llm.complete([UserMessage("A bat and a ball cost $1.10 ...")])

response.content                      # "$0.05" — the answer, never the reasoning
response.reasoning                    # what the model showed of its thinking, or None
response.usage["reasoning_tokens"]    # the part of completion_tokens spent thinking

llm.complete(messages, reasoning="high")   # one call can override the client
```

One vocabulary, lowest first: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`,
`"xhigh"`, `"max"`. A typo raises `ValueError` before anything is sent. Each provider
gets the level in its own field:

| Provider | What is sent |
|---|---|
| **Gemini 3.5+ / 4** | `thinkingConfig.thinkingLevel` (`minimal`–`high`; `xhigh`/`max` send `high`) and `includeThoughts`, so a thought summary comes back |
| Gemini 2.5 | `thinkingConfig.thinkingBudget`: `none` 0, `minimal` 512, `low` 1024, `medium` 8192, `high` and above 24576 |
| OpenAI and OpenAI-compatible | `reasoning_effort`, as given |
| Claude 5.x, Opus 4.6–4.8, Sonnet 4.6 | `output_config.effort` plus adaptive thinking with `display: "summarized"` (5.x models hide the text otherwise). `minimal` sends `low`, Anthropic's lowest |
| Claude Opus 4.5 | `effort` and a thinking `budget_tokens` |
| Claude Haiku 4.5, Sonnet 4.5, older | extended thinking with a `budget_tokens` (`low` 2048 … `max` 63999, always below `max_tokens`) |
| Claude, `"none"` | `thinking: {"type": "disabled"}`; on Sonnet 5.5, `between_tools` (its lowest setting) |
| Bedrock (Claude models) | the Claude fields, as `additionalModelRequestFields` |
| Ollama | `think`: a level for gpt-oss, `true`/`false` otherwise |

Which levels a model accepts varies. For example, `gpt-6-luna` takes `none`
through `xhigh`, Gemini 3.8 Flash has no `minimal`, and Opus 5.5 cannot turn
thinking off. The SDK passes your choice through and lets the provider answer,
because its 400 names the levels it supports:

```text
Unsupported value: 'reasoning_effort' does not support 'max' with this model.
Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'.
```

GPT-6 models on Chat Completions take function tools only with reasoning off.
An agent with tools and `reasoning="low"` on `gpt-6-luna` gets OpenAI's 400:
"Function tools with reasoning_effort are not supported for gpt-6-luna in
/v1/chat/completions". Use `reasoning="none"` there; the GPT-5 family takes
tools at every level.

Gemini 3 and later cannot turn thinking off. With `"none"`, the SDK sends the
lowest level, `minimal`, and logs a warning once.

**What comes back.** `LLMResponse.reasoning` holds the text the provider shows.

| Provider | Reasoning text |
|---|---|
| Anthropic | summarized thinking |
| Gemini | thought summaries |
| DeepSeek, Groq, OpenRouter, vLLM | `reasoning_content` / `reasoning` |
| OpenAI Chat Completions | none (`usage["reasoning_tokens"]` still counts it) |

Reasoning text is never part of `content`. Before 1.91.0 a Gemini thought summary
would have been glued onto the answer. Reasoning tokens are already inside
`completion_tokens` on every provider, so they are billed at the output rate.

**Streaming.** `astream` yields a `ReasoningDelta(text=...)` per chunk of reasoning,
alongside the `TextDelta`s of the answer:

```python
from fastaiagent import ReasoningDelta, TextDelta

async for event in agent.astream("Is it warmer in Paris or Rome?"):
    if isinstance(event, ReasoningDelta):
        show_thinking(event.text)
    elif isinstance(event, TextDelta):
        show_answer(event.text)
```

### Claude's thinking in the tool loop

Claude returns its thinking as `thinking` blocks carrying a signature. Claude 5.x
models think by default, without `reasoning=`. While a tool-use turn is in
progress, Anthropic requires the blocks to be sent back exactly as they came.

Before 1.91.0 the agent loop dropped them. The API does not return an error for
this; it just turns thinking off for the rest of the turn, so a Claude agent lost
its own reasoning after its first tool call. Now the blocks ride on the assistant
message (`Message.reasoning_blocks`) through `run`, `astream`, checkpoints and
resume. They go back only to the provider that issued them, because a model
cannot read another provider's reasoning. Bedrock's `reasoningContent` blocks are
carried the same way.

## Prompt caching

```python
llm = LLMClient(provider="anthropic", model="claude-sonnet-5-5", cache=True)    # 5 minutes
llm = LLMClient(provider="anthropic", model="claude-sonnet-5-5", cache="1h")    # one hour
```

| Provider | What `cache=` does |
|---|---|
| Anthropic | sends the automatic top-level `cache_control`: one breakpoint on the last cacheable block, moved forward as the conversation grows |
| Bedrock | a `cachePoint` after the system prompt, after the tools, and on the last turn |
| OpenAI, Gemini | nothing to send: both cache long prompts on their own |

Every provider's counts come back in one shape:

```python
response.usage["prompt_tokens"]        # the whole prompt, cached part included
response.usage["cached_tokens"]        # read from the cache
response.usage["cache_write_tokens"]   # written to it
```

Anthropic and Bedrock report only the uncached part as input, so the SDK adds the
cache back: `prompt_tokens` means the same thing on every provider. A prompt
shorter than the model's minimum is simply not cached. On Claude the minimum is
512 tokens for the 5.x models and 1,024 to 4,096 for older ones; on OpenAI it is
1,024.

**Cost.** Cached and written tokens are priced at their own rates. The provider
pricing pages, read 2026-10-18, give:

| Provider | Cache read | Cache write |
|---|---|---|
| OpenAI | 1/10 of input (1/20 on GPT-6.1 Sol) | 1.25× input from GPT-5.6; no charge before |
| Anthropic | 0.1× input (0.05× Opus/Sonnet 5.5, 0.025× Fable/Mythos 5.1) | 1.25× input, 2× for one hour |

A second identical prompt costs less than the first, and `AgentResult.cost`, the
`cost_limit` guardrail and the cost pages all see the difference. See
[cost tracking](../ui/cost-tracking.md#cached-written-and-long-prompts-1910).

Reasoning, a thinking agent streamed through a tool loop, and a cached prompt on
Gemini 3.8 Flash, Claude and OpenAI:
[`examples/112_reasoning_and_prompt_caching.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/112_reasoning_and_prompt_caching.py).

## In traces

Every `llm.*` span records the counts. With `FASTAIAGENT_TRACE_PAYLOADS=0` the
counts still leave the machine, but the reasoning text stays local.

| Attribute | What it holds | Exported with payloads off |
|---|---|---|
| `gen_ai.usage.reasoning_tokens` | tokens spent reasoning | yes |
| `gen_ai.usage.cache_read.input_tokens` | prompt tokens read from the cache (OpenTelemetry GenAI name) | yes |
| `gen_ai.usage.cache_creation.input_tokens` | prompt tokens written to the cache (OpenTelemetry GenAI name) | yes |
| `gen_ai.response.reasoning` | the reasoning text | no: kept locally ([security](../security.md)) |

## Saved with the agent

`reasoning` and `cache` are part of `LLMClient.to_dict()`, so a replayed or
plane-dispatched agent reasons and caches as the original did. They are written
only when set, so a default client's payload is unchanged.
