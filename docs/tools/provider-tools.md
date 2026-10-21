# Provider-run tools

*New in 1.91.0.*

`web_search()`, `code_execution()` and `file_search()` are tools the **model's
provider runs**, inside the model call. You add them to an agent like any tool.
The SDK sends each in the provider's own shape, never runs it locally, and
records every call the provider reports.

```python
from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.tool import code_execution, web_search

agent = Agent(
    name="analyst",
    llm=LLMClient(provider="anthropic", model="claude-sonnet-5-5"),
    tools=[web_search(max_uses=3), code_execution(), FunctionTool(name="get_weather", fn=get_weather)],
)
result = agent.run("What's the latest stable Python release, and the weather in Paris?")
[c["tool_name"] for c in result.tool_calls]   # ['get_weather', 'web_search']
```

Runnable version: [`115_responses_api_and_provider_tools.py`](https://github.com/fastaifoundry/fastaiagent-sdk/blob/main/examples/115_responses_api_and_provider_tools.py).

## Which providers run them

| Tool | OpenAI (`api="responses"`) | Anthropic | Gemini |
|---|---|---|---|
| `web_search()` | `web_search` | `web_search_20250305` | `google_search` |
| `code_execution()` | `code_interpreter` (auto container) | `code_execution_20250825` | `code_execution` |
| `file_search(stores)` | `file_search` (vector store ids) | — | `file_search` (file search store names) |

Every other setup refuses the request **before sending it**, with an `LLMError`
that names a setup that works. That covers OpenAI Chat Completions
(`api="chat"`), Ollama, Bedrock, and the OpenAI-compatible presets (Groq,
OpenRouter …). A provider-run tool is never silently dropped.

- **OpenAI** runs them only on the [Responses API](../llm/responses-api.md).
- **Gemini 3 and later** can use them beside your function tools. The SDK then
  sets `toolConfig.includeServerSideToolInvocations` and function-calling mode
  `VALIDATED`, which Google requires for that combination. **Gemini 2.5 and
  earlier** refuse built-in tools beside function tools (Google's 400 reads "Tool
  call context circulation is not enabled"), so the SDK refuses that pair before
  sending. One kind alone still works there.
- **Anthropic** pins the basic `web_search_20250305`. The later versions run the
  search from inside code execution, so results don't come back as search results.

### Options

| Option | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| `web_search(max_uses=)` | — | ✓ | — |
| `web_search(allowed_domains=, blocked_domains=)` | ✓ | ✓ (one of the two) | — |
| `web_search(user_location={"city", "region", "country", "timezone"})` | ✓ | ✓ | — |
| `file_search(stores, max_results=)` | ✓ | — | `stores` only |

An option the provider cannot honour raises `LLMError` before the request.

## Why the turn goes back unchanged

Each provider requires the turn in which its tool ran to be sent back exactly as
it came. Measured on 2026-10-19:

- **Anthropic.** When the model asks for a search beside one of your tools, the
  search doesn't run in that response. It runs once the turn is sent back with
  its `server_tool_use` block, and its result arrives first in the next response.
  If the block is stripped, the model searches again and both searches are billed.
  A long turn can stop with `stop_reason="pause_turn"`; the agent sends it back
  and the model carries on. Each continuation counts against `max_iterations`.
- **Gemini 3+.** The `toolCall` / `toolResponse` parts and their thought
  signatures must come back. Without them, the next request is refused (400).
- **OpenAI Responses.** The `web_search_call` / `code_interpreter_call` items go
  back with the reasoning and function calls.

The turn rides on the assistant message as `Message.provider_content` through
`run`, `astream`, checkpoints and resume, inside Chain, Swarm and Supervisor
runs. It goes back only to the provider that issued it. It is written only for a
turn in which a provider-run tool ran, so other checkpoints are unchanged. If
you keep your own history around `LLMClient.astream`, the stream's `ProviderTurn`
event carries it. `Agent.astream` consumes that event and does not pass it on.

## What is recorded

Each call the provider reports becomes:

- a **`tool.<name>` span**, with `tool.origin="provider"`,
  `fastaiagent.runner.type="tool"`, and replay class `side_effecting`. Replay never
  re-runs it; a live rerun asks the provider again. `tool.args` (the query or
  code) and `tool.result` are payloads, so they egress only with payloads on, like
  every tool's;
- a **record** in `AgentResult.tool_calls`, with the same keys as a function
  tool's (`tool_name`, `arguments`, `output` or `error`).

`output` is what the provider shows of the result:

| Tool | What you get |
|---|---|
| `web_search` | the queries, source titles and URLs, and the text the answer cites |
| `code_execution` | the code, stdout / stderr, return code, image URLs |
| `file_search` | the queries and the matched file chunks |

**No provider returns the text of the web pages it read**. Anthropic sends it
encrypted, and OpenAI returns only URLs. A guardrail can judge the sources and
cited passages, but not the full pages.

On Gemini, a search is recorded with the response that carries its sources,
which is the answer that follows the search.

## Guardrails

| Position | What happens with a provider-run tool |
|---|---|
| `tool_result` | Judges the `output` above. A rule that **rewrites** the result (`mask`, `override`) blocks instead, because the model has already read the original. A result with nothing to judge is *could-not-run*, never a pass. |
| `tool_call` | **Cannot check it.** The arguments exist only inside the model call, after the tool has run. The check reports *could-not-run*, and its `on_error` decides. |

For `tool_call` rules (decision B2-6):

- **`on_error="block"` (the default) on a blocking rule** stops the run **before
  the model is called**, with `GuardrailBlockedError`. Nothing is searched,
  executed or billed.
- **`on_error="allow"`** (or an observe-only rule) lets the tool be offered.
  Each call it makes is recorded as **unchecked** on `AgentResult.guardrails`
  and on the guardrail span.

```python
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailType

audit = Guardrail(
    name="audit_tool_args", guardrail_type=GuardrailType.code,
    position=GuardrailPosition.tool_call, fn=check_args,
    on_error="allow",   # let web_search() through, recorded as unchecked
)
```

When results stream (`astream`), the `tool_result` rules judge them at the end
of the turn, after that turn's text has streamed. Output guardrails behave the
same way.

## Cost

Provider-run tools are billed per call on top of the tokens, and the SDK prices
those fees with the tokens. They go into `AgentResult.cost`, `cost_limit`,
`max_cost_usd`, and the `llm.*` span's `fastaiagent.cost.total_usd`. Rates are
from the providers' pricing pages, read 2026-10-19:

| Fee | Price | Counted from |
|---|---|---|
| OpenAI `web_search` | $10 / 1,000 calls | `tool_usage.web_search.num_requests` |
| OpenAI `file_search` | $2.50 / 1,000 calls | `file_search_call` items |
| OpenAI `code_execution` container | $0.03 per container (1 GB, 20 min) | distinct `container_id`s, once per run |
| Anthropic `web_search` | $10 / 1,000 searches | `usage.server_tool_use.web_search_requests` |
| Gemini 3+ `google_search` | $14 / 1,000 queries | `groundingMetadata.webSearchQueries` |

Not included:

- **Gemini's 5,000 free searches a month.** The SDK prices every query, so a
  small account's figure can be higher than its bill.
- **Anthropic's code-execution container hours.** These are 1,550 free hours a
  month, then $0.05 an hour, and are free beside a 20260209+ web search or fetch
  tool. A call that used it still counts as priced.
- **OpenAI's file-search storage** ($0.10 / GB / day).

A fee the table doesn't know leaves the call **unpriced**, never guessed, so
the run's cost reads as unknown. This covers the same tool through Azure or a
custom gateway. Set your own rates with
`fastaiagent.ui.pricing.set_tool_fee_overrides({"openai/web_search": 0.008, "azure/web_search": 0.01})`.

Gemini also reports the tool's results fed back to the model as
`toolUsePromptTokenCount`. Google bills these as input, so the SDK counts them in
`prompt_tokens`.

## Elsewhere

- **MCP:** `agent.as_mcp_server(expose_tools=True)` does not expose them, because
  they run only inside a model call.
- **Chain:** a provider-run tool used as a **tool node** fails that node with an
  error that says to give the tool to an agent node instead.
- **`Agent.to_dict()`:** a provider-run tool serializes as
  `tool_type: "provider"`, with its kind and options in `config`.
