# Agents, LLMClient and run control

Read this before writing or changing code that builds an `Agent`, picks a model, calls `run`/`arun`/`stream`/`astream`, reads an `AgentResult`, asks for structured output, or adds middleware, run limits, memory or multimodal input. Tools are in `tools.md`.

## Agent: the real constructor

```signature
fastaiagent.Agent(name, system_prompt=, llm=, tools=, guardrails=, memory=, config=, output_type=, middleware=, checkpointer=, agent_path_label=, agent_id=, prompt_slug=)
```

- `name` is the only required parameter. There is no `model=`, `instructions=`, `max_iterations=` or `temperature=` on `Agent`. Put the model on `llm=LLMClient(...)` and the loop knobs on `config=AgentConfig(...)`. Passing an unknown keyword raises `TypeError`.
- `system_prompt` takes a `str`, a registry `Prompt`, or a callable `(RunContext | None) -> str` (dynamic instructions, below).
- `llm` must be an `LLMClient` (or `TestModel`/`FunctionModel`). A raw `openai.OpenAI` client raises `TypeError`; wrap it with `LLMClient(openai_client=...)`. Omitting `llm` silently builds `LLMClient()`, which is OpenAI `gpt-4o-mini` and needs `OPENAI_API_KEY`. That default is old: pick a current model yourself (for example `gpt-5-mini`) unless the user named one, and do not change a model id the user gave you or one that already runs (`SKILL.md` rule 10).
- `tools=` takes `Tool` objects (`FunctionTool`, `RESTTool`, `MCPTool`...), not bare functions. A plain `def` in the list fails; wrap it with `@tool` or `FunctionTool` (see `tools.md`).
- `checkpointer=` (for example `SQLiteCheckpointer`) enables `resume`/`fork` and pausing via `interrupt()`.

## Run it

```signature
fastaiagent.Agent.run(input, output_type=, context=, trace=, execution_id=, messages=, metadata=, **kwargs)
fastaiagent.Agent.arun(input, output_type=, context=, trace=, execution_id=, messages=, wait_for_approval=, metadata=, **kwargs)
fastaiagent.Agent.stream(input, context=, trace=, **kwargs)
fastaiagent.Agent.astream(input, context=, trace=, execution_id=, messages=, **kwargs)
```

- `run` is sync and returns `AgentResult`. `arun` is its async twin. Use `arun` inside async code; `run` also works inside a running loop (it offloads to a thread) but blocks it.
- `stream` is sync and does NOT yield: it drains the stream and returns one `AgentResult`. Only `astream` yields events (an async generator).
- `stream` is typed `str` only. `run`/`arun`/`astream` also take an `Image`, `PDF`, or a list mixing text and those.
- `**kwargs` go to the model call (for example `temperature=0`, `response_format=...`). They are not agent options.
- `messages=` prepends prior turns (a `list[Message]`); `result.messages` is shaped to be passed straight back in. Do not also rebuild history by hand.
- `execution_id=` names the run for `resume`/`cancel`; a UUID when omitted.
- `wait_for_approval=True` raises `ValueError` (removed in 1.76.0). `False` is accepted and does nothing. Paused runs come back as `status="paused"`; the application resumes them with `agent.resume(execution_id, resume_value=Resume(approved=...))`.
- `trace=False` skips the root span. Tracing is on by default and writes local files (a `.fastaiagent/` directory under the working directory was observed, even offline).

```signature
fastaiagent.AgentResult(output=, parsed=, tool_calls=, messages=, tokens_used=, cost=, cost_known=, latency_ms=, trace_id=, execution_id=, status=, pending_interrupt=, guardrails=, stop_reason=)
```

- `output` is the final text (`str`, `""` if none). `parsed` is the `output_type` instance, or `None`.
- `status` is `"completed"`, `"paused"` or `"cancelled"`. A run ended cooperatively by middleware is `"completed"` with `stop_reason` set and `output` = the stop message.
- `cost == 0.0` alone does not mean free: check `cost_known`.
- `guardrails` lists every guardrail that ran. Non-blocking outcomes (warn, mask) are visible only here. A blocking failure raises `GuardrailBlockedError` instead of returning.

### Minimal and realistic (offline)

```python
from fastaiagent import Agent
from fastaiagent.testing import TestModel

agent = Agent(name="hello", system_prompt="Be brief.", llm=TestModel(response="hi"))
result = agent.run("anything")
assert result.output == "hi" and result.status == "completed"
```

```python
import asyncio

from fastaiagent import Agent, AgentConfig, TextDelta
from fastaiagent.testing import FunctionModel
from fastaiagent.tool import tool


@tool(name="get_weather", replay_class="read_only")
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


def responder(messages):
    # Turn 1: ask for the tool. Turn 2 (a tool message is present): answer.
    if any(m.role.value == "tool" for m in messages):
        return "It is sunny in Paris."
    return "", [{"name": "get_weather", "arguments": {"city": "Paris"}}]


agent = Agent(
    name="weather",
    system_prompt="Use tools to answer.",
    llm=FunctionModel(responder),
    tools=[get_weather],
    config=AgentConfig(max_iterations=5),
)
result = agent.run("Weather in Paris?")
assert result.output == "It is sunny in Paris."
assert result.tool_calls[0]["tool_name"] == "get_weather"


async def main() -> str:
    text = ""
    async for event in agent.astream("Weather in Paris?"):
        if isinstance(event, TextDelta):
            text += event.text
    return text


assert asyncio.run(main()) == "It is sunny in Paris."
```

## Offline test models

```signature
fastaiagent.testing.TestModel(response=, tool_calls=, usage=, model=, delay_ms=, decisions=)
fastaiagent.testing.FunctionModel(fn, model=, delay_ms=, decide_fn=)
```

- `TestModel(response="x")` always answers `"x"`. A list is round-robin and the last entry repeats. `.calls` records every request (`messages`, `tools`, `kwargs`).
- Do not use `TestModel(response=..., tool_calls=[...])` for a tool round-trip with a single response: every turn repeats the same tool call and the run ends in `MaxIterationsError`. Use `FunctionModel`, or a list: `TestModel(response=["", "done"], tool_calls=[...])` attaches the calls to the first turn only.
- `FunctionModel` calls `fn(messages)` each turn. It returns `str`, `(text, [tool_call_dict])` with `{"name", "arguments"}`, or an `LLMResponse`. Sync or async. Switch on whether a `tool` message is already present; otherwise it asks for the tool forever.
- Both report `usage=(0, 0)` tokens by default. Pass `usage=(prompt, completion)` to `TestModel` to exercise token caps.

## LLMClient

```signature
fastaiagent.LLMClient(provider=, model=, api_key=, base_url=, temperature=, max_tokens=, max_retries=, fallbacks=, reasoning=, cache=, api=, openai_client=, extra_body=, extra_headers=, **kwargs)
```

- `provider` defaults to `"openai"`. Built in: `openai`, `anthropic`, `ollama`, `azure`, `bedrock`, `custom`. Presets (key read from the provider's own env var): `gemini`, `groq`, `openrouter`, `deepseek`, `together`, `fireworks`, `perplexity`, `mistral`, `lmstudio`, `vllm`, `sambanova`, `cerebras`.
- Always pass `model=` for every provider except `openai` and the presets: the built-ins `anthropic`, `ollama`, `azure`, `bedrock`, `custom` fall back to the OpenAI id `gpt-4o-mini`, which they cannot serve.
- Never hard-code `api_key=`. Leave it unset and export the provider's env var (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`...). A missing key raises `LLMProviderError` at the first call, not at construction.
- `max_retries=0` is the default: transient provider errors are not retried unless you set it. `fallbacks=[LLMClient(...)]` are tried in order after a failure.
- `api="responses"` selects OpenAI's Responses API and is valid only for `openai`/`azure`/`custom`; other providers raise `ValueError`.
- `reasoning` accepts `none|minimal|low|medium|high|xhigh|max`; `cache=True` or `"1h"` enables prompt caching (Anthropic, Bedrock).
- Call it directly with `complete`/`acomplete`/`stream`/`astream` taking `list[Message]` (`UserMessage`, `SystemMessage`, `AssistantMessage` from `fastaiagent.llm`).

## AgentConfig: the loop and the run limits

```signature
fastaiagent.AgentConfig(max_iterations=, tool_choice=, temperature=, max_tokens=, parallel_tools=, max_parallel_tools=, output_retries=, guardrail_retries=, strict_output=, timeout_seconds=, max_total_tokens=, max_cost_usd=, output_mode=)
```

- All fields are keyword-only. Range checks raise a pydantic `ValidationError`: `max_iterations` 1-100 (default 10), `output_retries` 0-5 (default 2), `guardrail_retries` 0-5 (default 1), `max_parallel_tools` 1-64 (default 4).
- Exceeding `max_iterations` raises `fastaiagent._internal.errors.MaxIterationsError` (not exported at the top level). Raise the limit or fix the loop; do not catch-and-ignore.
- `tool_choice` is `auto|required|none`. `required` forces only the first turn (so the loop can still answer); `none` applies to every turn. Any other value raises.
- `temperature`/`max_tokens` here go on every model call of the run and override the client's own values.
- `parallel_tools=True` runs one turn's tool calls concurrently, but falls back to sequential when a checkpointer, middleware or managed governance is active. Do not rely on it alongside middleware.
- Run limits (all default `None` = unbounded), live on `AgentConfig`, not on `Agent` or middleware:
  - `timeout_seconds`: wall clock. On expiry `run` raises `AgentTimeoutError` (top-level export).
  - `max_total_tokens`: checked before every model call. Over the cap the run ends cooperatively: `result.stop_reason == "max_total_tokens"`, no exception. It reads the model's reported usage, so a model that reports none never trips it.
  - `max_cost_usd`: same shape, `stop_reason == "max_cost_usd"`. If the model cannot be priced it raises `RunLimitError` instead of running unbudgeted. Do not set it on a custom/unknown model.
  - `Agent.cancel(execution_id) -> bool` stops a run in flight; the result has `status="cancelled"`.

## Structured output

- Pass `output_type=` (any Pydantic-compatible type: `BaseModel`, `list[Model]`, primitives, unions) on `Agent`, or per call: `agent.run(text, output_type=Model)`, which types the result as `AgentResult[Model]`.
- Read `result.parsed`, not `json.loads(result.output)`. On a parse or validation failure the agent re-asks up to `output_retries` times; if it still fails, `parsed` is `None` and `output` holds the raw text. It does not raise. Always check `result.parsed is not None`. Prefer this to hand-written `response_format=`, which returns raw JSON with no parsing or retry.
- `AgentConfig(output_mode=...)` is `auto` (default), `native`, `tool` or `prompt`. Use `"tool"` for an agent that calls tools and must still return a typed answer; it adds a reserved tool named `final_result`, so no tool of yours may use that name (`ValueError`).
- `astream` also yields `PartialOutput(data=...)` while the answer forms; `data` is unvalidated.

```python
import json

from pydantic import BaseModel

from fastaiagent import Agent
from fastaiagent.testing import TestModel


class Person(BaseModel):
    name: str
    age: int


llm = TestModel(response=json.dumps({"name": "Alice", "age": 30}))
agent = Agent(name="extractor", system_prompt="Extract the person.", llm=llm, output_type=Person)
result = agent.run("Alice is 30.")
assert result.parsed is not None and result.parsed.age == 30

bad = Agent(name="x", llm=TestModel(response="not json")).run("Alice", output_type=Person)
assert bad.parsed is None and bad.output == "not json"  # no exception
```

## Dynamic instructions

- `system_prompt` may be a callable taking the run's `RunContext` (or `None` when the caller passed no `context=`). It is called on every run. Always handle `None`: `lambda ctx: ... if ctx else "default"`.
- A callable prompt cannot be serialized: `agent.to_dict()` (and therefore pushing the agent) raises `ValueError`. Use a static string or `prompt_slug` for agents that are pushed.
- `RunContext(state=obj)` must be constructed explicitly; raw objects are not auto-wrapped. `context=` also reaches tools that declare a `RunContext` parameter (see `tools.md`).

```python
from dataclasses import dataclass

from fastaiagent import Agent, RunContext
from fastaiagent.testing import TestModel


@dataclass
class Session:
    user: str


def prompt(ctx):
    return f"You help {ctx.state.user}." if ctx else "You are a general assistant."


llm = TestModel(response="ok")
agent = Agent(name="dyn", system_prompt=prompt, llm=llm)
agent.run("hi", context=RunContext(state=Session(user="Alice")))
agent.run("hi")
assert llm.calls[0]["messages"][0].content == "You help Alice."
assert llm.calls[1]["messages"][0].content == "You are a general assistant."
```

## AgentMiddleware: the one extension model

Extend an agent by subclassing `AgentMiddleware` and passing `Agent(middleware=[...])`. Do not monkeypatch `Agent`, wrap `run`, or invent callback parameters. Every hook is optional, async, and takes a `MiddlewareContext` first.

```signature
fastaiagent.AgentMiddleware.before_run(ctx, input)
fastaiagent.AgentMiddleware.before_model(ctx, messages)
fastaiagent.AgentMiddleware.wrap_model(ctx, messages, call_next)
fastaiagent.AgentMiddleware.after_model(ctx, response)
fastaiagent.AgentMiddleware.wrap_tool(ctx, tool, args, call_next)
fastaiagent.AgentMiddleware.after_run(ctx, result)
fastaiagent.AgentMiddleware.on_error(ctx, error)
fastaiagent.AgentMiddleware.on_stream_event(ctx, event)
```

- Return what the hook receives (or a replacement): `before_model` returns the message list, `after_model` the response, `after_run` the `AgentResult`. Forgetting `return` breaks the run.
- `wrap_model`/`wrap_tool` are onions and must `await call_next(...)` to proceed (`call_next(messages)` / `call_next(tool, args)`). The first middleware in the list is outermost. `before_*` run in list order; `after_*`, `on_error` and `on_stream_event` run in reverse.
- To stop a run cleanly raise `StopAgent(message, reason="...")`: the result is a normal return with `output=message` and `stop_reason=reason`. Any other exception from a hook fails the run; `on_error` observes it but cannot swallow it.
- Share data within a run through `ctx.scratch` (not saved). Anything that must survive a pause/resume goes in `ctx.state` (a JSON-safe dict, checkpointed); non-JSON state raises `TypeError` at checkpoint time.
- Class attributes `tools` (extra tools) and `instructions` (extra system text) let middleware bring both; a name clash with an agent tool raises `ValueError` at construction.
- Built in: `ToolBudget(max_calls=, message=)` (caps tool calls; ends the run with `stop_reason="tool_budget"`), `TrimLongMessages(keep_last=)`, `RedactPII(patterns=, placeholder=, entities=)`, `Reflect(llm=, facts=, criteria=, max_revisions=)`. `Reflect` fails open: a reviewer error keeps the draft and only warns.

```python
from fastaiagent import Agent, AgentMiddleware, MiddlewareContext, ToolBudget
from fastaiagent.testing import FunctionModel
from fastaiagent.tool import tool


@tool(name="ping", replay_class="read_only")
def ping(x: str) -> str:
    """Echo x."""
    return x


class CountTurns(AgentMiddleware):
    name = "count_turns"

    async def before_model(self, ctx: MiddlewareContext, messages):
        ctx.scratch["turns"] = ctx.scratch.get("turns", 0) + 1
        return messages

    async def after_run(self, ctx: MiddlewareContext, result):
        result.output = f"{result.output} (turns={ctx.scratch['turns']})"
        return result


def always_call_tool(messages):
    return "", [{"name": "ping", "arguments": {"x": "a"}}]


agent = Agent(
    name="mw",
    llm=FunctionModel(always_call_tool),
    tools=[ping],
    middleware=[CountTurns(), ToolBudget(max_calls=2, message="Budget used.")],
)
result = agent.run("go")
assert result.stop_reason == "tool_budget"
```

## Memory (brief)

`Agent(memory=...)` takes any object with `get_context(query, max_messages)` and `add(message)`. Use `Memory(...)` (recommended; keyword-only: `location`, `user_id`, `agent_id`, `window`, `learn`, `summarize`, `recall`, `semantic`, `embedder`, `max_users`...) for a durable per-user store plus a conversation window; `AgentMemory(max_messages=)` is a plain sliding window; `ComposableMemory` mixes blocks. `Memory()` defaults to a local SQLite store. For many users, pass `user_id=lambda ctx: ...` and always call with `context=RunContext(...)`: a run whose resolver returns nothing gets no conversation memory (it is not shared with other users). `learn=llm` extracts durable facts from user messages only. The full surface (tiers, summaries, recall) is in the repo's `docs/agents/memory.md`; read it before using anything beyond the snippet.

```python
from dataclasses import dataclass

from fastaiagent import Agent, Memory, RunContext
from fastaiagent.testing import TestModel


@dataclass
class Session:
    user_id: str


llm = TestModel(response="noted")
agent = Agent(name="assistant", llm=llm, memory=Memory(user_id=lambda ctx: ctx.state.user_id))
ctx = RunContext(state=Session(user_id="ana"))
agent.run("My name is Ana", context=ctx)
agent.run("What is my name?", context=ctx)
assert "My name is Ana" in [m.content for m in llm.calls[1]["messages"]]
```

## Multimodal input (brief)

- `run`/`arun`/`astream` accept a string, an `Image`, a `PDF`, or a list mixing them: `agent.run(["What is in this?", Image.from_file("cat.jpg")])`. `stream()` accepts only `str`.
- Build parts with `Image.from_file(path)` / `from_url(url)` / `from_bytes(data, media_type)`, `PDF.from_file` / `from_url` / `from_bytes`, `File.from_url` / `from_bytes`. `Image.from_bytes` needs the media type as its second argument.
- The model must support the modality. `LLMClient(pdf_mode=, max_pdf_pages=, max_image_size_mb=)` tune PDF/image handling.

## Pitfalls

- `@tool` returns a `FunctionTool`, not the function: `get_weather("x")` raises `TypeError`. Call `.execute({...})` or keep a separate plain function.
- A tool that raises does not crash the run: the model receives `Error: Tool '<name>' failed: ...` and continues. Assert on `result.tool_calls`, not only on exceptions.
- `result.output` is only the last turn's text.
