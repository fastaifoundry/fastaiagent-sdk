# Testing agents offline

Read this before writing or changing any test for code that builds a `fastaiagent` `Agent`, `Chain`, `Supervisor` or `Swarm`; every test here runs with no network and no API key.

## The two fakes

`fastaiagent.testing` has exactly two classes. Both subclass `LLMClient` (they are not mocks), never make an HTTP call, and emit real trace spans tagged provider `"test"`, so tools, guardrails, middleware, memory, tracing and evals run for real.

```signature
fastaiagent.testing.TestModel(response=, tool_calls=, usage=, model=, delay_ms=, decisions=)
fastaiagent.testing.FunctionModel(fn, model=, delay_ms=, decide_fn=)
```

- `TestModel(response="ok")` returns the same text on every call.
- `TestModel(response=["a", "b"])` plays a tape: call 1 returns `a`, call 2 `b`, and every later call repeats the **last** entry (sticky). An agent that over-runs its script does not fail loudly.
- `tool_calls=[{"name": ..., "arguments": {...}, "id": ...}]` attaches to the **first** turn only (`id` optional, defaults to `call_<n>`). Pair it with a list `response` to script "call a tool, then answer": `response=["", "final answer"]`.
- `usage=(prompt_tokens, completion_tokens)` sets reported tokens. `delay_ms` adds latency. `model=` only changes the name shown in traces.
- `decisions=` supplies canned results for `llm.decide()` (the Decisions API); without it `decide()` raises `ValueError`.
- `FunctionModel(fn)`: `fn(messages)` gets the whole `list[Message]` and may be sync or async. Return `"text"`, `("text", [tool_call_dict_or_ToolCall, ...])` (use `""` for a tool-call-only turn), or an `LLMResponse`. Anything else raises `TypeError`.
- Both record every call in `.calls`: a list of `{"messages": [...], "tools": [...] | None, "kwargs": {...}}`. Assert on it to check the prompt the agent really sent. Messages have `.role` (a `MessageRole`; compare `.role.value`) and `.content`.

Do use `TestModel` when the reply is fixed and `FunctionModel` when the reply depends on the conversation or you need turn state. Don't script a multi-turn tool flow with a single-string `TestModel(response="", tool_calls=[...])` — the sticky last turn repeats the tool call forever and the run dies with a max-iterations error (default limit 10, `AgentConfig(max_iterations=)`).

## Minimal offline test

```python
from fastaiagent import Agent
from fastaiagent.testing import TestModel

def test_agent_answers():
    llm = TestModel(response="Paris")
    agent = Agent(name="geo", system_prompt="Answer in one word.", llm=llm)
    result = agent.run("Capital of France?")
    assert result.output == "Paris"
    sent = llm.calls[0]["messages"]
    assert [(m.role.value, m.content) for m in sent] == [
        ("system", "Answer in one word."),
        ("user", "Capital of France?"),
    ]

test_agent_answers()
```

## Tool calls: script them and assert on them

```python
from fastaiagent import Agent
from fastaiagent.testing import FunctionModel, TestModel
from fastaiagent.tool import tool

@tool()
def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"sunny in {city}"

def test_tool_scripted_with_testmodel():
    llm = TestModel(
        response=["", "It is sunny in Paris."],
        tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
    )
    result = Agent(name="w", llm=llm, tools=[get_weather]).run("Weather in Paris?")
    assert result.output == "It is sunny in Paris."
    assert [c["tool_name"] for c in result.tool_calls] == ["get_weather"]
    assert result.tool_calls[0]["arguments"] == {"city": "Paris"}
    assert result.tool_calls[0]["output"] == "sunny in Paris"
    assert len(llm.calls) == 2                       # one turn per script entry
    assert llm.calls[1]["messages"][-1].content == "sunny in Paris"   # tool result fed back

def test_tool_scripted_with_functionmodel():
    def responder(messages):
        if messages[-1].role.value == "tool":        # branch on conversation state
            return f"Done: {messages[-1].content}"
        return "", [{"name": "get_weather", "arguments": {"city": "Rome"}}]

    result = Agent(name="w", llm=FunctionModel(responder), tools=[get_weather]).run("go")
    assert result.output == "Done: sunny in Rome"

test_tool_scripted_with_testmodel()
test_tool_scripted_with_functionmodel()
```

`AgentResult.tool_calls` is a list of dicts with keys `iteration`, `tool_name`, `arguments`, `tool_call_id`, plus `output` on success or `error` on failure. A tool that raises, or a name the agent does not have, does **not** raise out of `run()`: the error text goes back to the model as the tool message and lands in the `error` key. Assert on it:

```python
from fastaiagent import Agent
from fastaiagent.testing import TestModel
from fastaiagent.tool import tool

@tool()
def flaky(x: str) -> str:
    """Always fails."""
    raise ValueError("nope")

def test_tool_failure_is_recorded():
    llm = TestModel(response=["", "recovered"], tool_calls=[{"name": "flaky", "arguments": {"x": "a"}}])
    result = Agent(name="f", llm=llm, tools=[flaky]).run("go")
    assert result.output == "recovered"
    assert "nope" in result.tool_calls[0]["error"]

test_tool_failure_is_recorded()
```

## Overriding the LLM of an existing agent

Do build agents through a factory that takes the model, so tests pass a fake and production passes the real client:

```python
from fastaiagent import Agent, LLMClient
from fastaiagent.testing import TestModel

def build_agent(llm=None) -> Agent:
    return Agent(
        name="support",
        system_prompt="You are a support agent.",
        llm=llm or LLMClient(provider="openai", model="gpt-4o-mini"),
    )

def test_support_agent():
    agent = build_agent(llm=TestModel(response="Refund issued."))
    assert agent.run("refund please").output == "Refund issued."

test_support_agent()
```

If the module already builds a module-level agent, assign the attribute: `agent.llm = TestModel(response="...")` (`Agent.llm` is a plain attribute read on each run; verified). Restore it with `monkeypatch.setattr(agent, "llm", fake)` so the swap is undone after the test. Don't rebuild the agent with copied arguments; you will drop tools or guardrails.

`Agent(llm=None)` silently creates a default `LLMClient()` (real provider), so always pass the fake explicitly.

## Failure paths

`FunctionModel` can raise to simulate a provider outage; the exception propagates out of `run()` unchanged:

```python
import pytest
from fastaiagent import Agent
from fastaiagent.testing import FunctionModel

def test_provider_outage_propagates():
    def down(messages):
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        Agent(name="x", llm=FunctionModel(down)).run("hi")

test_provider_outage_propagates()
```

## Async tests

`agent.arun(...)` is the coroutine form of `agent.run(...)`. Either wrap it in `asyncio.run` inside a sync test (no extra dependency), or use `pytest-asyncio` (`pip install pytest-asyncio`, then `@pytest.mark.asyncio`). Don't call `agent.run()` from inside a running event loop in an async test: use `await agent.arun()`.

```python
import asyncio
import pytest
from fastaiagent import Agent
from fastaiagent.testing import TestModel

def test_arun_with_asyncio_run():
    agent = Agent(name="a", llm=TestModel(response="async ok"))
    assert asyncio.run(agent.arun("hi")).output == "async ok"

@pytest.mark.asyncio
async def test_arun_with_pytest_asyncio():
    agent = Agent(name="a", llm=TestModel(response="async ok"))
    result = await agent.arun("hi")
    assert result.output == "async ok"

test_arun_with_asyncio_run()
asyncio.run(test_arun_with_pytest_asyncio())
```

Streaming: `agent.astream(input)` yields stream events (`TextDelta`, `ToolCallStart`, `ToolCallEnd`, `Usage`, `StreamDone`, importable from `fastaiagent.llm.stream`). Both fakes support it.

## Keep tests from writing to the project database

Every run writes traces to `./.fastaiagent/local.db` (path set by `FASTAIAGENT_LOCAL_DB`; the `.fastaiagent/` directory is created even when the DB is moved). Ignore `.fastaiagent/` in `.gitignore`. To isolate plain unit tests, add this to `conftest.py`:

```python
import pytest

@pytest.fixture(autouse=True)
def isolated_local_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
```

Don't use that fixture for eval-gated suites (`evals.md`): the aggregated eval run is persisted at session end, after the per-test patch is undone, so it lands in the default DB and baselines are never found. For those, export `FASTAIAGENT_LOCAL_DB` in the shell or CI step instead.

## Real-model tests

Do keep real-provider tests out of the default run. Mark them and skip without a key; never hard-code a key and never read one from a file in the repo:

```python
# needs: OPENAI_API_KEY
import os
import pytest

needs_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

@pytest.mark.e2e
@needs_openai
def test_real_model_answers():
    from fastaiagent import Agent, LLMClient

    agent = Agent(name="real", llm=LLMClient(provider="openai", model="gpt-4o-mini"))
    assert agent.run("Reply with the single word: pong").output.strip()
```

Register the marker (`markers = ["e2e: needs a real model key"]` under `[tool.pytest.ini_options]`) so `pytest -m "not e2e"` is clean. A skipped test is not a check: report skips explicitly instead of counting them as passes.

## What not to do

- Don't mock `LLMClient`, `httpx` or `openai` with `unittest.mock`. The fakes exercise the real loop; a mock does not and produces no trace.
- Don't put real keys in tests, fixtures or examples. Offline tests need none; `TestModel` ignores `api_key`.
- Don't assert on a real model's wording. Assert on tool calls, arguments, the messages sent (`.calls`), guardrail outcomes and structure. Score wording with evals (`evals.md`).
- Don't import `MockLLMClient`, `isolated_local_db` or `CaptureServer`: they are the SDK repo's own test scaffolding, not part of the installed package.
- Don't treat a green offline suite as proof the prompt works with a real model: it validates your wiring only.
- `pytest` is not a dependency of `fastaiagent`; install it yourself. The pytest plugin (`evals.md`) registers automatically once both are installed and does nothing for tests that don't use it.
