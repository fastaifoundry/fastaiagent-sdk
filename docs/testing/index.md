# Testing your agents

`fastaiagent.testing` ships two deterministic stand-ins for `LLMClient`
that swap into `Agent(llm=...)` without any HTTP. They emit the same
`StreamEvent` types as real providers and record OTel spans tagged
`gen_ai.system="test"`, so the rest of the SDK (tracing, replay, the
local UI, evals) works end-to-end against fake runs.

## TestModel — canned responses

```python
from fastaiagent.testing import TestModel
from fastaiagent.agent import Agent

agent = Agent(name="hello", llm=TestModel(response="hi"))
assert agent.run("anything").output == "hi"
```

Constructor:

```python
TestModel(
    response: str | list[str] = "ok",       # list = round-robin
    *,
    tool_calls: list[dict] | None = None,   # canned tool calls
    usage: tuple[int, int] = (0, 0),
    model: str = "test-model",
    delay_ms: int = 0,
)
```

Pass a list to round-robin through several responses on successive calls:

```python
TestModel(response=["one", "two", "three"])
```

Tool-call canned responses pair a (possibly empty) text response with
one or more tool calls:

```python
TestModel(
    response="",
    tool_calls=[{"name": "search", "arguments": {"q": "x"}}],
)
```

`tm.calls` records every invocation (`messages`, `tools`, `kwargs`) for
test assertions.

## FunctionModel — state-driven responders

`FunctionModel` wraps a callable so you can drive the LLM behaviour from
test state. Common pattern: turn 1 fires a tool, turn 2 returns the
final answer.

```python
from fastaiagent.testing import FunctionModel

state = {"calls": 0}

def responder(messages):
    state["calls"] += 1
    if state["calls"] == 1:
        return ("", [{"name": "search", "arguments": {"q": "x"}}])
    return ("done", [])

agent = Agent(name="searcher",
              llm=FunctionModel(responder),
              tools=[search_tool])
```

The responder may be sync or async, and may return:

- `str` — final answer text.
- `(text, list[tool_call_dict | ToolCall])` — text + tool calls.
- `LLMResponse(...)` — full control over the wire shape (including
  `usage`, `finish_reason`, etc.).

## When should I use these?

- **TestModel** — you know the exact response your agent should produce
  for a given input. Snapshot tests, eval harness scaffolding, simple
  smoke tests.
- **FunctionModel** — multi-turn flows where the response depends on
  conversation state, or where you want to assert the prompt your agent
  actually sends.
- **Real LLMs** — system tests where you need to verify behaviour
  against the actual provider. Gate these with
  `pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"))`.

## See also

- [`@case` and `@pytest_dataset`](../evaluation/pytest.md) — turn these
  stand-ins into first-class pytest evals.
