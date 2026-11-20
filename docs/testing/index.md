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
    decisions: DecisionResult | dict | list | None = None,  # canned decide() results
)
```

`decisions=` serves canned [Decisions API](../llm/decisions.md#testing-without-the-network)
results (one for every call, or a list served in order, the last repeating).
Without it, `decide()` raises.

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

The tool calls attach to the **first** canned response. With a single string the
last turn repeats, so an agent keeps calling the tool until it hits the
max-iterations error. To call a tool and then answer, pass a list:
`TestModel(response=["", "final"], tool_calls=[...])`.

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

Constructor:

```python
FunctionModel(
    fn,                          # the responder, below
    *,
    model: str = "function-model",
    delay_ms: int = 0,
    decide_fn=None,              # (input, questions) -> DecisionResult | dict, for decide()
)
```

Without `decide_fn`, `decide()` raises. See
[Decisions: testing without the network](../llm/decisions.md#testing-without-the-network).

The responder may be sync or async, and may return:

- `str` — final answer text.
- `(text, list[tool_call_dict | ToolCall])` — text + tool calls.
- `LLMResponse(...)` — full control over the wire shape (including
  `usage`, `finish_reason`, etc.).

## Platform guardrail rules, offline

*New in 1.105.0.* A connected agent enforces the guardrail rules its platform authored
([managed governance](../guardrails/managed-governance.md)). `plane_rules` runs a block as if the
platform had sent a set of rules, so you can test how your app handles a block with no platform:

```python
import pytest
from fastaiagent import Agent, GuardrailBlockedError
from fastaiagent.testing import TestModel, plane_rules

RULE = {
    "name": "no-codenames",
    "guardrail_type": "input",            # input | output | tool
    "implementation_type": "regex",
    "config": {"pattern": r"(?i)\bbluejay\b", "should_match": False},
    "validation_mode": "blocking",
    "on_error": "block",
}

def test_a_plane_block_reaches_the_caller():
    agent = Agent(name="support", llm=TestModel("ok"))
    with plane_rules([RULE]):
        with pytest.raises(GuardrailBlockedError) as err:
            agent.run("tell me about bluejay")
    assert err.value.guardrail_name == "no-codenames"
```

The rules use the platform's policy shape and pass through the same cache and builder as a real
policy pull, so a rule gives the same verdict here as on a connected run.

- A rule with `agent_ids` applies only to an `Agent(agent_id=...)` with a matching id; a rule
  without `agent_ids` applies to every agent, as on the platform.
- A rule that cannot be enforced from its configuration (a `code` rule, whose logic lives on the
  platform) raises `ValueError`. A connected agent would skip it, and a test that silently checked
  nothing would be worse than one that fails.
- The policy is process-wide, like the real one: do not nest `plane_rules` or use it from parallel
  threads. The previous policy is restored when the block exits.

!!! warning "What it does not prove"
    `plane_rules` tests **your app's handling** of a platform rule. It does not show that your
    platform has such a rule, or that a connected agent receives it. Keep a connected test for
    that. Do not write to `fastaiagent.client._connection` in tests: it is private state.

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

An eval run that `TestModel` or `FunctionModel` answered or judged proves the
wiring, not the agent, so a connected SDK keeps it local: it is never sent to the
plane as evidence (*since 1.104.0*; see
[Connected eval export](../platform/connected-eval-export.md)).

## See also

- [`@case` and `@pytest_dataset`](../evaluation/pytest.md) — turn these
  stand-ins into first-class pytest evals.
