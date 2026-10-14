# Trajectory Scoring

Evaluate the **path** an agent took — which tools it called and in what order. Trajectory scorers help you ensure agents are not only producing correct output but also following the right process.

## ToolUsageAccuracy

Did the agent use the correct tools?

```python
from fastaiagent.eval.trajectory import ToolUsageAccuracy

scorer = ToolUsageAccuracy()
result = scorer.score(
    input="", output="",
    actual_trajectory=["search", "calculate"],
    expected_trajectory=["search", "calculate", "format"],
)
# score = 2/3 = 0.667 (used 2 of 3 expected tools)
```

## StepEfficiency

Did the agent solve the problem in the expected number of steps?

```python
from fastaiagent.eval.trajectory import StepEfficiency

scorer = StepEfficiency()
result = scorer.score(
    input="", output="",
    actual_steps=6,
    expected_steps=3,
)
# score = 3/6 = 0.5 (took twice as many steps as expected)
```

## PathCorrectness

Did the agent follow the correct sequence? Uses Longest Common Subsequence to measure ordering fidelity.

```python
from fastaiagent.eval.trajectory import PathCorrectness

scorer = PathCorrectness()
result = scorer.score(
    input="", output="",
    actual_trajectory=["search", "validate", "calculate", "respond"],
    expected_trajectory=["search", "calculate", "respond"],
)
# LCS = ["search", "calculate", "respond"] → score = 3/3 = 1.0
```

## CycleEfficiency

Did the agent avoid unnecessary repeated tool calls?

```python
from fastaiagent.eval.trajectory import CycleEfficiency

scorer = CycleEfficiency()
result = scorer.score(
    input="", output="",
    actual_trajectory=["search", "search", "search", "respond"],
)
# 2 repeated consecutive calls out of 4 → score = 1.0 - 2/4 = 0.5
```

## ToolCallCorrectness

Did the agent call the right tools with the right arguments? This is stricter than `ToolUsageAccuracy` — it validates both tool name **and** arguments using deep equality.

```python
from fastaiagent.eval.trajectory import ToolCallCorrectness

scorer = ToolCallCorrectness()
result = scorer.score(
    input="", output="",
    actual_tool_calls=[
        {"name": "search", "arguments": {"query": "Paris"}},
        {"name": "format", "arguments": {"style": "markdown"}},
    ],
    expected_tool_calls=[
        {"name": "search", "arguments": {"query": "Paris"}},
        {"name": "format", "arguments": {"style": "markdown"}},
    ],
)
# score = 2/2 = 1.0 (both calls match name + args)
```

If the tool name matches but arguments differ, the call is **not** counted:

```python
result = scorer.score(
    input="", output="",
    actual_tool_calls=[{"name": "search", "arguments": {"query": "London"}}],
    expected_tool_calls=[{"name": "search", "arguments": {"query": "Paris"}}],
)
# score = 0/1 = 0.0 (wrong arguments)
```

## Using these scorers in `evaluate()`

Trajectory scorers compare the agent's **actual** tool calls against an
**expected** sequence. Since 1.89.0 `evaluate()` fills in both:

- **Actual**: when `agent_fn` returns an `AgentResult` (`agent.run`,
  `agent.arun`), its `tool_calls` become `actual_trajectory` (tool names),
  `actual_tool_calls` (names and arguments) and `actual_steps` (the count).
- **Expected**: a dataset row's own `expected_trajectory`,
  `expected_tool_calls` and `expected_steps` are forwarded to the scorers for
  that row. A run-wide `evaluate(..., expected_trajectory=[...])` still works
  and applies to rows that carry none.

```python
from fastaiagent.eval import evaluate
from fastaiagent.eval.trajectory import PathCorrectness, ToolUsageAccuracy

results = evaluate(
    agent.run,
    [
        {"input": "Where is order A17?", "expected_trajectory": ["lookup_order"]},
        {"input": "Refund order A17", "expected_trajectory": ["lookup_order", "issue_refund"]},
        {"input": "What are your opening hours?"},  # no expectation: skipped
    ],
    scorers=[ToolUsageAccuracy(), PathCorrectness()],
)
```

What each row gets:

| The row | `agent_fn` returns | Outcome |
|---|---|---|
| has an expectation | an `AgentResult` | a real verdict (an empty tool list against an expectation fails) |
| has no expectation | anything | **skipped**: the check does not apply |
| has an expectation | a plain string or dict | **errored**: there is no record of what the agent did |

Before 1.89.0 `evaluate()` passed neither side, so every trajectory scorer
compared two empty lists and **passed**: an agent that called the wrong tool,
or none, scored 1.0. A trajectory result's `details` carries the tool names
only, never the arguments, which can hold case content.

## Calling them directly

Pass the trajectory you collected yourself, e.g. from `AgentResult.tool_calls`
or the tool-call span names in a trace:

```python
from fastaiagent.eval import ToolUsageAccuracy, PathCorrectness

# Read the real tool sequence off the run. Each AgentResult.tool_calls entry is
# a dict like {"tool_name": ..., "arguments": {...}, "tool_call_id": ...}.
run = agent.run("...")
actual = [tc["tool_name"] for tc in run.tool_calls]
expected = ["search", "calculate", "respond"]

for scorer in (ToolUsageAccuracy(), PathCorrectness()):
    result = scorer.score(
        input="", output="",
        actual_trajectory=actual,
        expected_trajectory=expected,
    )
    print(scorer.name, result.score, result.passed)
```

See `examples/76_trajectory_eval.py` for a runnable end-to-end script.

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [LLM Judge](llm-judge.md) — Use an LLM to evaluate output quality
- [Session Scoring](session-scoring.md) — Evaluate multi-turn conversations
