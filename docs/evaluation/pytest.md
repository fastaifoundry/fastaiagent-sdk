# Pytest plugin

fastaiagent registers a pytest plugin that lets you express evals in the
same files as your unit tests. Each eval-tagged test runs the agent,
scores the output, and persists one `eval_runs` row to
`./.fastaiagent/local.db` so the local UI's `/evals` page picks it up
automatically.

The plugin is automatic — installing fastaiagent activates it. Tests
that don't import any of these helpers are unaffected.

## `@case` — single-row evals

```python
from fastaiagent.testing import TestModel
from fastaiagent.agent import Agent
from fastaiagent.eval import case

@case(input="hello", expected="hi")
def test_greet(evaluate_one):
    agent = Agent(name="greeter", llm=TestModel(response="hi"))
    evaluate_one(agent.run, scorers=["exact_match"])
```

`evaluate_one` is the fixture exposed by the plugin. It reads the
`@case` tag, runs the agent, scores the output, and asserts pass on
failure (with a rich error message including every scorer's score and
reason).

## `@pytest_dataset` — parametrise over a JSONL/CSV file

```python
from fastaiagent.eval import pytest_dataset as dataset

@dataset("tests/data/cases.jsonl")
def test_dataset(eval_case, evaluate_one):
    agent = Agent(name="bot", llm=...)
    evaluate_one(agent.run, scorers=["exact_match"])
```

Each row of the dataset becomes one parametrised pytest invocation.
`eval_case` is a dict (`{"input": ..., "expected_output": ...}`) and
`evaluate_one` reads it automatically when no explicit input is passed.

## `evaluate_one` reference

```python
evaluate_one(
    agent_fn,                    # the agent callable (e.g. agent.run)
    *,
    input=None,                  # overrides @case / @dataset
    expected=None,               # overrides @case / @dataset
    scorers=["exact_match"],     # str names or Scorer instances
    assert_pass=True,            # set False to inspect the record manually
    case_name=None,              # appended to the run_name for grouping
    persist=True,                # write to local.db
)
```

Returns the [`EvalCaseRecord`](../api-reference/index.md) so the test
body can do additional assertions on `actual_output`, `per_scorer`, or
`trace_id`.

## Local UI integration

Each persisted run shows up at `/evals` with
`run_name="pytest::<test-id>"`, so you can:

- Compare the same eval across CI runs over time.
- Click through to the trace if your agent recorded one
  (`AgentResult.trace_id`).

## Running

Just `pytest`. The plugin is registered via the
`[project.entry-points.pytest11]` group in fastaiagent's `pyproject.toml`,
so no opt-in flag is needed.

## See also

- [`fastaiagent.testing`](../testing/index.md) — `TestModel` and
  `FunctionModel` for deterministic agent fixtures.
- [Evaluation framework](index.md) — the underlying `evaluate()` API
  used by the plugin.
