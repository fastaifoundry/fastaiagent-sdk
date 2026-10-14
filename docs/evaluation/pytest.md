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

The test also fails when the case is **errored** (*1.89.0*): a scorer could
not run (a judge outage, a scorer that raised) or no scorer applied (say,
`exact_match` on a case with no expected output). The message says so,
`fastaiagent eval case errored (scorer could not run, or none applied)`, so
you can tell it apart from a wrong answer. Either way the case is recorded in
the session's run; a judge that raised used to escape the fixture and leave
the case out of the run altogether. Reasons in the message are scrubbed of
credentials.

A row's `expected_trajectory`, `expected_tool_calls`, `expected_steps`,
`context` and `contexts` reach the scorers, and so do the tool calls of an
`AgentResult`, exactly as in `evaluate()`.

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

All cases in a pytest session aggregate into **one** persisted run
(v1.48.0; previously each case wrote its own single-case run). It shows
up at `/evals` named `pytest::<rootdir>` — override with
`--eval-run-name` — so you can:

- Compare the same eval across CI runs over time.
- Click through to the trace if your agent recorded one
  (`AgentResult.trace_id`).

## Gating CI

The plugin can fail the pytest session on aggregate quality:

```bash
pytest --eval-fail-under "overall.pass_rate=0.9" \
       --eval-max-error-rate 0.1 \
       --eval-baseline main --eval-tolerance 0.02
```

Infra failures (your agent raised, or a scorer could not run) are recorded
as *errored* — unscored, excluded from pass/fail, and counted — so an outage
can't green a build. A gate on a scorer that could not run makes the session
**invalid**, not failed. The session summary prints the checks rate and a
`cases:` line; gate on cases with `--eval-fail-under "cases.pass_rate=0.8"`.

The session records one verdict, and it is the one the exit status enforces
(*1.89.0*): `passed` when the session passed; `failed` for a missed threshold
or a regression against the baseline; `invalid` for an infra-disqualified run
or a baseline that could not be loaded. A malformed `--eval-fail-under` is a
usage error before any test runs.
See **[Agent CI](agent-ci.md)** for the full option list, threshold
grammar, baselines, and a GitHub Actions recipe.

## Running

Just `pytest`. The plugin is registered via the
`[project.entry-points.pytest11]` group in fastaiagent's `pyproject.toml`,
so no opt-in flag is needed. Passing no `--eval-*` options means no gating.

## See also

- [`fastaiagent.testing`](../testing/index.md) — `TestModel` and
  `FunctionModel` for deterministic agent fixtures.
- [Evaluation framework](index.md) — the underlying `evaluate()` API
  used by the plugin.
