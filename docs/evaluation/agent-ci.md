# Agent CI — gate every change on agent quality

Your pytest run *is* the agent quality gate. No new config file, no second
tool, no export pipeline: the cases you already evaluate roll up into one
scored run, thresholds gate the build, and a baseline catches regressions.

The loop this closes:

```
production trace  →  fastaiagent eval curate --filter failed  →  cases.jsonl
      →  @dataset("cases.jsonl") in pytest  →  gate in CI  →  merge blocked
```

Failures become tests; tests become gates.

---

## Quick start: gate a pytest suite

Write eval cases the way you already do — `@case` for one-offs, `@dataset`
for a curated JSONL file:

```python
from fastaiagent.agent import Agent
from fastaiagent.eval import pytest_dataset as dataset

@dataset("cases.jsonl")
def test_support_agent(eval_case, evaluate_one):
    evaluate_one(agent.run, scorers=["exact_match", "faithfulness"])
```

Then gate the aggregate in CI:

```bash
pytest --eval-fail-under "overall.pass_rate=0.9" --eval-max-error-rate 0.1
```

A breach fails the session (exit 1) and prints exactly what missed:

```
=========================== fastaiagent eval ===========================
Scorecard
exact_match            avg=0.85  pass_rate=85%  (n=20)
faithfulness           avg=0.91  pass_rate=90%  (n=20)
--------------------------------------------------
overall pass_rate=88%
persisted run_id=8f2c… (Local UI: /evals/8f2c…)
overall.pass_rate: 0.8750 < 0.9 required
GATE: FAILED — quality threshold(s) missed
```

## Pytest options

| Option | Meaning |
|---|---|
| `--eval-fail-under METRIC=VALUE` | Aggregate quality threshold. Repeatable. |
| `--eval-max-error-rate RATE` | Max fraction of infra-failed (unscored) cases before the run is **invalid**. |
| `--eval-baseline RUN` | `run_id` or `run_name` to compare against. |
| `--eval-tolerance DELTA` | Allowed pass-rate drop vs the baseline (default `0.0`). |
| `--eval-run-name NAME` | Name this session's persisted run (publish a baseline). |

No options means no gating — the plugin stays invisible until you ask for it.

### Threshold grammar

| Spec | Meaning |
|---|---|
| `overall.pass_rate=0.9` | 90% of all scorer verdicts must pass |
| `geval.avg_score=0.7` | that scorer's mean score |
| `exact_match=0.9` | bare scorer name = its `pass_rate` |

A threshold naming a scorer the run didn't produce **fails** the gate — a
typo can never silently green a build.

## Infra failures can't green a build

When your agent raises (provider 500, timeout, auth error), the case is
recorded as **errored**: unscored, excluded from pass/fail, and counted
separately. This matters because unscored cases used to be invisible —
a run where 190 of 200 cases crashed and 10 passed reported a perfect
`pass_rate` of 1.0.

Now that run is **invalid**, not passing:

```
errored cases: 190 of 200 (error_rate=0.9500, max allowed 0.1) — unscored, excluded from pass/fail
GATE: INVALID — infra failures disqualify this run (not an agent regression)
```

Invalid outranks failed on purpose: during an outage, threshold misses are
noise, and reporting them as agent regressions would send you debugging the
wrong thing. The CLI gives it its own exit code (`3`) so CI can tell
"your agent got worse" apart from "the provider was down".

## Baselines: catch regressions, not absolutes

Publish a baseline from your default branch:

```bash
pytest --eval-run-name main          # on main, after merge
```

Gate pull requests against it:

```bash
pytest --eval-baseline main --eval-tolerance 0.02
```

```
baseline: main pass_rate=0.9500
current:  (current session) pass_rate=0.8000 (delta -0.1500)
regressed=3 improved=0 unchanged_pass=17 unchanged_fail=0
  regressed: "refund past 30 days" (scorers: exact_match)
GATE: REGRESSION — overall pass-rate dropped 0.1500 vs baseline (tolerance 0.02)
```

Baselines resolve by `run_id` first, then by `run_name` (latest wins), and
live in the same `local.db` the Local UI reads — so every gated run is
browsable at `/evals/<run_id>` and diffable at `/evals/compare`.

Runs also record git provenance (`git_sha`, `git_branch`, from the GitHub
Actions env or `git rev-parse`) in `eval_runs.metadata`.

## GitHub Actions

```yaml
name: agent-ci
on: pull_request

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e ".[dev]"

      # Restore the baseline DB published by main (see below).
      - uses: actions/cache@v4
        with:
          path: .fastaiagent/local.db
          key: agent-ci-baseline-${{ github.base_ref }}

      - run: pytest tests/evals
             --eval-fail-under "overall.pass_rate=0.9"
             --eval-max-error-rate 0.1
             --eval-baseline main
             --eval-tolerance 0.02
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

On `main`, run the same suite with `--eval-run-name main` and save the cache
so pull requests have something to compare against.

## The CLI

For ad-hoc runs and for gating agents you don't drive from pytest:

```bash
# Run + gate a dataset against an agent target
fastaiagent eval run \
  --agent app/agents.py:support_agent \
  --dataset cases.jsonl \
  --scorers exact_match,faithfulness \
  --fail-under "overall.pass_rate=0.9" \
  --max-error-rate 0.1 \
  --run-name main \
  --json report.json

# Compare two persisted runs
fastaiagent eval compare main pr --tolerance 0.02
```

`--agent` accepts `path/to/file.py:attr` or `pkg.module:attr`, resolving to
either a callable or any object with a `.run` method (e.g. an `Agent`).

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Gate passed |
| `1` | Quality gate failed (threshold miss or regression) |
| `3` | Run invalid — infra error rate exceeded, or nothing scored |

(`2` is reserved by the CLI framework for usage errors.)

## From production failure to gate

The demand-validated loop, end to end:

```bash
# 1. Yesterday's failures become cases (guardrail blocks, errored runs, …)
fastaiagent eval curate --filter failed --since 24 -o cases/regressions.jsonl

# 2. Fill in expected outputs for anything marked needs_review, then gate on them
pytest --eval-fail-under "overall.pass_rate=0.95"
```

`Replay` closes the same loop from the other side — fix a bug against a real
trace, then freeze the fixed behavior as a case:

```python
rerun.save_as_test("cases/regressions.jsonl",
                   input="What is our refund policy?",
                   expected_output=rerun.new_output,
                   source_trace_id=failed.trace_id)
```

Those rows load directly through `@dataset(...)`.

## Python API

The gate and comparison are library functions — use them anywhere:

```python
from fastaiagent.eval import evaluate, gate, compare_runs

results = evaluate(agent.run, "cases.jsonl", scorers=["exact_match"], run_name="pr")

report = gate(results, fail_under=["overall.pass_rate=0.9"], max_error_rate=0.1)
print(report.outcome)        # "passed" | "failed" | "invalid"
print(report.describe())     # human-readable reasons

comparison = compare_runs("main", "pr")
print(comparison.pass_rate_delta, len(comparison.regressed))
```

`EvalResults` now also exposes `errored_count` and `error_rate`, and
`Scorecard` carries `errored`, so infra failures are visible wherever you
aggregate.

## Limitations

- **Not xdist-aware.** Under `pytest -n`, each worker persists its own run
  and gates its own subset. Run eval-gated suites in a single process.
- Gating covers cases scored through `evaluate_one`; assertions you write by
  hand in a test body are ordinary pytest failures, not eval metrics.
