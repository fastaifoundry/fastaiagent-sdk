# Repeated runs: evals when the model answers differently each time

A model does not give the same answer twice. An eval that runs each case once
tells you what happened on that run, not what the agent usually does: a pass
can be luck, and so can a fail. Running each case several times and reading a
**pass rate** per case ("4 of 5") separates a solid case from a flaky one.

The SDK has no built-in "repeat" option yet. This page shows how to do it with
`evaluate()` as it is today (since 1.89.0), and when it is worth the cost.

## What one run hides

A support agent with a written returns policy, five runs of four questions,
real gpt-5-mini:

```
case 1: 5/5  errored: 0  How long do I have to return an item?
case 2: 5/5  errored: 0  Can I return an item I have already opened?
case 3: 5/5  errored: 0  My parcel arrived damaged. What should I do?
case 4: 4/5  errored: 0  Who pays for return shipping?
```

The policy says nothing about return shipping. Four answers said the customer
pays; the fifth said:

> Returns are free—Northwind covers return shipping; please note items
> returned within 30 days are issued store credit (not refunds).

The agent invented a policy, once in five runs. A single eval run shows it one
time in five, and a CI gate on that run goes green the other four.

## Who decides what

| You decide | The SDK provides |
|---|---|
| The cases, and what counts as a good answer | `evaluate()`, which runs and scores each case and saves the run |
| How many runs, and on which builds | `results.cases`, in dataset order, so case *n* is the same input in every run |
| The pass rate each case must reach | `case_outcome()`, the same passed / failed / errored rule the Local UI uses |
| The model spend: N runs cost N times one run | `concurrency=` on `evaluate()` |

## The recipe

Run the dataset several times, collect each case's outcome per run, and fail
when a case falls below your rate:

```python
from collections import defaultdict

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import evaluate
from fastaiagent.eval.compare import case_outcome

agent = Agent(
    name="support",
    system_prompt=(
        "You answer for Northwind's support desk. Returns: within 30 days. "
        "Opened items get store credit, not a refund. Damaged parcels: ask "
        "the customer for a photo of the damage. Answer in one or two sentences."
    ),
    llm=LLMClient(provider="openai", model="gpt-5-mini", reasoning="minimal"),
)

RUNS = 5        # how many times each case runs
MIN_RATE = 0.8  # the pass rate every case must reach

outcomes = defaultdict(list)  # case number -> one outcome per run
for run in range(1, RUNS + 1):
    results = evaluate(
        agent.run, "cases.jsonl", scorers=["contains"], run_name=f"support-repeat-{run}"
    )
    # results.cases come back in dataset order, so case n is the same input in every run
    for n, case in enumerate(results.cases):
        outcomes[n].append(case_outcome({"error": case.error, "per_scorer": case.per_scorer}))

below = []
for n, runs in outcomes.items():
    judged = [o for o in runs if o != "errored"]  # a run that could not be scored is not a fail
    rate = judged.count("passed") / len(judged) if judged else None
    shown = "no verdict" if rate is None else f"{judged.count('passed')}/{len(judged)}"
    print(f"case {n + 1}: {shown}  errored: {runs.count('errored')}  {results.cases[n].input}")
    if rate is None or rate < MIN_RATE:
        below.append(n + 1)

if below:
    raise SystemExit(f"cases below {MIN_RATE:.0%}: {below}")
print("every case reached the pass rate")
```

`cases.jsonl` holds one case per line:

```json
{"input": "How long do I have to return an item?", "expected_output": "30 days"}
{"input": "Who pays for return shipping?", "expected_output": "customer"}
```

The script exits non-zero when a case misses the rate, so it works as a CI step
as it stands. Each run is saved like any other eval run, so all five appear in
the Local UI under their `run_name`, and you can open the answer that failed.

An errored outcome (the agent call failed, or a scorer could not run) is
counted on its own and left out of the rate: a provider outage is not a wrong
answer. A case with no verdict at all fails the step, so a broken scorer
cannot pass it.

## Choosing the number of runs and the rate

- **Where it pays off:** nightly runs and release gates. On every commit, a
  single run plus the usual [Agent CI](agent-ci.md) gate is usually enough.
- **How many runs:** 5 is a common start. More runs make a rate steadier and
  cost proportionally more model calls.
- **Which rate:** set it per kind of case. "Mostly right" may be fine for tone
  or wording. For a policy or a safety rule, one wrong answer in five is a
  defect: require 5/5, or fix the cause. In the example the fix is in the
  prompt (state who pays for return shipping), not in the threshold.
- **Lowering randomness** helps less than it seems: a lower `temperature`
  narrows the spread without removing it, and GPT-5-class reasoning models
  refuse a `temperature` setting.

## Tests of your own code are different

When a test checks your code (a tool, a guardrail, a parser), not the model's
judgement, make it deterministic instead of repeating it: give the agent a
[`TestModel`](../testing/index.md) with a canned reply, or choose inputs a
model cannot produce by itself. Repeated runs are for questions only the model
can answer.

## Limits today

- `gate()`, `pytest --eval-fail-under` and baseline comparison
  (`compare_runs`) each judge **one** run. They do not know that five runs
  belong together, so gate on the rate as in the recipe.
- In connected mode an eval run reaches the plane only with a gate verdict
  (`gate()` or the pytest gate). The recipe records none, so its runs stay
  local; the traces of each case are exported as usual.
- Rates per case live in your script. The Local UI shows each run, not the
  combined rate.
