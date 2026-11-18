# Evals, scorers, pytest gates and the eval CLI

Read this when you score an agent against a dataset, write a scorer or an LLM judge, add `@case` / `evaluate_one` tests, gate CI on quality, or run `fastaiagent eval`.

## evaluate()

```text
from fastaiagent.eval import evaluate
evaluate(agent_fn, dataset, scorers=None, concurrency=4, persist=True, run_name=None, dataset_name=None, agent_name=None, run_kind="evaluate", **kwargs)
```

```signature
fastaiagent.eval.Dataset.from_jsonl(path)
fastaiagent.eval.Dataset.from_csv(path)
fastaiagent.eval.Dataset.from_list(items)
fastaiagent.eval.Dataset.to_jsonl(path, append=)
```

- `agent_fn` is any callable taking the input and returning a string or an `AgentResult` (`agent.run` is the usual value). Sync callables run on a worker thread; async callables are awaited. `dataset` is a `Dataset`, a list of dicts, or a `.jsonl` / `.csv` path (anything else is read as JSONL).
- Row keys: `input` (required), `expected_output` or `expected`, and the per-row scorer inputs `expected_trajectory`, `expected_tool_calls`, `expected_steps`, `context`, `contexts`. Other keys are ignored. Extra `**kwargs` to `evaluate()` go to every scorer for every row; a row's own value wins.
- `scorers` defaults to `["exact_match"]`. Names are strings or `Scorer` instances. An unknown name, a name needing constructor arguments (`regex_match`), or two scorers with the same `name` raises `EvalError` before any agent call.
- `persist=True` (default) writes the run to `./.fastaiagent/local.db`. Pass `persist=False` in tests and one-off scripts. The async twin is `from fastaiagent.eval.evaluate import aevaluate`.
- An agent that raises or pauses (approval, `interrupt()`) makes the case **errored**: it is not scored and not counted as a quality failure. A scorer that raises is recorded as errored for that case; it does not abort the run.

Every scorer verdict has an `outcome`: `passed`, `failed`, `errored` (could not run: judge outage, unparseable reply, scorer bug) or `skipped` (does not apply: no expected output, no trajectory). Rates exclude errored and skipped. Never read an errored check as a failure or as a pass.

```python
from fastaiagent import Agent
from fastaiagent.eval import evaluate, gate
from fastaiagent.testing import FunctionModel

agent = Agent(name="capitals", llm=FunctionModel(
    lambda msgs: "Paris" if "France" in str(msgs[-1].content) else "Rome"))

cases = [
    {"input": "Capital of France?", "expected_output": "Paris"},
    {"input": "Capital of Italy?", "expected": "Rome"},
    {"input": "Capital of Spain?", "expected_output": "Madrid"},
]
results = evaluate(agent.run, cases, scorers=["exact_match", "contains"], persist=False)
print(results.summary())
report = gate(results, fail_under=["exact_match=0.5"], max_error_rate=0.1)
assert report.outcome == "passed", report.describe()
assert gate(results, fail_under=["exact_match=0.9"]).outcome == "failed"
```

## Built-in scorers

String names usable in `scorers=[...]` and `--scorers`:

- Offline, no key: `exact_match`, `contains` (case-insensitive substring), `json_valid`, `length_between`, `latency`, `cost_under`, `pii_leakage`, `prompt_injection`, `bleu`, `rouge`, `levenshtein`.
- Need an LLM (default `LLMClient()` unless you build the instance with `llm=`): `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `toxicity`, `bias`, `task_completion`, `hallucination`, `reflection_quality`. `moderation` calls the OpenAI moderation API.
- `semantic_similarity` embeds with fastembed if installed, else OpenAI, else a crude `SimpleEmbedder` (silent fallback; scores then mean little).
- Needs an argument, so instance only: `RegexMatch(pattern)`. Pass instances to set non-default options on the others, e.g. `LengthBetween(min_len=, max_len=)`, `Latency(max_ms=)`, `CostUnder(max_usd=)`, `Faithfulness(llm=, threshold=)`.
- Instances only, not resolvable by name: `LLMJudge`, `GEval`, `DecisionJudge`, trajectory scorers (`ToolUsageAccuracy`, `StepEfficiency`, `PathCorrectness`, `CycleEfficiency`, `ToolCallCorrectness`) and session scorers (`ConversationCoherence`, `GoalCompletion`, `KnowledgeRetention`, `RoleAdherence`, `ConversationRelevancy`). All import from `fastaiagent.eval`.
- Trajectory scorers read the agent's real tool calls from the `AgentResult` plus the row's `expected_trajectory`; with no expected value they are `skipped`, and with a plain-string agent output they are `errored`. `evaluate()` does not forward a row's `conversation`: pass it as an `evaluate(..., conversation=...)` keyword or call the session scorer's `.score(...)` yourself.
- `latency` and `cost_under` use run facts `evaluate()` supplies. `cost_under` is **errored** (not passed) when the model has no known price, because unknown cost is not free cost.

## Custom scorers

```signature
fastaiagent.eval.Scorer.code(name=)
fastaiagent.eval.ScorerResult.error(reason, **details)
fastaiagent.eval.ScorerResult.not_applicable(reason, **details)
```

`@Scorer.code("name")` turns a plain function into a scorer. The function gets keyword arguments `input`, `output`, `expected` (optional) and may declare any of the extra kwargs it wants (`latency_ms`, `cost`, `actual_trajectory`, `context`, ...); undeclared extras are dropped unless it takes `**kwargs`. Return a `bool` (passed), a `float` (score; passed when `>= 0.5`), a `ScorerResult`, or anything else (recorded as a failed verdict with `str(result)` as the reason). Use `ScorerResult.not_applicable(...)` when the row lacks what you need and `ScorerResult.error(...)` when the check could not run. Don't return a clean pass for a check that could not run.

```python
from fastaiagent.eval import Scorer, ScorerResult, evaluate

@Scorer.code("is_short")
def is_short(input, output, expected=None):
    return len(output) < 20

@Scorer.code("mentions_expected")
def mentions_expected(input, output, expected=None, **kwargs):
    if expected is None:
        return ScorerResult.not_applicable("row has no expected output")
    ok = expected.lower() in output.lower()
    return ScorerResult(score=1.0 if ok else 0.0, passed=ok, reason=f"wanted {expected!r}")

results = evaluate(
    lambda text: "Paris",
    [{"input": "Capital of France?", "expected": "Paris"}],
    scorers=[is_short, mentions_expected],
    persist=False,
)
assert results.cases[0].per_scorer["mentions_expected"]["passed"] is True
```

Subclass `Scorer` (set `name`, implement `score(self, input, output, expected=None, **kwargs) -> ScorerResult`) when the scorer needs state or constructor arguments.

## LLM judge

```signature
fastaiagent.eval.LLMJudge(criteria=, prompt_template=, llm=, scale=, evaluation_steps=, rubric=, threshold=, name=, auto_steps=)
fastaiagent.eval.GEval(name=, criteria=, evaluation_steps=, rubric=, scale=, threshold=, llm=, auto_steps=)
```

`LLMJudge(criteria="correctness", llm=...)` makes one call, normalises the verdict from `scale` (`"binary"` default, `"0-1"`, `"1-5"`, `"1-10"`) to 0-1, and passes when `score >= threshold` (default 0.5). Passing `evaluation_steps` or `rubric` (list of `(score, description)`) switches to G-Eval mode; `GEval` is that mode by default on a `1-5` scale. A custom `prompt_template` must contain `{input}`, `{output}` or `{expected}`, else the judge never sees the content (a `UserWarning`). Always pass `llm=` explicitly; with `llm=None` it builds a default `LLMClient()` that needs a provider key. An unparseable reply is retried once, then recorded as errored; a judge outage is errored too, so a gate on it is `invalid`, not `failed`. In offline tests use a `TestModel` that returns the JSON verdict:

```python
from fastaiagent.eval import LLMJudge
from fastaiagent.testing import TestModel

judge = LLMJudge(
    criteria="helpfulness",
    llm=TestModel(response='{"score": 1, "reasoning": "answers the question"}'),
)
verdict = judge.score(input="2+2?", output="4", expected="4")
assert verdict.passed and verdict.outcome == "passed"

bad = LLMJudge(llm=TestModel(response="no json here")).score(input="q", output="a")
assert bad.outcome == "errored" and not bad.passed
```

## Gates

```text
from fastaiagent.eval import gate
gate(results, *, fail_under=None, max_error_rate=None) -> GateReport
```

`gate()` returns a `GateReport` with `.outcome` (`"passed"`, `"failed"`, `"invalid"`), `.describe()` (reason lines), `.checks`, `.scored`, `.errored`, `.error_rate`. Threshold strings, all `metric=value`:

- `overall.pass_rate=0.9`: share of applicable checks passed across all scorers.
- `cases.pass_rate=0.8`: share of cases where every applicable check passed.
- `<scorer>.avg_score=0.7` or `<scorer>.pass_rate=0.9`; a bare `<scorer>=0.9` means its pass rate.
- `overall.avg_score` is not a metric: it raises `EvalError`.

A run is `invalid` (not `failed`) when `max_error_rate` is exceeded, nothing was scored, a gated scorer errored on some cases and `max_error_rate` is unset or exceeded, or a gated named scorer applied to no case. `invalid` outranks `failed`. A threshold naming a scorer the run did not produce **fails**, so a typo never greens a build. Do set `max_error_rate` whenever any scorer is an LLM judge or the dataset can trigger approvals. Don't gate on one run when the agent is stochastic and the dataset is small: a single run can pass or fail on the model's luck.

**What an offline eval proves.** With `TestModel` or `FunctionModel` the agent's answers are
scripted, so a gate over them proves the wiring (the dataset loads, the scorer runs, the gate
fails when it should) and never the quality of the agent. A scripted model that echoes the tool's
output scores 100% by construction. Say that in your summary, and keep the quality eval as a
separate `e2e`-marked run against a real model (skipped without a key, and reported as skipped).

## pytest plugin

Installing `fastaiagent` plus `pytest` registers the plugin automatically (entry point `pytest11`); tests that do not use it are unaffected. `pytest` itself is not a fastaiagent dependency. Not xdist-aware: run eval-gated suites in one process.

```python
from fastaiagent import Agent
from fastaiagent.eval import case, pytest_dataset
from fastaiagent.testing import FunctionModel

agent = Agent(name="capitals", llm=FunctionModel(
    lambda msgs: "Paris" if "France" in str(msgs[-1].content) else "Rome"))

@case(input="Capital of France?", expected="Paris")
def test_france(evaluate_one):
    evaluate_one(agent.run, scorers=["exact_match", "contains"])

@pytest_dataset("cases.jsonl")          # one pytest test per row; file read at import time
def test_rows(eval_case, evaluate_one):  # eval_case is the row dict
    evaluate_one(agent.run, scorers=["exact_match"])
```

```signature
fastaiagent.eval.case(input, expected=, name=)
fastaiagent.eval.pytest_dataset(path, ids_from=)
```

`evaluate_one(agent_fn, *, input=None, expected=None, scorers=None, assert_pass=True, case_name=None, persist=True)` is a fixture returning a callable. Input and expected resolve from explicit args, then `@case`, then the dataset row. It returns the `EvalCaseRecord` (`.actual_output`, `.per_scorer`, `.trace_id`). Rules:

- Pass a **sync** callable (`agent.run`). `agent.arun` is not awaited: its coroutine is stringified and the case just fails.
- The test fails on a failed case, and also on an errored case (scorer could not run, none applied, agent raised or paused); the message says which. Pass `assert_pass=False` to inspect the record yourself.
- `scorers=None` means `exact_match`. A case with no expected value then has no applicable check and is errored; pass `contains`, `json_valid` or an instance that does not need `expected`.
- `case_name` and `@case(name=)` are accepted but have no effect (verified in source). `persist=False` keeps the case out of the stored run but it still counts for the gates.
- Row keys `expected_trajectory`, `context`, ... and the agent's tool calls reach scorers as in `evaluate()`.

All `evaluate_one` cases in a session aggregate into ONE stored run named `pytest::<rootdir name>`. Options (no option means no gating, but the run is still stored):

| Option | Meaning |
|---|---|
| `--eval-fail-under METRIC=VALUE` | Threshold (grammar above). Repeatable. Malformed value is a usage error before any test runs. |
| `--eval-max-error-rate RATE` | Max fraction of errored cases before the session verdict is `invalid`. |
| `--eval-baseline RUN` | `run_id` or `run_name` of a stored run to compare with. |
| `--eval-tolerance DELTA` | Allowed overall pass-rate drop vs the baseline (default `0.0`). |
| `--eval-run-name NAME` | Name this session's stored run (default `pytest::<rootdir name>`). |

A breached gate raises a passing session's exit status to 1 and prints `GATE: ...` lines and a last line `fastaiagent eval verdict: PASSED|FAILED|INVALID`. It never lowers a failing status. A name baseline skips stored runs whose verdict was `failed` or `invalid`, so a regression cannot become the next baseline. `--eval-run-name main --eval-baseline main` is a ratchet: the baseline is resolved before this session is stored, and the first run (no accepted `main` yet) publishes it, gated on thresholds only. A baseline that cannot be loaded makes the session `invalid`.

Publish on the default branch, gate pull requests against it, and keep ONE database across both steps:

```bash
export FASTAIAGENT_LOCAL_DB=.ci/local.db      # same path in both jobs (cache it between runs)
pytest tests/evals --eval-run-name main
pytest tests/evals --eval-fail-under "overall.pass_rate=0.9" --eval-max-error-rate 0.1 \
       --eval-baseline main --eval-tolerance 0.02
```

Don't isolate the DB per test with a monkeypatch fixture for these suites: the session run is stored after the patch is undone, so it lands in the default DB and baselines are never found.

## CLI

```bash
fastaiagent eval run --agent app/agents.py:support --dataset cases.jsonl \
  --scorers exact_match,contains --fail-under "overall.pass_rate=0.9" \
  --max-error-rate 0.1 --run-name pr --json report.json
fastaiagent eval compare main pr --tolerance 0.02
```

`eval run` options: `--dataset` and `--agent` (both required; `path/file.py:attr` or `pkg.module:attr`, an `Agent` or a callable), `--scorers` (comma-separated built-in names, default `exact_match`; no instances, so no `regex_match`, no `LLMJudge`), `--concurrency` (4), `--run-name`, `--fail-under` (repeatable), `--max-error-rate`, `--json PATH` (report with `scorecard` and `gate`), `--no-persist`, `--db`. `eval compare BASELINE CURRENT` takes run ids or names (latest wins); `--tolerance`, `--fail-on-regression/--no-fail-on-regression`, `--db`. Also `eval curate` (turn captured traces into a dataset) and `eval export --dry-run` (print what a connected run would send).

| `eval run` exit | Meaning |
|---|---|
| `0` | Gate passed (also when no `--fail-under` and something was scored) |
| `1` | Quality gate failed (including a misspelled scorer in `--fail-under`) |
| `2` | Usage error from the CLI: missing option, malformed `--agent` spec, or the module has no such attribute |
| `3` | Run invalid: error rate over `--max-error-rate`, nothing scored, gated scorer could not run, unknown or argument-needing scorer name, `overall.avg_score`, dataset file missing, agent module fails to import |

`eval compare` exits `1` on a pass-rate drop beyond `--tolerance` and `3` when a run is not found.
