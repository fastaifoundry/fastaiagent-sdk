# Evaluation

The eval framework lets you systematically test agents against datasets with multiple scorers. It runs entirely offline — no cloud service required. Supports built-in scorers, LLM-as-judge, custom code scorers, trajectory evaluation, and multi-turn session scoring.

## Quick Start

```python
from fastaiagent.eval import evaluate

def my_agent(input_text: str) -> str:
    # Your agent logic here
    return input_text.upper()

results = evaluate(
    agent_fn=my_agent,
    dataset=[
        {"input": "hello", "expected": "HELLO"},
        {"input": "world", "expected": "WORLD"},
    ],
    scorers=["exact_match"],
)

print(results.summary())
# Evaluation Results
# ==================================================
# exact_match: avg=1.00 pass_rate=100% (2 cases)
```

## The evaluate() Function

```python
from fastaiagent.eval import evaluate

results = evaluate(
    agent_fn=my_agent,          # Any callable: function, agent.run, lambda
    dataset=dataset,             # Dataset, file path, or list of dicts
    scorers=["exact_match"],     # Built-in names or Scorer instances
    concurrency=4,               # Parallel evaluation (default: 4)
)
```

**agent_fn** accepts any callable that takes a string and returns a string (or an object with `.output`):

```python
# Plain function
evaluate(agent_fn=lambda x: x.upper(), dataset=dataset)

# Agent.run
evaluate(agent_fn=my_agent.run, dataset=dataset)

# Custom wrapper
def run_pipeline(input_text):
    result = chain.execute({"message": input_text})
    return result.output
evaluate(agent_fn=run_pipeline, dataset=dataset)
```

## Async

Every entry point has an `a`-prefixed coroutine — `aevaluate`, `asimulate`,
`agenerate_scenarios`, `aharden` — for use inside async apps (FastAPI, etc.); the
sync versions just wrap them. `aevaluate` lives in `fastaiagent.eval.evaluate`:

```python
from fastaiagent.eval.evaluate import aevaluate

results = await aevaluate(agent.run, dataset, scorers=["contains"])
```

See `examples/79_async_eval.py` for the full async loop.

## Datasets

### From a List

```python
from fastaiagent.eval import Dataset

dataset = Dataset.from_list([
    {"input": "What is 2+2?", "expected": "4"},
    {"input": "Capital of France?", "expected": "Paris"},
])
```

### From JSONL

```
{"input": "What is 2+2?", "expected": "4"}
{"input": "Capital of France?", "expected": "Paris"}
```

```python
dataset = Dataset.from_jsonl("test_cases.jsonl")
```

#### From a failing trace (Replay → regression test)

Failed production traces are the most valuable test cases — they're
the bugs your users actually hit. Once you've debugged one with
[Replay](../replay/index.md), `ReplayResult.save_as_test()` appends
the corrected case directly to the JSONL dataset `evaluate()` reads:

```python
rerun = replay.fork_at(step=3).modify_prompt("...").rerun()
rerun.save_as_test(
    "regression_tests.jsonl",
    input="...",
    expected_output=str(rerun.new_output),
    source_trace_id=failure.trace_id,  # provenance back to the bug
)
# Same file, now ready for evaluate()
results = evaluate(
    agent_fn=agent.run,
    dataset="regression_tests.jsonl",
    scorers=["exact_match"],  # or LLMJudge for semantic checks
)
```

The Local UI's **Save as regression test** button calls the same
underlying endpoint and writes an identical record — UI-saved and
code-saved cases are interchangeable. See
[Replay → From a Rerun to a Regression Test](../replay/index.md#from-a-rerun-to-a-regression-test)
for the full walkthrough.

#### From captured traces in bulk (curation)

To turn many captured traces into a dataset at once — by favorites, notes,
guardrail-fired, or all — use `Dataset.from_traces(...)` or
`fastaiagent eval curate`. See [Trace Curation](curation.md).

### From CSV

```csv
input,expected
What is 2+2?,4
Capital of France?,Paris
```

```python
dataset = Dataset.from_csv("test_cases.csv")
```

### Dataset Item Fields

Each item is a dict. The only required field is `input`. Other common fields:

| Field | Used By | Description |
|-------|---------|-------------|
| `input` | All scorers | The input to send to the agent |
| `expected` or `expected_output` | ExactMatch, Contains, LLMJudge | The expected correct answer |
| `conversation` | Session scorers | Multi-turn chat history |
| `expected_trajectory` | Trajectory scorers | Expected tool call sequence |
| `tags` | Filtering | Labels for grouping test cases |

> **What `evaluate()` forwards to scorers:** per case, the eval loop passes only
> `input` and `expected`/`expected_output`. Fields like `conversation`,
> `expected_trajectory`, and `context` are consumed by specific scorers — pass
> them as keyword arguments to `evaluate()` (applied to every case) or call the
> scorer's `.score(...)` directly per case. See the
> [trajectory](trajectory-scoring.md) and [session](session-scoring.md) docs.

## Built-in Scorers

### ExactMatch

Passes if the agent's output exactly matches the expected output (whitespace trimmed).

```python
from fastaiagent.eval.builtins import ExactMatch

scorer = ExactMatch()
result = scorer.score(input="q", output="Hello", expected="Hello")
# score=1.0, passed=True
```

### Contains

Passes if the expected text appears anywhere in the output (case-insensitive).

```python
from fastaiagent.eval.builtins import Contains

scorer = Contains()
result = scorer.score(input="q", output="The answer is 42", expected="42")
# score=1.0, passed=True
```

### JSONValid

Passes if the output is valid JSON.

```python
from fastaiagent.eval.builtins import JSONValid

scorer = JSONValid()
scorer.score(input="q", output='{"key": "value"}')  # passed=True
scorer.score(input="q", output="not json")            # passed=False
```

### RegexMatch

Passes if the output matches a regex pattern.

```python
from fastaiagent.eval.builtins import RegexMatch

scorer = RegexMatch(pattern=r"\d{3}-\d{4}")
scorer.score(input="q", output="Call 555-1234")  # passed=True
```

### LengthBetween

Passes if the output length is within a range.

```python
from fastaiagent.eval.builtins import LengthBetween

scorer = LengthBetween(min_len=10, max_len=500)
scorer.score(input="q", output="Short")           # passed=False (5 chars)
scorer.score(input="q", output="A longer answer")  # passed=True
```

### Latency

Passes if execution latency is under a threshold.

`evaluate()` supplies `latency_ms` automatically — from the `AgentResult` when
your callable returns one, and measured around the call otherwise, so the gate
works even for a callable that returns a bare string. You can still pass it by
hand when calling the scorer directly.

```python
from fastaiagent.eval.builtins import Latency

# Inside evaluate(): latency is filled in for you.
evaluate(agent_fn=agent.run, dataset=ds, scorers=[Latency(max_ms=2000)])

# Standalone:
scorer = Latency(max_ms=2000)
scorer.score(input="q", output="answer", latency_ms=1500)  # passed=True
scorer.score(input="q", output="answer", latency_ms=3000)  # passed=False
```

### CostUnder

Passes if the run's USD spend is under a threshold.

`evaluate()` supplies `cost` and `cost_known` automatically from the
`AgentResult`. A value you pass yourself still wins.

```python
from fastaiagent.eval.builtins import CostUnder

evaluate(agent_fn=agent.run, dataset=ds, scorers=[CostUnder(max_usd=0.05)])

scorer = CostUnder(max_usd=0.05)
scorer.score(input="q", output="answer", cost=0.03)  # passed=True
```

!!! warning "Unknown cost fails the gate — it is not treated as free"
    When the model has no rate in the pricing table (a private fine-tune, a
    bedrock or azure deployment id), the run cannot be priced and `cost_known`
    is `False`. `CostUnder` then **fails** and says so, rather than certifying a
    run it never priced. Give the model a rate with `set_rate_overrides()` or
    the `pricing` block of `models.json` to gate on it properly.

    **Self-hosted providers are the exception**, and they are a known zero
    rather than an unknown: `ollama`, `lmstudio` and `vllm` run on hardware you
    already pay for, so a run costs nothing *to the provider*, `cost_known` is
    `True`, and a budget gate passes it. Treating local inference as unpriced
    would refuse every laptop running ollama — a false positive, not a safety
    property.

!!! danger "Before 1.67.0 both of these could not fail"
    `evaluate()` passed neither `cost` nor `latency_ms`, so both scorers read
    `0` out of an empty `**kwargs` and passed every case — `CostUnder` reported
    "Cost: $0.0000" and `Latency` "Latency: 0ms" regardless of the run. If you
    have a `CostUnder` or `Latency` gate in CI that has always been green,
    expect it to start giving a real verdict.

### Using by Name

Pass built-in scorer names as strings to `evaluate()`:

```python
results = evaluate(
    agent_fn=my_agent,
    dataset=dataset,
    scorers=["exact_match", "contains"],  # Resolved automatically
)
```

Available names (resolved by `evaluate()` automatically):

- **Core:** `exact_match`, `contains`, `json_valid`, `regex_match`, `length_between`, `latency`, `cost_under`
- **RAG:** `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- **Safety:** `toxicity`, `bias`, `pii_leakage`, `prompt_injection`, `moderation`
- **Agent metrics:** `task_completion`, `hallucination`, `reflection_quality`
- **Similarity:** `semantic_similarity`, `bleu`, `rouge`, `levenshtein`

> **Trajectory and session scorers are not string-resolvable.** They need
> per-call trajectory/turn data that `evaluate()`'s dataset loop does not
> forward automatically, so instantiate them directly (e.g. `ToolUsageAccuracy()`)
> and call `.score(...)`. See [Trajectory Scoring](trajectory-scoring.md) and
> [Session Scoring](session-scoring.md).

## Custom Code Scorers

### The @Scorer.code Decorator

```python
from fastaiagent.eval import Scorer, ScorerResult

@Scorer.code("has_greeting")
def has_greeting(input, output, expected=None):
    """Check if the output starts with a greeting."""
    greetings = ["hello", "hi", "hey", "greetings"]
    starts_with_greeting = any(output.lower().startswith(g) for g in greetings)
    return ScorerResult(
        score=1.0 if starts_with_greeting else 0.0,
        passed=starts_with_greeting,
        reason=f"Starts with greeting: {starts_with_greeting}",
    )

# Use in evaluation
results = evaluate(agent_fn=my_agent, dataset=dataset, scorers=[has_greeting])
```

### Return Types

Custom scorers can return different types:

```python
# Return ScorerResult (full control)
@Scorer.code("detailed")
def detailed(input, output, expected=None):
    return ScorerResult(score=0.8, passed=True, reason="Almost perfect")

# Return bool (simple pass/fail)
@Scorer.code("simple")
def simple(input, output, expected=None):
    return len(output) > 10

# Return float (score, passed if >= 0.5)
@Scorer.code("scored")
def scored(input, output, expected=None):
    return len(output) / 100  # Score based on length
```

## EvalResults

### Summary

```python
results = evaluate(agent_fn=my_fn, dataset=data, scorers=[ExactMatch(), Contains()])

print(results.summary())
# Evaluation Results
# ==================================================
# exact_match: avg=0.80 pass_rate=80% (10 cases)
# contains: avg=0.95 pass_rate=95% (10 cases)
```

### Accessing Scores

```python
for scorer_name, scores in results.scores.items():
    for s in scores:
        print(f"{scorer_name}: score={s.score}, passed={s.passed}, reason={s.reason}")
```

### Export

```python
results.export("eval_results.json")
```

Produces:
```json
{
  "exact_match": [
    {"score": 1.0, "passed": true, "reason": null},
    {"score": 0.0, "passed": false, "reason": null}
  ],
  "contains": [...]
}
```

### Compare

Compare two evaluation runs:

```python
results_v1 = evaluate(agent_fn=agent_v1, dataset=data, scorers=scorers)
results_v2 = evaluate(agent_fn=agent_v2, dataset=data, scorers=scorers)

print(results_v1.compare(results_v2))
# Comparison
# ==================================================
# exact_match: 0.80 → 0.90 (+0.10)
# contains: 0.95 → 0.98 (+0.03)
```

## Combining Multiple Scorers

```python
results = evaluate(
    agent_fn=my_agent.run,
    dataset=Dataset.from_jsonl("test_cases.jsonl"),
    scorers=[
        ExactMatch(),                         # Exact string match
        Contains(),                           # Substring check
        LengthBetween(min_len=20, max_len=500),  # Length constraint
        has_greeting,                         # Custom code scorer
        LLMJudge(criteria="helpfulness"),     # LLM-as-judge
    ],
)
```

## ScorerResult

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Numeric score (0.0-1.0) |
| `passed` | `bool` | Whether the test case passed |
| `reason` | `str \| None` | Explanation of the score |

## CLI Commands

```bash
# Run a dataset against an agent and gate on the result
fastaiagent eval run \
  --agent app/agents.py:support_agent \
  --dataset cases.jsonl \
  --scorers exact_match,faithfulness \
  --fail-under "overall.pass_rate=0.9" \
  --max-error-rate 0.1 \
  --json report.json

# Compare two persisted runs (baseline first)
fastaiagent eval compare main pr --tolerance 0.02

# Turn production failures into eval cases
fastaiagent eval curate --filter failed -o cases/regressions.jsonl
```

Exit codes: `0` gate passed · `1` quality gate failed · `3` run invalid
(infra error rate exceeded). See **[Agent CI](agent-ci.md)** for the
pytest-native gate, baselines, and the GitHub Actions recipe.

The equivalent Python API:

```python
from fastaiagent.eval import evaluate

results = evaluate(
    agent_fn=my_agent.run,
    dataset="test_cases.jsonl",
    scorers=["exact_match", "contains"],
)
print(results.summary())
```

## Error Handling

```python
from fastaiagent._internal.errors import EvalError

try:
    results = evaluate(
        agent_fn=my_agent,
        dataset=data,
        scorers=["nonexistent_scorer"],
    )
except ValueError as e:
    print(f"Unknown scorer: {e}")
```

---

## Platform Integration

When connected to the FastAIAgent Platform, you can pull shared datasets, publish results, and pull scorer configs:

```python
import fastaiagent as fa

fa.connect(api_key="fa-...", project="my-project")

# Pull dataset from platform
dataset = Dataset.from_platform("golden-test-set")

# Run eval locally — scoring happens on your machine
results = evaluate(agent, dataset=dataset)

# Publish results to platform dashboard
results.publish(run_name="v2.1-release-candidate")

# Push a local dataset to platform for team sharing
local_dataset = Dataset.from_jsonl("my_tests.jsonl")
local_dataset.publish("regression-tests")

# Pull scorer config from platform (e.g., LLM judge)
scorer = Scorer.from_platform("correctness-judge")
results = evaluate(agent, dataset=dataset, scorers=[scorer])
results.publish()
```

All eval execution runs locally (your scorers, your LLM costs). The platform provides dataset sharing, result dashboards, and score trend tracking.

---

## Internals

For contributors who need to understand the evaluation loop, scorer resolution pipeline, how built-in scorers are implemented (pure code vs LLM-based vs embedding-based), or how to add a new scorer, see [Evaluation System Internals](../internals/evaluation-system.md).

## Next Steps

- [LLM Judge](llm-judge.md) — Use an LLM to evaluate output quality
- [RAG Metrics](rag-metrics.md) — Faithfulness, relevancy, and context evaluation
- [Safety Metrics](safety-metrics.md) — Toxicity, bias, and PII detection
- [Similarity Metrics](similarity-metrics.md) — Embedding-based and classical NLP metrics
- [Trajectory Scoring](trajectory-scoring.md) — Evaluate the path an agent took
- [Session Scoring](session-scoring.md) — Evaluate multi-turn conversations
- [Trace Curation](curation.md) — Build datasets from captured agent traces
- [Agents](../agents/index.md) — Build agents to evaluate
