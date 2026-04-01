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
evaluate(agent_fn=lambda x: x.upper(), ...)

# Agent.run
evaluate(agent_fn=my_agent.run, ...)

# Custom wrapper
def run_pipeline(input_text):
    result = chain.execute({"message": input_text})
    return result.output
evaluate(agent_fn=run_pipeline, ...)
```

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

Passes if execution latency is under a threshold. Pass `latency_ms` as a kwarg.

```python
from fastaiagent.eval.builtins import Latency

scorer = Latency(max_ms=2000)
scorer.score(input="q", output="answer", latency_ms=1500)  # passed=True
scorer.score(input="q", output="answer", latency_ms=3000)  # passed=False
```

### CostUnder

Passes if cost is under a threshold. Pass `cost` as a kwarg.

```python
from fastaiagent.eval.builtins import CostUnder

scorer = CostUnder(max_usd=0.05)
scorer.score(input="q", output="answer", cost=0.03)  # passed=True
```

### Using by Name

Pass built-in scorer names as strings to `evaluate()`:

```python
results = evaluate(
    agent_fn=my_agent,
    dataset=dataset,
    scorers=["exact_match", "contains"],  # Resolved automatically
)
```

Available names: `exact_match`, `contains`, `json_valid`

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
# Run evaluation
fastaiagent eval run \
    --dataset test_cases.jsonl \
    --agent myapp:agent \
    --scorers exact_match,contains

# Compare two runs
fastaiagent eval compare results_v1.json results_v2.json
```

> **Note:** The CLI provides basic eval invocation. For full features (custom scorers, LLM judge, trajectory scoring), use the Python API.

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

## Next Steps

- [LLM Judge](llm-judge.md) — Use an LLM to evaluate output quality
- [Trajectory Scoring](trajectory-scoring.md) — Evaluate the path an agent took
- [Session Scoring](session-scoring.md) — Evaluate multi-turn conversations
- [Agents](../agents/index.md) — Build agents to evaluate
