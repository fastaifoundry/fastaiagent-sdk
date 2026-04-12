# Evaluation System (Internals)

This document explains how the SDK's evaluation framework works end-to-end — from `evaluate()` entry to scorer execution to results aggregation. It covers the execution loop, the scorer resolution system, every built-in scorer's implementation approach, the LLM judge pattern, and the platform integration. Written for contributors who need to add new scorers, modify the evaluation pipeline, or debug scoring behavior.

For the user-facing evaluation guide, see [docs/evaluation/index.md](../evaluation/index.md).

---

## Architecture Overview

```
evaluate(agent_fn, dataset, scorers, concurrency=4)
    │
    ├── Dataset resolution (path/list/object → Dataset)
    │
    ├── Scorer resolution (string names → Scorer instances via BUILTIN_SCORERS)
    │
    └── Concurrent evaluation loop (asyncio.Semaphore)
            │
            for each item in dataset (up to `concurrency` in parallel):
            │
            ├── Call agent_fn(input) → output
            │
            ├── Score with each scorer(input, output, expected) → ScorerResult
            │
            └── Aggregate into EvalResults
                    │
                    ├── .summary() → formatted table
                    ├── .export(path) → JSON file
                    ├── .publish(run_name) → POST /public/v1/eval/runs
                    └── .compare(other) → delta table
```

---

## The `evaluate()` / `aevaluate()` Entry Point

**File:** `fastaiagent/eval/evaluate.py`

### Signature

```python
def evaluate(
    agent_fn: Callable[..., Any],     # Your agent — agent.run, a lambda, any callable
    dataset: Dataset | str | list[dict[str, Any]],  # .jsonl path, .csv path, list, or Dataset
    scorers: list[Scorer | str] | None = None,      # Scorer objects or string names
    concurrency: int = 4,             # Max parallel evaluations
    **kwargs: Any,                    # Forwarded to each scorer's score() call
) -> EvalResults:
```

`evaluate()` is a thin sync wrapper around `aevaluate()` via `run_sync()`.

### Step-by-Step Execution

**Step 1 — Dataset resolution (lines 46–58):**

```python
if isinstance(dataset, str):
    p = Path(dataset)
    if p.suffix == ".jsonl":
        ds = Dataset.from_jsonl(p)
    elif p.suffix == ".csv":
        ds = Dataset.from_csv(p)
elif isinstance(dataset, list):
    ds = Dataset.from_list(dataset)
elif isinstance(dataset, Dataset):
    ds = dataset
```

**Step 2 — Scorer resolution (lines 60–75):**

```python
resolved_scorers = []
for s in scorers:
    if isinstance(s, str):
        if s not in BUILTIN_SCORERS:
            raise EvalError(f"Unknown scorer '{s}'. Available: {list(BUILTIN_SCORERS.keys())}")
        resolved_scorers.append(BUILTIN_SCORERS[s]())  # Instantiate the class
    else:
        resolved_scorers.append(s)  # Already a Scorer instance
```

Default when no scorers provided: `["exact_match"]`.

String names are looked up in `BUILTIN_SCORERS` dict (defined in `fastaiagent/eval/builtins.py`). If the name isn't found, the error message includes the full list of available scorers.

**Step 3 — Concurrent evaluation loop (lines 77–112):**

```python
sem = asyncio.Semaphore(concurrency)
results = EvalResults()

async def eval_one(item):
    async with sem:
        input_text = item.get("input") or str(item)
        expected = item.get("expected_output") or item.get("expected")

        # Call the agent (handles both sync and async callables)
        if asyncio.iscoroutinefunction(agent_fn):
            output = await agent_fn(input_text)
        else:
            output = agent_fn(input_text)
        output = str(output)

        # Score with each scorer
        for scorer in resolved_scorers:
            if hasattr(scorer, "ascore"):
                sr = await scorer.ascore(input=input_text, output=output, expected=expected, **kwargs)
            else:
                sr = scorer.score(input=input_text, output=output, expected=expected, **kwargs)
            results.add(scorer.name, sr)

tasks = [eval_one(item) for item in ds]
await asyncio.gather(*tasks)
return results
```

**Key concurrency behavior:** the `Semaphore(concurrency)` limits how many `eval_one` coroutines run simultaneously. With `concurrency=4` (default) and a 100-item dataset, up to 4 agent calls run in parallel at any moment. This is important for rate-limited LLM providers.

---

## Dataset

**File:** `fastaiagent/eval/dataset.py`

### Internal Structure

```python
class Dataset:
    def __init__(self, items: list[dict[str, Any]] | None = None):
        self._items: list[dict[str, Any]] = items or []
```

Each item is a dict. The evaluation loop reads these keys:
- `"input"` — the prompt to send to the agent (required)
- `"expected_output"` or `"expected"` — the reference answer for scoring (optional, used by scorers that compare output to a ground truth)

Additional keys (like `"context"`, `"contexts"`) can be present and are forwarded to scorers via `**kwargs`.

### Factory Methods

| Method | Input | Behavior |
|--------|-------|----------|
| `from_jsonl(path)` | `.jsonl` file | One JSON object per line |
| `from_csv(path)` | `.csv` file | `csv.DictReader`, header row becomes keys |
| `from_list(items)` | `list[dict]` | Direct pass-through |
| `from_dict(data)` | `dict` | Extracts `data["items"]` |
| `from_platform(name)` | Platform slug | GET `/public/v1/eval/datasets/{name}` |

### Platform Operations

```python
# Publish to platform
dataset.publish("my-golden-set")
# → POST /public/v1/eval/datasets {"name": "my-golden-set", "items": [...]}

# Fetch from platform
dataset = Dataset.from_platform("my-golden-set")
# → GET /public/v1/eval/datasets/my-golden-set → {"items": [...]}
```

---

## Scorer System

**File:** `fastaiagent/eval/scorer.py`

### Base Interface

```python
class Scorer:
    name: str

    def score(self, input: str, output: str, expected: str | None = None, **kwargs) -> ScorerResult:
        raise NotImplementedError
```

Some scorers also implement `ascore()` for async execution (LLM-based scorers that call `LLMClient.acomplete()`).

### ScorerResult

```python
class ScorerResult(BaseModel):
    score: float = 0.0       # Typically 0.0–1.0
    passed: bool = False     # Whether the item passed the scorer's threshold
    reason: str | None = None  # Human-readable explanation
```

### The `@Scorer.code()` Decorator

Creates a custom scorer from a Python function:

```python
@Scorer.code("my_scorer")
def my_scorer(input: str, output: str, expected: str | None = None) -> ScorerResult:
    if expected and expected.lower() in output.lower():
        return ScorerResult(score=1.0, passed=True, reason="Match")
    return ScorerResult(score=0.0, passed=False, reason="No match")
```

The decorated function can return:
- `ScorerResult` — used directly
- `bool` — converted to `ScorerResult(score=1.0 if True else 0.0, passed=bool_value)`
- `int` or `float` — converted to `ScorerResult(score=float(value), passed=value > 0.5)`
- `str` — treated as an error reason

The decorator wraps the function in a `CodeScorer` instance that handles the type coercion.

### String Name Resolution

When you pass `scorers=["exact_match", "contains"]` to `evaluate()`, each string is looked up in the `BUILTIN_SCORERS` registry:

```python
# fastaiagent/eval/builtins.py (lines 118-156)
BUILTIN_SCORERS: dict[str, type[Scorer]] = {
    "exact_match": ExactMatch,
    "contains": Contains,
    "json_valid": JSONValid,
    "regex_match": RegexMatch,
    "length_between": LengthBetween,
    "latency": Latency,
    "cost_under": CostUnder,
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecision,
    "context_recall": ContextRecall,
    "toxicity": Toxicity,
    "bias": Bias,
    "pii_leakage": PIILeakage,
    "semantic_similarity": SemanticSimilarity,
    "bleu": BLEUScore,
    "rouge": ROUGEScore,
    "levenshtein": LevenshteinDistance,
}
```

The string is used to look up the class, which is then instantiated with default parameters. If you need non-default parameters (e.g., `LengthBetween(min_len=10, max_len=500)`), pass the instantiated scorer object instead of the string.

---

## Built-in Scorers — Complete Reference

### Core Scorers (Pure Code — No Dependencies)

**File:** `fastaiagent/eval/builtins.py`

| Scorer | String Name | What It Measures | Algorithm |
|--------|-------------|-----------------|-----------|
| `ExactMatch` | `"exact_match"` | Exact string equality | `output.strip() == expected.strip()` |
| `Contains` | `"contains"` | Substring presence | `expected.lower() in output.lower()` |
| `JSONValid` | `"json_valid"` | Valid JSON output | `json.loads(output)` succeeds |
| `RegexMatch` | `"regex_match"` | Pattern match | `re.search(pattern, output)` — requires `pattern` constructor arg |
| `LengthBetween` | `"length_between"` | Output length bounds | `min_len <= len(output) <= max_len` — requires constructor args |
| `Latency` | `"latency"` | Response time | Reads `latency_ms` from kwargs, checks `<= max_ms` |
| `CostUnder` | `"cost_under"` | API cost budget | Reads `cost` from kwargs, checks `<= max_usd` |

### RAG Scorers (LLM-Based)

**File:** `fastaiagent/eval/rag.py`

These evaluate retrieval-augmented generation quality. All use LLM-as-judge internally.

| Scorer | String Name | Inputs | Algorithm | Default Threshold |
|--------|-------------|--------|-----------|-------------------|
| `Faithfulness` | `"faithfulness"` | output + context | 2-step: extract claims from output, verify each against context | 0.7 |
| `AnswerRelevancy` | `"answer_relevancy"` | input (question) + output | Single LLM call rates relevance 0–1 | 0.7 |
| `ContextPrecision` | `"context_precision"` | input + contexts (ordered list) | Average Precision: LLM judges each chunk's relevance, computes AP | 0.5 |
| `ContextRecall` | `"context_recall"` | input + expected + context | 2-step: extract claims from expected, check each against context | 0.7 |

**How they access context:** Scorers read `context` (single string) or `contexts` (list of strings) from `**kwargs`. The evaluation loop forwards any extra keys from the dataset item, so if your dataset has `{"input": "...", "expected": "...", "context": "..."}`, the context arrives at the scorer automatically.

**Faithfulness detailed flow:**
```
output → [LLM: extract claims] → ["claim 1", "claim 2", "claim 3"]
                                        │
    for each claim:                     ▼
        [LLM: is this claim supported by the context?] → yes/no
                                        │
    score = supported_claims / total_claims
```

### Safety Scorers (Mixed)

**File:** `fastaiagent/eval/safety.py`

| Scorer | String Name | Algorithm | LLM Required? |
|--------|-------------|-----------|---------------|
| `PIILeakage` | `"pii_leakage"` | Regex patterns: email, phone, SSN, credit card | No |
| `Toxicity` | `"toxicity"` | LLM judges for hate speech, threats, profanity, sexual content, self-harm, discrimination | Yes |
| `Bias` | `"bias"` | LLM judges for gender, racial, age, political, religious, socioeconomic bias | Yes |

**PIILeakage patterns:**
```python
_PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",                    # SSN
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",            # Phone
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
]
```
Score = 1.0 if no matches (safe), 0.0 if any found (leaking PII).

### Similarity Scorers (Mixed)

**File:** `fastaiagent/eval/similarity.py`

| Scorer | String Name | Algorithm | Dependencies |
|--------|-------------|-----------|-------------|
| `SemanticSimilarity` | `"semantic_similarity"` | Cosine similarity between embedding vectors | Embedder (fastembed/OpenAI) |
| `BLEUScore` | `"bleu"` | Brevity penalty × geometric mean of n-gram precisions (n=1..4) | None (pure Python) |
| `ROUGEScore` | `"rouge"` | Unigram F1 (rouge-1) or LCS-based F1 (rouge-l) | None (pure Python) |
| `LevenshteinDistance` | `"levenshtein"` | Normalized edit distance: `1.0 - (dist / max_len)` | None (pure Python) |

**SemanticSimilarity flow:**
```
output → embedder.embed([output]) → vector_a
expected → embedder.embed([expected]) → vector_b
score = cosine_similarity(vector_a, vector_b)
```

---

## LLM Judge Pattern

**File:** `fastaiagent/eval/llm_judge.py`

A generic LLM-based scorer that delegates evaluation to an LLM. Used internally by RAG scorers and safety scorers, and available directly for custom criteria.

### Constructor

```python
LLMJudge(
    criteria: str,                     # What to evaluate (e.g., "correctness", "helpfulness")
    prompt_template: str | None = None, # Custom template with {input}/{output}/{expected} placeholders
    llm: Any = None,                   # Custom LLMClient (defaults to LLMClient())
    scale: str = "0-1",               # "binary", "0-1", or "1-5"
)
```

### How It Works

1. Build the evaluation prompt by substituting `{input}`, `{output}`, `{expected}` into the template
2. Call the LLM with:
   - System message: `"You are an evaluation judge. Evaluate the following and respond in JSON format with 'score' (0-1) and 'reasoning'."`
   - User message: the filled template
3. Parse the LLM's JSON response: `{"score": 0.85, "reasoning": "The output is accurate but..."}`
4. Return `ScorerResult(score=parsed_score, passed=score >= threshold, reason=reasoning)`

### Default Prompt Template

```
Evaluate the following based on the criterion: {criteria}

Input: {input}
Output: {output}
Expected: {expected}

Provide a score from 0 to 1 and your reasoning.
Respond in JSON: {{"score": <float>, "reasoning": "<string>"}}
```

### Usage

```python
from fastaiagent.eval.llm_judge import LLMJudge

judge = LLMJudge(criteria="helpfulness", scale="0-1")
result = judge.score(input="How do I reset?", output="Click Settings > Reset", expected="Go to Settings")
print(result.score, result.reason)
```

RAG scorers use `LLMJudge` internally — `Faithfulness`, `AnswerRelevancy`, `ContextPrecision`, and `ContextRecall` each construct their own LLM calls with specialized prompts for claim extraction, relevance rating, and context verification.

---

## EvalResults

**File:** `fastaiagent/eval/results.py`

### Structure

```python
class EvalResults:
    scores: dict[str, list[ScorerResult]]
    # Key = scorer name, Value = one ScorerResult per dataset item
```

### Methods

**`summary()`** — Formatted results table:
```
Evaluation Results
==================================================
contains_keyword: avg=0.75 pass_rate=75% (4 cases)
exact_match: avg=0.50 pass_rate=50% (4 cases)
```

Computed as:
- `avg_score = sum(r.score for r in results) / len(results)`
- `pass_rate = sum(1 for r in results if r.passed) / len(results)`

**`export(path, format="json")`** — Writes results to a JSON file:
```json
{
    "contains_keyword": [
        {"score": 1.0, "passed": true, "reason": "contains 'shipped'"},
        {"score": 0.0, "passed": false, "reason": "missing 'processing'"}
    ]
}
```

**`compare(other)`** — Diffs two eval runs:
```
Comparison
==================================================
contains_keyword: 0.75 → 0.85 (+0.10)
exact_match: 0.50 → 0.50 (0.00)
```

Shows delta per scorer. Useful for A/B testing prompt changes or model swaps.

**`publish(run_name)`** — Pushes to platform:
```python
# POST /public/v1/eval/runs
{
    "run_name": "v1-golden",
    "scores": {
        "contains_keyword": [
            {"score": 1.0, "passed": true, "reason": "..."},
            ...
        ]
    }
}
```

---

## Adding a New Built-in Scorer

For contributors adding new scorers, follow this pattern:

**Step 1:** Create the scorer class in the appropriate file:
- Pure code → `fastaiagent/eval/builtins.py`
- LLM-based → `fastaiagent/eval/rag.py`, `safety.py`, or a new file
- Embedding-based → `fastaiagent/eval/similarity.py`

```python
class MyScorer(Scorer):
    name = "my_scorer"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def score(self, input: str, output: str, expected: str | None = None, **kwargs) -> ScorerResult:
        # Your scoring logic
        my_score = compute_something(output, expected)
        return ScorerResult(
            score=my_score,
            passed=my_score >= self.threshold,
            reason=f"Score: {my_score:.2f}",
        )
```

**Step 2:** Register in `BUILTIN_SCORERS` at the bottom of `builtins.py`:

```python
BUILTIN_SCORERS["my_scorer"] = MyScorer
```

**Step 3:** Add a test in the e2e gate or the unit suite that exercises the new scorer against a real dataset.

**Conventions:**
- Scorer names are `snake_case` strings
- Scores are typically 0.0–1.0 (higher is better)
- `passed` defaults to `score >= threshold` where threshold is configurable
- `reason` should be a one-line human-readable explanation
- For LLM-based scorers, use `ascore()` for the async implementation and `score()` as a sync wrapper via `run_sync(self.ascore(...))`

---

## Files Reference

| File | What it does |
|------|-------------|
| `fastaiagent/eval/evaluate.py` | `evaluate()` / `aevaluate()` entry points, dataset/scorer resolution, concurrent loop |
| `fastaiagent/eval/dataset.py` | `Dataset` class — from_jsonl/csv/list/dict/platform, publish |
| `fastaiagent/eval/scorer.py` | `Scorer` base class, `ScorerResult` model, `@Scorer.code()` decorator, `CodeScorer` |
| `fastaiagent/eval/builtins.py` | 7 core scorers (ExactMatch, Contains, JSONValid, RegexMatch, LengthBetween, Latency, CostUnder) + `BUILTIN_SCORERS` registry |
| `fastaiagent/eval/llm_judge.py` | `LLMJudge` — generic LLM-as-judge scorer with customizable criteria and prompts |
| `fastaiagent/eval/rag.py` | 4 RAG scorers (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall) |
| `fastaiagent/eval/safety.py` | 3 safety scorers (PIILeakage, Toxicity, Bias) |
| `fastaiagent/eval/similarity.py` | 4 similarity scorers (SemanticSimilarity, BLEUScore, ROUGEScore, LevenshteinDistance) |
| `fastaiagent/eval/results.py` | `EvalResults` — summary, export, publish, compare |
