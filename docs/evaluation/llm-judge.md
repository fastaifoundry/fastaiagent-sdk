# LLM Judge

Use an LLM to evaluate output quality. The judge LLM scores agent responses based on criteria you define.

## Basic Usage

```python
from fastaiagent.eval import LLMJudge, evaluate
from fastaiagent import LLMClient

judge = LLMJudge(
    criteria="correctness",
    llm=LLMClient(provider="openai", model="gpt-4.1"),
)

# Use in evaluation
results = evaluate(
    agent_fn=my_agent,
    dataset=dataset,
    scorers=[judge],
)
```

## Custom Judge Prompt

Provide a custom prompt template to control exactly how the judge evaluates:

```python
judge = LLMJudge(
    criteria="helpfulness",
    prompt_template=(
        "Rate the following response for helpfulness.\n\n"
        "User question: {input}\n"
        "Expected answer: {expected}\n"
        "Actual response: {output}\n\n"
        'Respond with JSON: {{"score": <0.0-1.0>, "reasoning": "<explanation>"}}'
    ),
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-6"),
)
```

The judge LLM must respond with JSON containing `score` and `reasoning` fields.

## G-Eval (evaluation steps + rubric)

For richer, more reliable judging, pass `evaluation_steps` and/or a score-band `rubric`. This turns the judge into a **G-Eval**: it reasons step-by-step through your evaluation steps, scores against the rubric, and normalizes the result to 0–1. The plain `criteria`-only judge above is unchanged — G-Eval activates only when you provide steps or a rubric (or use the `GEval` class).

```python
from fastaiagent.eval import GEval

judge = GEval(
    name="correctness",
    criteria="Is the answer factually correct and complete?",
    evaluation_steps=[
        "Identify the factual claim the answer makes.",
        "Compare it against the expected answer.",
        "Penalize fabricated, missing, or contradicted facts.",
    ],
    rubric=[
        (1, "Mostly incorrect"),
        (3, "Partially correct"),
        (5, "Fully correct"),
    ],
    scale="1-5",
    threshold=0.6,   # on the normalized 0–1 score
)

result = judge.score(input="Capital of France?", output="Paris", expected="Paris")
print(result.score, result.passed, result.reason)
```

`GEval` is a thin, DeepEval-familiar wrapper over `LLMJudge`'s G-Eval mode — these are equivalent:

```python
from fastaiagent.eval import GEval, LLMJudge

GEval(name="x", criteria="...", evaluation_steps=[...], rubric=[...])
LLMJudge(criteria="...", evaluation_steps=[...], rubric=[...], scale="1-5", name="x")
```

Each instance can carry its own `name`, so several judges don't collide in the results (e.g. `GEval(name="correctness")` and `GEval(name="tone")`).

### Auto-generated steps (Auto-CoT)

Give `GEval` a `criteria` but no `evaluation_steps` and it generates them from the criteria on first use (one extra LLM call, cached on the instance):

```python
judge = GEval(name="helpfulness", criteria="Does the response directly answer the question?")
judge.score(input=question, output=answer)
print(judge.evaluation_steps)   # derived steps, cached
```

The rubric is a list of `(score_value, description)` anchors on your `scale`; the judge interpolates between them and the final score is normalized to 0–1, with `passed = score >= threshold`.

See `examples/81_g_eval.py` for a runnable end-to-end script.

## Scale Types

`scale` sets the range the judge scores on; the raw score is then normalized to 0–1. It applies on the **G-Eval path** (with `evaluation_steps`/`rubric`, or `GEval`); the legacy `criteria`-only judge always scores 0–1.

```python
GEval(name="q", criteria="quality", scale="binary")   # 0 or 1
GEval(name="q", criteria="quality", scale="0-1")      # 0.0 to 1.0
GEval(name="q", criteria="quality", scale="1-5")      # 1 to 5, normalized to 0–1
```

## Combining with Other Scorers

LLM judges work alongside built-in and custom scorers:

```python
from fastaiagent.eval import evaluate
from fastaiagent.eval.builtins import ExactMatch, LengthBetween

results = evaluate(
    agent_fn=my_agent.run,
    dataset=dataset,
    scorers=[
        ExactMatch(),
        LengthBetween(min_len=20, max_len=500),
        LLMJudge(criteria="helpfulness"),
        LLMJudge(criteria="correctness"),
    ],
)
```

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [Trajectory Scoring](trajectory-scoring.md) — Evaluate the path an agent took
- [Session Scoring](session-scoring.md) — Evaluate multi-turn conversations
