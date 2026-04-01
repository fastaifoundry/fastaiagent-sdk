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
    llm=LLMClient(provider="anthropic", model="claude-sonnet-4-20250514"),
)
```

The judge LLM must respond with JSON containing `score` and `reasoning` fields.

## Scale Types

Control the scoring scale:

```python
LLMJudge(criteria="quality", scale="binary")   # 0 or 1
LLMJudge(criteria="quality", scale="0-1")      # 0.0 to 1.0 (default)
LLMJudge(criteria="quality", scale="1-5")      # 1 to 5, normalized
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
