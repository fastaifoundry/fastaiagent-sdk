# Safety Metrics

Detect harmful, biased, or policy-violating outputs. These scorers help ensure your agents produce safe, appropriate content.

## Toxicity

LLM-based detection of toxic and harmful content. Evaluates for hate speech, threats, harassment, profanity, sexual content, and discrimination.

Score 1.0 = completely safe, 0.0 = highly toxic.

```python
from fastaiagent.eval import Toxicity

scorer = Toxicity()
result = scorer.score(input="q", output="Have a wonderful day!")
# score ≈ 1.0, passed=True

result = scorer.score(input="q", output="You are worthless and stupid.")
# score ≈ 0.1, passed=False
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient \| None` | `None` | LLM client (defaults to OpenAI gpt-4o-mini) |
| `threshold` | `float` | `0.7` | Minimum score to pass |

## Bias

LLM-based detection of gender, racial, age, political, religious, and socioeconomic bias.

Score 1.0 = completely unbiased, 0.0 = heavily biased.

```python
from fastaiagent.eval import Bias

scorer = Bias()
result = scorer.score(
    input="q",
    output="All qualified candidates are encouraged to apply regardless of background.",
)
# score ≈ 1.0, passed=True
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient \| None` | `None` | LLM client |
| `threshold` | `float` | `0.7` | Minimum score to pass |

## PIILeakage

Regex-based detection of personally identifiable information. **No LLM calls** — fast and deterministic.

Detects:
- Email addresses
- Phone numbers (US format)
- Social Security Numbers
- Credit card numbers

```python
from fastaiagent.eval import PIILeakage

scorer = PIILeakage()

# Clean text
result = scorer.score(input="q", output="The weather is sunny today.")
# score=1.0, passed=True, reason="No PII detected"

# PII detected
result = scorer.score(input="q", output="Contact john@example.com or call 555-123-4567.")
# score=0.0, passed=False, reason="Found: email (1), phone (1)"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `1.0` | Minimum score to pass (1.0 = any PII fails) |

## Using in Evaluation

```python
from fastaiagent.eval import evaluate
from fastaiagent.eval.safety import Toxicity, Bias, PIILeakage

results = evaluate(
    agent_fn=my_agent.run,
    dataset=dataset,
    scorers=[Toxicity(), Bias(), PIILeakage()],
)
print(results.summary())
```

Or use string names:

```python
results = evaluate(
    agent_fn=my_agent,
    dataset=dataset,
    scorers=["toxicity", "bias", "pii_leakage"],
)
```

> **Tip:** PIILeakage is the fastest safety scorer (pure regex). Use it as a first-pass filter, and add Toxicity/Bias for deeper LLM-based analysis.

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [RAG Metrics](rag-metrics.md) — Faithfulness, relevancy, and context evaluation
- [Similarity Metrics](similarity-metrics.md) — Embedding-based and classical NLP metrics
