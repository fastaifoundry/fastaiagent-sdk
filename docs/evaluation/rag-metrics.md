# RAG Metrics

Evaluate retrieval-augmented generation pipelines. These scorers measure whether your agent's responses are grounded in the retrieved context, relevant to the query, and whether the retrieval itself is effective.

All RAG scorers use LLM-as-judge under the hood. Pass context via `context="..."` or `contexts=["chunk1", "chunk2"]` as keyword arguments.

## Faithfulness

Measures factual consistency of the response with the retrieved context. Breaks the response into individual claims and verifies each against the context.

```python
from fastaiagent.eval import Faithfulness

scorer = Faithfulness()
result = scorer.score(
    input="What is Python?",
    output="Python is a programming language created by Guido van Rossum.",
    context="Python is a high-level programming language created by Guido van Rossum in 1991.",
)
# score = supported_claims / total_claims
# score ≈ 1.0, passed=True
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient \| None` | `None` | LLM client (defaults to OpenAI gpt-4o-mini) |
| `threshold` | `float` | `0.7` | Minimum score to pass |

**Kwargs:** `context` (str) or `contexts` (list of str)

## AnswerRelevancy

Measures how relevant the response is to the user's query. Does not require `expected` or `context` — evaluates the output purely against the input question.

```python
from fastaiagent.eval import AnswerRelevancy

scorer = AnswerRelevancy()
result = scorer.score(
    input="What is the capital of France?",
    output="Paris is the capital of France, located in northern France.",
)
# score ≈ 1.0, passed=True
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient \| None` | `None` | LLM client |
| `threshold` | `float` | `0.7` | Minimum score to pass |

## ContextPrecision

Measures whether relevant context chunks are ranked higher in the retrieval results. Uses Average Precision — rewards having relevant documents appear earlier in the list.

```python
from fastaiagent.eval import ContextPrecision

scorer = ContextPrecision()
result = scorer.score(
    input="What is the capital of France?",
    output="Paris",
    contexts=[
        "Paris is the capital and largest city of France.",  # relevant, rank 1
        "France is a country in Western Europe.",             # partially relevant, rank 2
        "The Eiffel Tower is in Paris.",                      # relevant, rank 3
    ],
)
# AP = high (relevant chunks ranked first)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient \| None` | `None` | LLM client |
| `threshold` | `float` | `0.5` | Minimum score to pass |

**Kwargs:** `contexts` (list of str, ordered by retrieval rank)

## ContextRecall

Measures what fraction of the expected answer's claims are present in the retrieved context. Helps identify when your retriever is missing important information.

```python
from fastaiagent.eval import ContextRecall

scorer = ContextRecall()
result = scorer.score(
    input="What is Python?",
    output="anything",
    expected="Python is a programming language created by Guido van Rossum in 1991.",
    context="Python is a programming language. Guido van Rossum created it in 1991.",
)
# score = claims_in_context / total_claims ≈ 1.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient \| None` | `None` | LLM client |
| `threshold` | `float` | `0.5` | Minimum score to pass |

**Kwargs:** `context` (str) or `contexts` (list of str). Also requires `expected`.

## Using in Evaluation

Combine RAG scorers in `evaluate()` — pass context through `**kwargs`:

```python
from fastaiagent.eval import evaluate
from fastaiagent.eval.rag import Faithfulness, AnswerRelevancy, ContextRecall

results = evaluate(
    agent_fn=my_rag_agent.run,
    dataset=[
        {
            "input": "What is Python?",
            "expected": "Python is a programming language.",
        },
    ],
    scorers=[Faithfulness(), AnswerRelevancy(), ContextRecall()],
    context="Python is a high-level programming language created by Guido van Rossum.",
)
print(results.summary())
```

Or use string names:

```python
results = evaluate(
    agent_fn=my_agent,
    dataset=dataset,
    scorers=["faithfulness", "answer_relevancy", "context_recall"],
    context="...",
)
```

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [Safety Metrics](safety-metrics.md) — Toxicity, bias, and PII detection
- [Similarity Metrics](similarity-metrics.md) — Embedding-based and classical NLP metrics
