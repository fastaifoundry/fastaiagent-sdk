# Similarity & NLP Metrics

Compare agent outputs against expected answers using embeddings, n-gram overlap, and edit distance. These scorers are ideal for regression testing and measuring output quality without LLM judge costs.

## SemanticSimilarity

Cosine similarity between embeddings of the output and expected text. Uses the SDK's embedding infrastructure — auto-detects the best available embedder.

```python
from fastaiagent.eval import SemanticSimilarity

scorer = SemanticSimilarity(threshold=0.8)
result = scorer.score(
    input="q",
    output="Paris is the capital of France.",
    expected="The capital of France is Paris.",
)
# score ≈ 0.95, passed=True
```

Use a specific embedder:

```python
from fastaiagent.kb.embedding import OpenAIEmbedder

scorer = SemanticSimilarity(embedder=OpenAIEmbedder(model="text-embedding-3-small"))
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedder` | `Embedder \| None` | `None` | Embedding provider (auto-detects if None) |
| `threshold` | `float` | `0.7` | Minimum similarity to pass |

## BLEUScore

N-gram precision with brevity penalty — the standard machine translation metric. Pure Python, no API calls.

```python
from fastaiagent.eval import BLEUScore

scorer = BLEUScore(max_n=4, threshold=0.3)
result = scorer.score(
    input="q",
    output="the cat sat on the mat",
    expected="the cat is on the mat",
)
# Measures n-gram precision overlap
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_n` | `int` | `4` | Maximum n-gram order (BLEU-1 through BLEU-N) |
| `threshold` | `float` | `0.3` | Minimum score to pass |

## ROUGEScore

Recall-oriented n-gram evaluation. Supports unigram recall (`rouge-1`) and LCS-based F1 (`rouge-l`). Pure Python, no API calls.

```python
from fastaiagent.eval import ROUGEScore

# Unigram F1
scorer = ROUGEScore(variant="rouge-1")
result = scorer.score(
    input="q",
    output="the cat sat on the mat",
    expected="the cat is on the mat",
)

# LCS-based F1
scorer = ROUGEScore(variant="rouge-l")
result = scorer.score(
    input="q",
    output="the cat sat on the mat",
    expected="the cat is on the mat",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variant` | `str` | `"rouge-1"` | `"rouge-1"` (unigram F1) or `"rouge-l"` (LCS F1) |
| `threshold` | `float` | `0.3` | Minimum score to pass |

## LevenshteinDistance

Normalized edit distance similarity. Measures character-level differences between output and expected. Pure Python, no API calls.

```python
from fastaiagent.eval import LevenshteinDistance

scorer = LevenshteinDistance(threshold=0.7)
result = scorer.score(input="q", output="kitten", expected="sitting")
# edit_distance=3, similarity = 1 - 3/7 ≈ 0.571
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Minimum similarity to pass |

## Using in Evaluation

Combine similarity scorers for comprehensive text comparison:

```python
from fastaiagent.eval import evaluate
from fastaiagent.eval.similarity import SemanticSimilarity, BLEUScore, ROUGEScore

results = evaluate(
    agent_fn=my_agent.run,
    dataset=dataset,
    scorers=[
        SemanticSimilarity(),
        BLEUScore(max_n=2),
        ROUGEScore(variant="rouge-l"),
    ],
)
print(results.summary())
```

Or use string names:

```python
results = evaluate(
    agent_fn=my_agent,
    dataset=dataset,
    scorers=["semantic_similarity", "bleu", "rouge", "levenshtein"],
)
```

> **Cost comparison:** BLEU, ROUGE, and Levenshtein are pure Python (zero cost, instant). SemanticSimilarity requires an embedding API call but is still much cheaper than LLM-as-judge.

---

## Next Steps

- [Evaluation](index.md) — Core evaluation documentation
- [RAG Metrics](rag-metrics.md) — Faithfulness, relevancy, and context evaluation
- [Safety Metrics](safety-metrics.md) — Toxicity, bias, and PII detection
