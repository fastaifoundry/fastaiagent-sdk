"""Example 26: Compare agent outputs with similarity and NLP metrics.

Demonstrates:
- BLEUScore: N-gram precision with brevity penalty.
- ROUGEScore: Recall-oriented n-gram evaluation.
- LevenshteinDistance: Character-level edit distance similarity.
- SemanticSimilarity: Embedding-based cosine similarity.

BLEU, ROUGE, and Levenshtein are pure Python — no API calls needed.
SemanticSimilarity requires an embedding provider (auto-detected).
"""

from fastaiagent.eval import Dataset, evaluate
from fastaiagent.eval.similarity import BLEUScore, LevenshteinDistance, ROUGEScore, SemanticSimilarity


def paraphrase_agent(input_text: str) -> str:
    """Agent that paraphrases input — used to test similarity scorers."""
    paraphrases = {
        "The cat sat on the mat.": "A cat was sitting on the mat.",
        "Paris is the capital of France.": "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.": "At 100 degrees Celsius, water reaches its boiling point.",
    }
    return paraphrases.get(input_text, input_text)


if __name__ == "__main__":
    dataset = Dataset.from_list(
        [
            {"input": "The cat sat on the mat.", "expected": "The cat sat on the mat."},
            {"input": "Paris is the capital of France.", "expected": "Paris is the capital of France."},
            {
                "input": "Water boils at 100 degrees Celsius.",
                "expected": "Water boils at 100 degrees Celsius.",
            },
        ]
    )

    # Pure Python metrics (no API calls)
    print("=== Pure Python Metrics (no API cost) ===\n")
    results = evaluate(
        agent_fn=paraphrase_agent,
        dataset=dataset,
        scorers=[
            BLEUScore(max_n=2),
            ROUGEScore(variant="rouge-1"),
            ROUGEScore(variant="rouge-l"),
            LevenshteinDistance(),
        ],
    )
    print(results.summary())

    # Detailed per-case results
    print()
    for scorer_name, scores in results.scores.items():
        for i, s in enumerate(scores):
            print(f"  {scorer_name} case {i + 1}: {s.score:.4f} — {s.reason}")

    # Semantic similarity (requires embeddings)
    print("\n=== Semantic Similarity (uses embeddings) ===\n")
    try:
        sem_results = evaluate(
            agent_fn=paraphrase_agent,
            dataset=dataset,
            scorers=[SemanticSimilarity()],
        )
        print(sem_results.summary())
    except Exception as e:
        print(f"Skipped SemanticSimilarity: {e}")
        print("Set OPENAI_API_KEY or install fastembed for local embeddings.")
