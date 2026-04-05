"""Example 24: Evaluate a RAG pipeline with faithfulness, relevancy, and context metrics.

Demonstrates:
- Faithfulness: Is the response grounded in the context?
- AnswerRelevancy: Is the response relevant to the question?
- ContextPrecision: Are relevant context chunks ranked higher?
- ContextRecall: Does the context cover the expected answer?

Requires: OPENAI_API_KEY environment variable.
"""

import os
import sys

from fastaiagent.eval import Dataset, evaluate
from fastaiagent.eval.rag import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

# Simulated knowledge base
KNOWLEDGE = {
    "python": (
        "Python is a high-level, general-purpose programming language. "
        "It was created by Guido van Rossum and first released in 1991. "
        "Python emphasizes code readability with significant whitespace."
    ),
    "rust": (
        "Rust is a systems programming language focused on safety and performance. "
        "It was created by Graydon Hoare at Mozilla Research. "
        "Rust prevents data races at compile time."
    ),
}


def rag_agent(query: str) -> str:
    """Simple RAG agent that retrieves context and generates a response."""
    query_lower = query.lower()
    if "python" in query_lower:
        return "Python is a high-level programming language created by Guido van Rossum in 1991."
    if "rust" in query_lower:
        return "Rust is a systems programming language focused on safety, created by Graydon Hoare."
    return "I don't have information about that topic."


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    dataset = Dataset.from_list(
        [
            {
                "input": "What is Python and who created it?",
                "expected": "Python is a programming language created by Guido van Rossum in 1991.",
            },
            {
                "input": "Tell me about Rust programming language.",
                "expected": "Rust is a systems programming language created by Graydon Hoare.",
            },
        ]
    )

    # Run evaluation with RAG scorers
    results = evaluate(
        agent_fn=rag_agent,
        dataset=dataset,
        scorers=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextRecall(),
        ],
        # Context passed as kwargs to all scorers
        context=KNOWLEDGE["python"] + "\n\n" + KNOWLEDGE["rust"],
    )

    print(results.summary())

    # Context precision with ranked chunks
    from fastaiagent.eval.rag import ContextPrecision

    precision = ContextPrecision()
    result = precision.score(
        input="What is Python?",
        output="Python is a programming language.",
        contexts=[
            KNOWLEDGE["python"],  # Relevant — rank 1
            "Unrelated text about cooking recipes.",  # Irrelevant — rank 2
            "More info about Python's ecosystem.",  # Relevant — rank 3
        ],
    )
    print(f"\nContext Precision: score={result.score:.2f} — {result.reason}")
