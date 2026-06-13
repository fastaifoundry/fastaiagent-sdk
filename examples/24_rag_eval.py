"""Example 24: Evaluate a REAL RAG agent — faithfulness, relevancy, context metrics.

Demonstrates (real Agent + real LLM scorers — needs OPENAI_API_KEY):
- Faithfulness    : is the response grounded in the context?
- AnswerRelevancy : is the response relevant to the question?
- ContextPrecision: are relevant context chunks ranked higher?
- ContextRecall   : does the context cover the expected answer's claims?

The agent is grounded on a small knowledge base via its system prompt and runs
real completions; the scorers then make real LLM judgements about its answers.

Run:
    zsh -lc 'python examples/24_rag_eval.py'
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import Dataset, evaluate
from fastaiagent.eval.rag import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

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
CONTEXT = KNOWLEDGE["python"] + "\n\n" + KNOWLEDGE["rust"]


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    # A real agent grounded on the knowledge base.
    agent = Agent(
        name="rag-bot",
        system_prompt=(
            "Answer the user's question using ONLY the context below. Be concise.\n\n"
            f"CONTEXT:\n{CONTEXT}"
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )

    dataset = Dataset.from_list(
        [
            {
                "input": "What is Python and who created it?",
                "expected": "Python is a programming language created by Guido van Rossum in 1991.",
            },
            {
                "input": "Tell me about the Rust programming language.",
                "expected": "Rust is a systems programming language created by Graydon Hoare.",
            },
        ]
    )

    # Response-quality metrics through evaluate(). `agent.run` is passed directly;
    # evaluate() unwraps AgentResult.output. Context is a global kwarg here.
    print("== Response-quality metrics (evaluate) ==")
    results = evaluate(
        agent_fn=agent.run,
        dataset=dataset,
        scorers=[Faithfulness(), AnswerRelevancy()],
        context=CONTEXT,
        persist=False,
    )
    print(results.summary())

    # Retrieval-quality metrics, scored directly with ranked chunks:
    #   ContextPrecision — are relevant chunks ranked above irrelevant ones?
    #   ContextRecall    — do the chunks cover the expected answer's claims?
    print("\n== Retrieval-quality metrics (direct) ==")
    chunks = [
        KNOWLEDGE["python"],  # relevant — rank 1
        "Unrelated text about cooking recipes.",  # irrelevant — rank 2
        "Python also has a large third-party package ecosystem.",  # relevant — rank 3
    ]
    question = "What is Python and who created it?"

    precision = ContextPrecision().score(
        input=question,
        output="Python is a programming language created by Guido van Rossum.",
        contexts=chunks,
    )
    print(f"  context_precision: score={precision.score:.2f} — {precision.reason}")

    recall = ContextRecall().score(
        input=question,
        output="",  # recall scores the context against `expected`, not the output
        expected="Python is a programming language created by Guido van Rossum in 1991.",
        contexts=chunks,
    )
    print(f"  context_recall:    score={recall.score:.2f} — {recall.reason}")


if __name__ == "__main__":
    main()
