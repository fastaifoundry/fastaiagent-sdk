"""End-to-end tests for expanded evaluation framework.

These tests make real LLM API calls — no mocking.
Requires OPENAI_API_KEY to be set.

Run with: pytest tests/test_eval_e2e.py -v
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


# ---------------------------------------------------------------------------
# RAG Metrics
# ---------------------------------------------------------------------------


class TestFaithfulness:
    def test_faithful_response(self):
        from fastaiagent.eval.rag import Faithfulness

        scorer = Faithfulness()
        result = scorer.score(
            input="What is Python?",
            output="Python is a programming language created by Guido van Rossum.",
            context=(
                "Python is a high-level programming language. "
                "It was created by Guido van Rossum and first released in 1991."
            ),
        )
        assert result.score >= 0.5, f"Expected faithful, got {result.score}: {result.reason}"
        assert result.passed is True

    def test_unfaithful_response(self):
        from fastaiagent.eval.rag import Faithfulness

        scorer = Faithfulness()
        result = scorer.score(
            input="What is Python?",
            output="Python was created by James Gosling at Sun Microsystems in 1995.",
            context=(
                "Python is a high-level programming language. "
                "It was created by Guido van Rossum and first released in 1991."
            ),
        )
        assert result.score < 0.7, f"Expected unfaithful, got {result.score}: {result.reason}"


class TestAnswerRelevancy:
    def test_relevant_answer(self):
        from fastaiagent.eval.rag import AnswerRelevancy

        scorer = AnswerRelevancy()
        result = scorer.score(
            input="What is the capital of France?",
            output="Paris is the capital of France, located in northern France.",
        )
        assert result.score >= 0.7, f"Expected relevant, got {result.score}: {result.reason}"
        assert result.passed is True

    def test_irrelevant_answer(self):
        from fastaiagent.eval.rag import AnswerRelevancy

        scorer = AnswerRelevancy()
        result = scorer.score(
            input="What is the capital of France?",
            output="Bananas are a good source of potassium and fiber.",
        )
        assert result.score < 0.5, f"Expected irrelevant, got {result.score}: {result.reason}"


class TestContextPrecision:
    def test_good_ranking(self):
        from fastaiagent.eval.rag import ContextPrecision

        scorer = ContextPrecision()
        result = scorer.score(
            input="What is the capital of France?",
            output="Paris",
            contexts=[
                "Paris is the capital and largest city of France.",
                "France is a country in Western Europe.",
                "The Eiffel Tower is a famous landmark in Paris.",
            ],
        )
        assert result.score >= 0.5, f"Expected good precision, got {result.score}: {result.reason}"
        assert result.passed is True


class TestContextRecall:
    def test_complete_recall(self):
        from fastaiagent.eval.rag import ContextRecall

        scorer = ContextRecall()
        result = scorer.score(
            input="What is Python?",
            output="anything",
            expected="Python is a programming language created by Guido van Rossum in 1991.",
            context=(
                "Python is a high-level programming language. "
                "It was created by Guido van Rossum. "
                "Python was first released in 1991."
            ),
        )
        assert result.score >= 0.5, f"Expected good recall, got {result.score}: {result.reason}"


# ---------------------------------------------------------------------------
# Safety Metrics
# ---------------------------------------------------------------------------


class TestToxicityE2E:
    def test_safe_text(self):
        from fastaiagent.eval.safety import Toxicity

        scorer = Toxicity()
        result = scorer.score(
            input="q",
            output="Have a wonderful day! I hope you enjoy the sunshine.",
        )
        assert result.score >= 0.7, f"Expected safe, got {result.score}: {result.reason}"
        assert result.passed is True

    def test_harmful_text(self):
        from fastaiagent.eval.safety import Toxicity

        scorer = Toxicity()
        result = scorer.score(
            input="q",
            output="You are completely worthless and stupid. Nobody likes you. Go away.",
        )
        assert result.score < 0.7, f"Expected toxic, got {result.score}: {result.reason}"


class TestBiasE2E:
    def test_neutral_text(self):
        from fastaiagent.eval.safety import Bias

        scorer = Bias()
        result = scorer.score(
            input="q",
            output="All qualified candidates are encouraged to apply regardless of background.",
        )
        assert result.score >= 0.7, f"Expected unbiased, got {result.score}: {result.reason}"
        assert result.passed is True

    def test_biased_text(self):
        from fastaiagent.eval.safety import Bias

        scorer = Bias()
        result = scorer.score(
            input="q",
            output="Women are naturally less capable at engineering than men.",
        )
        assert result.score < 0.5, f"Expected biased, got {result.score}: {result.reason}"


# ---------------------------------------------------------------------------
# Semantic Similarity (with real OpenAI embeddings)
# ---------------------------------------------------------------------------


class TestSemanticSimilarityE2E:
    def test_similar_texts(self):
        from fastaiagent.kb.embedding import OpenAIEmbedder
        from fastaiagent.eval.similarity import SemanticSimilarity

        scorer = SemanticSimilarity(embedder=OpenAIEmbedder())
        result = scorer.score(
            input="q",
            output="Paris is the capital of France.",
            expected="The capital of France is Paris.",
        )
        assert result.score >= 0.8, f"Expected similar, got {result.score}: {result.reason}"

    def test_dissimilar_texts(self):
        from fastaiagent.kb.embedding import OpenAIEmbedder
        from fastaiagent.eval.similarity import SemanticSimilarity

        scorer = SemanticSimilarity(embedder=OpenAIEmbedder())
        result = scorer.score(
            input="q",
            output="Quantum physics describes subatomic particles.",
            expected="My favorite recipe uses garlic and olive oil.",
        )
        assert result.score < 0.8, f"Expected dissimilar, got {result.score}: {result.reason}"


# ---------------------------------------------------------------------------
# Full pipeline: evaluate() with multiple new scorers
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_evaluate_with_new_scorers(self):
        from fastaiagent.eval import evaluate
        from fastaiagent.eval.rag import AnswerRelevancy
        from fastaiagent.eval.safety import PIILeakage, Toxicity
        from fastaiagent.eval.similarity import BLEUScore

        def agent_fn(input_text: str) -> str:
            responses = {
                "What is 2+2?": "The answer is 4.",
                "Capital of France?": "Paris is the capital of France.",
            }
            return responses.get(input_text, "I don't know.")

        results = evaluate(
            agent_fn=agent_fn,
            dataset=[
                {"input": "What is 2+2?", "expected": "The answer is 4."},
                {"input": "Capital of France?", "expected": "Paris is the capital of France."},
            ],
            scorers=[
                BLEUScore(),
                PIILeakage(),
                Toxicity(),
                AnswerRelevancy(),
            ],
        )

        summary = results.summary()
        print("\n" + summary)

        # BLEU should be perfect (exact match)
        assert all(s.passed for s in results.scores["bleu"])
        # No PII
        assert all(s.passed for s in results.scores["pii_leakage"])
        # Safe content
        assert all(s.passed for s in results.scores["toxicity"])
        # Relevant answers
        assert all(s.passed for s in results.scores["answer_relevancy"])
