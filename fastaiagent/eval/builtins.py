"""Built-in evaluation scorers."""

from __future__ import annotations

import json
import re
from typing import Any

from fastaiagent.eval.scorer import Scorer, ScorerResult


class ExactMatch(Scorer):
    name = "exact_match"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if expected is None:
            return ScorerResult(score=0.0, passed=False, reason="No expected output")
        passed = output.strip() == expected.strip()
        return ScorerResult(score=1.0 if passed else 0.0, passed=passed)


class Contains(Scorer):
    name = "contains"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if expected is None:
            return ScorerResult(score=0.0, passed=False, reason="No expected output")
        passed = expected.lower() in output.lower()
        return ScorerResult(score=1.0 if passed else 0.0, passed=passed)


class JSONValid(Scorer):
    name = "json_valid"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        try:
            json.loads(output)
            return ScorerResult(score=1.0, passed=True)
        except (json.JSONDecodeError, TypeError):
            return ScorerResult(score=0.0, passed=False, reason="Invalid JSON")


class RegexMatch(Scorer):
    name = "regex_match"

    def __init__(self, pattern: str):
        self.pattern = pattern

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        passed = bool(re.search(self.pattern, output))
        return ScorerResult(score=1.0 if passed else 0.0, passed=passed)


class LengthBetween(Scorer):
    name = "length_between"

    def __init__(self, min_len: int = 0, max_len: int = 10000):
        self.min_len = min_len
        self.max_len = max_len

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        length = len(output)
        passed = self.min_len <= length <= self.max_len
        return ScorerResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"Length: {length}",
        )


class Latency(Scorer):
    name = "latency"

    def __init__(self, max_ms: int = 5000):
        self.max_ms = max_ms

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        latency_ms = kw.get("latency_ms", 0)
        passed = latency_ms <= self.max_ms
        return ScorerResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"Latency: {latency_ms}ms (max: {self.max_ms}ms)",
        )


class CostUnder(Scorer):
    name = "cost_under"

    def __init__(self, max_usd: float = 0.10):
        self.max_usd = max_usd

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        cost = kw.get("cost", 0.0)
        passed = cost <= self.max_usd
        return ScorerResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"Cost: ${cost:.4f} (max: ${self.max_usd:.4f})",
        )


# Registry of built-in scorers by name
BUILTIN_SCORERS: dict[str, type[Scorer]] = {
    # Core
    "exact_match": ExactMatch,
    "contains": Contains,
    "json_valid": JSONValid,
    "regex_match": RegexMatch,
    "length_between": LengthBetween,
    "latency": Latency,
    "cost_under": CostUnder,
}


def _register_extended_scorers() -> None:
    """Lazily register RAG, safety, and similarity scorers on first access."""
    from fastaiagent.eval.rag import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from fastaiagent.eval.safety import Bias, PIILeakage, Toxicity
    from fastaiagent.eval.similarity import BLEUScore, LevenshteinDistance, ROUGEScore, SemanticSimilarity

    BUILTIN_SCORERS.update(
        {
            # RAG
            "faithfulness": Faithfulness,
            "answer_relevancy": AnswerRelevancy,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            # Safety
            "toxicity": Toxicity,
            "bias": Bias,
            "pii_leakage": PIILeakage,
            # Similarity & NLP
            "semantic_similarity": SemanticSimilarity,
            "bleu": BLEUScore,
            "rouge": ROUGEScore,
            "levenshtein": LevenshteinDistance,
        }
    )


_register_extended_scorers()
