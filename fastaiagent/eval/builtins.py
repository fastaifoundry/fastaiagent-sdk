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
    """Budget gate on wall-clock time.

    ``latency_ms`` is supplied by :func:`fastaiagent.eval.evaluate` — from the
    result when the callable returns an :class:`~fastaiagent.agent.AgentResult`,
    and measured around the call otherwise. Before 1.67.0 ``evaluate`` passed
    neither, so this gate read ``0`` out of an empty ``**kwargs`` and reported
    "Latency: 0ms" for every case it ever scored: a documented budget gate that
    could not fail.
    """

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
    """Budget gate on USD spend.

    ``cost`` and ``cost_known`` are supplied by
    :func:`fastaiagent.eval.evaluate` from the run's
    :class:`~fastaiagent.agent.AgentResult`. Before 1.67.0 nothing populated
    ``AgentResult.cost`` and ``evaluate`` passed no ``cost`` at all, so this
    gate read ``0.0`` and reported "Cost: $0.0000" for every case — a
    documented budget gate that could not fail.

    **Unknown is not free.** A model with no rate in the pricing table (a local
    ollama model, a private fine-tune, a bedrock/azure deployment id) yields
    ``cost_known=False``. Certifying that run as under budget would be the
    §2.4 shape the guardrail sweep exists to prevent: a check that could not
    check anything returning a clean verdict. It fails, and says why.
    """

    name = "cost_under"

    def __init__(self, max_usd: float = 0.10):
        self.max_usd = max_usd

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        cost = kw.get("cost", 0.0)
        # Absent entirely -> an old caller passing ``cost=`` by hand, which has
        # always meant "here is the cost". Present and False -> the run really
        # could not be priced.
        cost_known = kw.get("cost_known", True)
        if not cost_known:
            return ScorerResult(
                score=0.0,
                passed=False,
                reason=(
                    f"Cost: unknown (max: ${self.max_usd:.4f}) — no rate for this "
                    "model, so the run could not be priced. An unpriced run is "
                    "not a free one; set a rate via set_rate_overrides() or the "
                    "models.json 'pricing' block to gate on it."
                ),
            )
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
    """Lazily register RAG, safety, similarity, and agent-eval scorers."""
    from fastaiagent.eval.agent_metrics import Hallucination, ReflectionQuality, TaskCompletion
    from fastaiagent.eval.rag import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
    from fastaiagent.eval.safety import (
        Bias,
        OpenAIModeration,
        PIILeakage,
        PromptInjection,
        Toxicity,
    )
    from fastaiagent.eval.similarity import (
        BLEUScore,
        LevenshteinDistance,
        ROUGEScore,
        SemanticSimilarity,
    )

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
            "prompt_injection": PromptInjection,
            "moderation": OpenAIModeration,
            # Agent-eval metrics
            "task_completion": TaskCompletion,
            "hallucination": Hallucination,
            "reflection_quality": ReflectionQuality,
            # Similarity & NLP
            "semantic_similarity": SemanticSimilarity,
            "bleu": BLEUScore,
            "rouge": ROUGEScore,
            "levenshtein": LevenshteinDistance,
        }
    )


_register_extended_scorers()
