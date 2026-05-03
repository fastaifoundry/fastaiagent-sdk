"""Similarity and classical NLP evaluation scorers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from fastaiagent.eval.scorer import Scorer, ScorerResult


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Pure-Python cosine similarity between two equal-length vectors.

    Inlined here so the scorer doesn't depend on the optional ``[kb]``
    extra (numpy/faiss). Returns 0.0 when either vector is zero-length
    or zero-magnitude rather than raising.
    """
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class SemanticSimilarity(Scorer):
    """Cosine similarity between embeddings of output and expected.

    Uses the SDK's embedding infrastructure — auto-detects the best
    available embedder (FastEmbed local > OpenAI API > SimpleEmbedder).

    Example:
        scorer = SemanticSimilarity(threshold=0.8)
        result = scorer.score(input="q", output="Paris is the capital", expected="Capital of France is Paris")
    """

    name = "semantic_similarity"

    def __init__(self, embedder: Any = None, threshold: float = 0.7):
        self.threshold = threshold
        self._embedder = embedder

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder
        from fastaiagent.kb.embedding import get_default_embedder

        return get_default_embedder()

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if expected is None:
            return ScorerResult(score=0.0, passed=False, reason="No expected output")

        embedder = self._get_embedder()
        vecs = embedder.embed([output, expected])
        similarity = _cosine_similarity(vecs[0], vecs[1])
        similarity = max(0.0, min(1.0, similarity))

        return ScorerResult(
            score=round(similarity, 4),
            passed=similarity >= self.threshold,
            reason=f"Cosine similarity: {similarity:.4f} (threshold: {self.threshold})",
        )


class BLEUScore(Scorer):
    """BLEU score — n-gram precision with brevity penalty.

    Pure Python implementation. No LLM or API calls.

    Example:
        scorer = BLEUScore(max_n=4, threshold=0.3)
        result = scorer.score(input="q", output="the cat sat on the mat", expected="the cat is on the mat")
    """

    name = "bleu"

    def __init__(self, max_n: int = 4, threshold: float = 0.3):
        self.max_n = max_n
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if expected is None:
            return ScorerResult(score=0.0, passed=False, reason="No expected output")

        out_tokens = output.lower().split()
        ref_tokens = expected.lower().split()

        if not out_tokens or not ref_tokens:
            return ScorerResult(score=0.0, passed=False, reason="Empty text")

        # Brevity penalty
        bp = math.exp(1 - len(ref_tokens) / len(out_tokens)) if len(out_tokens) < len(ref_tokens) else 1.0

        # N-gram precisions
        log_precisions = []
        for n in range(1, self.max_n + 1):
            precision = self._ngram_precision(out_tokens, ref_tokens, n)
            if precision == 0:
                # If any n-gram precision is 0, BLEU is 0
                return ScorerResult(
                    score=0.0,
                    passed=False,
                    reason=f"Zero {n}-gram precision",
                )
            log_precisions.append(math.log(precision))

        bleu = bp * math.exp(sum(log_precisions) / len(log_precisions))
        bleu = max(0.0, min(1.0, bleu))

        return ScorerResult(
            score=round(bleu, 4),
            passed=bleu >= self.threshold,
            reason=f"BLEU-{self.max_n}: {bleu:.4f}",
        )

    @staticmethod
    def _ngram_precision(output_tokens: list[str], ref_tokens: list[str], n: int) -> float:
        if len(output_tokens) < n or len(ref_tokens) < n:
            return 0.0

        # Count n-grams in reference
        ref_ngrams: dict[tuple[str, ...], int] = {}
        for i in range(len(ref_tokens) - n + 1):
            ng = tuple(ref_tokens[i : i + n])
            ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

        # Count clipped matches in output
        matches = 0
        for i in range(len(output_tokens) - n + 1):
            ng = tuple(output_tokens[i : i + n])
            if ref_ngrams.get(ng, 0) > 0:
                matches += 1
                ref_ngrams[ng] -= 1

        total = len(output_tokens) - n + 1
        return matches / total if total > 0 else 0.0


class ROUGEScore(Scorer):
    """ROUGE score — recall-oriented n-gram evaluation.

    Supports ``rouge-1`` (unigram recall) and ``rouge-l`` (LCS-based F1).
    Pure Python implementation. No LLM or API calls.

    Example:
        scorer = ROUGEScore(variant="rouge-1")
        result = scorer.score(input="q", output="the cat sat on the mat", expected="the cat is on the mat")
    """

    name = "rouge"

    def __init__(self, variant: str = "rouge-1", threshold: float = 0.3):
        if variant not in ("rouge-1", "rouge-l"):
            raise ValueError(f"Unsupported variant: {variant}. Use 'rouge-1' or 'rouge-l'.")
        self.variant = variant
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if expected is None:
            return ScorerResult(score=0.0, passed=False, reason="No expected output")

        out_tokens = output.lower().split()
        ref_tokens = expected.lower().split()

        if not out_tokens or not ref_tokens:
            return ScorerResult(score=0.0, passed=False, reason="Empty text")

        if self.variant == "rouge-1":
            rouge = self._rouge_1(out_tokens, ref_tokens)
        else:
            rouge = self._rouge_l(out_tokens, ref_tokens)

        return ScorerResult(
            score=round(rouge, 4),
            passed=rouge >= self.threshold,
            reason=f"{self.variant}: {rouge:.4f}",
        )

    @staticmethod
    def _rouge_1(out_tokens: list[str], ref_tokens: list[str]) -> float:
        """Unigram F1."""
        out_set = {}
        for t in out_tokens:
            out_set[t] = out_set.get(t, 0) + 1
        ref_set = {}
        for t in ref_tokens:
            ref_set[t] = ref_set.get(t, 0) + 1

        overlap = 0
        for t, count in ref_set.items():
            overlap += min(count, out_set.get(t, 0))

        precision = overlap / len(out_tokens) if out_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _rouge_l(out_tokens: list[str], ref_tokens: list[str]) -> float:
        """LCS-based F1."""
        m, n = len(out_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if out_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs_len = dp[m][n]

        precision = lcs_len / m if m > 0 else 0.0
        recall = lcs_len / n if n > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class LevenshteinDistance(Scorer):
    """Normalized Levenshtein similarity (1 - normalized edit distance).

    Pure Python implementation. No LLM or API calls.

    Example:
        scorer = LevenshteinDistance(threshold=0.7)
        result = scorer.score(input="q", output="kitten", expected="sitting")
    """

    name = "levenshtein"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if expected is None:
            return ScorerResult(score=0.0, passed=False, reason="No expected output")

        a, b = output.lower(), expected.lower()
        max_len = max(len(a), len(b))
        if max_len == 0:
            return ScorerResult(score=1.0, passed=True, reason="Both empty")

        dist = self._edit_distance(a, b)
        similarity = 1.0 - (dist / max_len)

        return ScorerResult(
            score=round(similarity, 4),
            passed=similarity >= self.threshold,
            reason=f"Edit distance: {dist}, similarity: {similarity:.4f}",
        )

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        return prev[n]
