"""``DecisionJudge`` — an eval judge on OpenAI's Decisions API (1.84.0).

:class:`~fastaiagent.eval.llm_judge.LLMJudge` asks a chat model for a JSON verdict
and parses a number out of free text. ``DecisionJudge`` asks the Decisions API
instead, which answers with a probability distribution and nothing to parse:

* **Predicate** (default) — ``criteria`` is a statement that should be true of a
  good output; the score is the probability that it is.
* **Score** — pass ``levels`` (worst first); the score is the expected level,
  rescaled to 0..1.

``passed = score >= threshold`` either way, so it drops into ``evaluate()``, the
pytest gates and ``simulate()`` wherever an ``LLMJudge`` goes.

A refusal or a failed call scores ``0.0`` with the reason spelled out — the same
convention every scorer here uses — never a passing score.
"""

from __future__ import annotations

from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.llm_judge import JUDGE_PLACEHOLDERS, render_judge_template
from fastaiagent.eval.scorer import Scorer, ScorerResult
from fastaiagent.llm.decisions import (
    DEFAULT_DECISION_MODEL,
    Level,
    Predicate,
    PredicateAnswer,
    Refusal,
    Score,
    ScoreAnswer,
)

DEFAULT_TEMPLATE = "Input:\n{input}\n\nExpected output:\n{expected}\n\nActual output:\n{output}"


class DecisionJudge(Scorer):
    """Judge an output with one Decisions API question.

    Args:
        criteria: What a good output looks like, phrased as a statement —
            ``"The actual output answers the input correctly and completely."``
        levels: Optional ordered levels, worst first
            (``["Wrong", "Partially correct", "Correct"]`` or ``Level`` objects).
            Without them the judge asks a yes/no predicate.
        llm: An ``LLMClient`` (or anything with ``adecide``). Defaults to
            ``LLMClient(model="gpt-6-luna")``.
        threshold: Pass bar on the 0..1 score (default 0.5).
        template: Evidence template; accepts the same ``{input}`` / ``{output}``
            / ``{expected}`` placeholders (and aliases) as ``LLMJudge``.
        name: Scorer name in results (default ``"decision_judge"``).

    Example:
        judge = DecisionJudge("The actual output answers the input correctly.")
        result = judge.score(input="What is 2+2?", output="4", expected="4")

        graded = DecisionJudge(
            "How well does the actual output answer the input?",
            levels=["Wrong", "Partially correct", "Correct"],
            threshold=0.75,
        )
    """

    name = "decision_judge"

    def __init__(
        self,
        criteria: str,
        *,
        levels: list[str | Level] | None = None,
        llm: Any = None,
        threshold: float = 0.5,
        template: str | None = None,
        name: str | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in 0..1 (the score is), got {threshold}")
        self.criteria = criteria
        self.levels = levels
        self._llm = llm
        self.threshold = threshold
        self.template = template or DEFAULT_TEMPLATE
        if not any(p in self.template for p in JUDGE_PLACEHOLDERS):
            # A template with no placeholder would judge nothing but itself.
            raise ValueError(
                "template contains no {input} / {output} / {expected} placeholder, so "
                "the judge would never see what it is scoring."
            )
        # Built now so a criteria the Decisions API cannot ask (empty, one level)
        # fails at construction rather than scoring every case 0.0.
        self.question: Predicate | Score = (
            Score(name="judgement", instructions=criteria, levels=levels)  # type: ignore[arg-type]
            if levels is not None
            else Predicate(name="judgement", instructions=criteria)
        )
        if name is not None:
            self.name = name

    def with_criteria(self, criteria: str) -> DecisionJudge:
        """A copy judging a different criterion with the same client and settings."""
        return DecisionJudge(
            criteria,
            levels=self.levels,
            llm=self._llm,
            threshold=self.threshold,
            template=self.template,
            name=self.name,
        )

    def _client(self) -> Any:
        if self._llm is not None:
            return self._llm
        from fastaiagent.llm import LLMClient

        return LLMClient(model=DEFAULT_DECISION_MODEL)

    def score(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kwargs))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kwargs: Any
    ) -> ScorerResult:
        evidence = render_judge_template(self.template, input, output, expected or "N/A")
        try:
            result = await self._client().adecide(evidence, [self.question])
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Decision judge error: {e}")

        answer = result[0]
        if isinstance(answer, Refusal):
            return ScorerResult(
                score=0.0,
                passed=False,
                reason="Decision judge refused to answer this case; it was not scored.",
            )
        if isinstance(answer, PredicateAnswer):
            score = answer.probability
            reason = f"p={score:.2f} that: {self.criteria}"
        elif isinstance(answer, ScoreAnswer):
            score = answer.normalized
            reason = (
                f"most likely level {answer.level!r}; expected level {answer.score:.2f} "
                f"of {len(answer.probabilities) - 1} (normalized {score:.2f}, "
                f"confidence {answer.confidence:.2f})"
            )
        else:  # pragma: no cover — parse_decision already matched type to question
            return ScorerResult(score=0.0, passed=False, reason=f"Unexpected answer {answer!r}")
        return ScorerResult(score=score, passed=score >= self.threshold, reason=reason)


__all__ = ["DecisionJudge"]
