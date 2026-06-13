"""Agent-eval named metrics — task completion, hallucination, reflection quality.

These round out the AgentEval-style metric set alongside the existing RAG
scorers (``faithfulness``, ``context_precision``/``recall``), safety scorers
(``toxicity``, ``bias``, ``pii_leakage``, ``prompt_injection``, ``moderation``),
trajectory scorers (tool-call accuracy), and the LLM judge.

``Hallucination`` reuses the groundedness engine shared with the runtime
``grounded()`` guardrail and the ``faithfulness`` scorer — one core detector,
several surfaces.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.scorer import Scorer, ScorerResult


def _strip_fences(text: str) -> str:
    """Remove markdown code fences that LLMs sometimes wrap around JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _resolve_context(kw: dict[str, Any]) -> str | None:
    """Extract a context string from kwargs (supports 'context' or 'contexts')."""
    ctx = kw.get("context") or kw.get("contexts")
    if ctx is None:
        return None
    if isinstance(ctx, list):
        return "\n\n".join(str(c) for c in ctx)
    return str(ctx)


class TaskCompletion(Scorer):
    """Did the response accomplish the task / goal implied by the input?

    Single LLM call. Score 0.0 (task not addressed) .. 1.0 (fully completed).

    Example:
        scorer = TaskCompletion()
        result = scorer.score(
            input="Book me a table for 2 at 7pm and confirm.",
            output="Booked table for 2 at 7pm — confirmation #A12.",
        )
    """

    name = "task_completion"

    def __init__(self, llm: Any = None, threshold: float = 0.7):
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        llm = self._llm or LLMClient()
        expected_block = f"\nReference / expected outcome: {expected}\n" if expected else ""
        prompt = (
            "Judge whether the assistant's response accomplishes the user's task or goal.\n\n"
            f"User request: {input}\n"
            f"Assistant response: {output}\n"
            f"{expected_block}\n"
            "Score 0.0 (task not addressed at all) to 1.0 (task fully completed).\n"
            'Respond with JSON only: {"score": <0.0-1.0>, "reasoning": "<short>"}'
        )
        try:
            resp = await llm.acomplete(
                [
                    SystemMessage("You are a task-completion evaluator. Respond with JSON only."),
                    UserMessage(prompt),
                ]
            )
            data = json.loads(_strip_fences(resp.content or ""))
            score_val = float(data.get("score", 0.0))
            return ScorerResult(
                score=score_val,
                passed=score_val >= self.threshold,
                reason=str(data.get("reasoning", "")),
            )
        except Exception as e:
            return ScorerResult(score=0.0, passed=False, reason=f"Task-completion check error: {e}")


class Hallucination(Scorer):
    """Named hallucination metric — fraction of output claims supported by context.

    Reuses :func:`fastaiagent._internal.safety_detectors.score_groundedness` (the
    same engine behind the ``grounded()`` guardrail and the ``faithfulness``
    scorer). Higher score = fewer hallucinations. Requires ``context`` /
    ``contexts`` in kwargs.

    Example:
        scorer = Hallucination()
        result = scorer.score(
            input="Who wrote Hamlet?",
            output="Hamlet was written by Shakespeare in 1601.",
            context="Hamlet is a tragedy by William Shakespeare.",
        )
    """

    name = "hallucination"

    def __init__(self, llm: Any = None, threshold: float = 0.7):
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent._internal.safety_detectors import score_groundedness

        context = _resolve_context(kw)
        if not context:
            return ScorerResult(score=0.0, passed=False, reason="No context provided")
        res = await score_groundedness(output, context, llm=self._llm)
        reason = f"Grounded claims: {res.supported}/{res.total}" if res.total else res.reason
        return ScorerResult(score=res.score, passed=res.score >= self.threshold, reason=reason)


class ReflectionQuality(Scorer):
    """Internal consistency / reasoning quality of the response.

    Single LLM call judging whether the response is self-consistent, free of
    internal contradictions, and hedges appropriately on uncertain claims.
    Score 0.0 (contradictory / overconfident) .. 1.0 (consistent, well-reasoned).

    Example:
        scorer = ReflectionQuality()
        result = scorer.score(input="Is the Earth flat?", output="No — it is an oblate spheroid.")
    """

    name = "reflection_quality"

    def __init__(self, llm: Any = None, threshold: float = 0.7):
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

        llm = self._llm or LLMClient()
        prompt = (
            "Judge the reflective quality of the assistant's response: is it internally "
            "consistent, free of self-contradiction, and does it hedge appropriately on "
            "uncertain or unverifiable claims?\n\n"
            f"User request: {input}\n"
            f"Assistant response: {output}\n\n"
            "Score 0.0 (contradictory / overconfident / sloppy) to 1.0 "
            "(consistent, well-reasoned).\n"
            'Respond with JSON only: {"score": <0.0-1.0>, "reasoning": "<short>"}'
        )
        try:
            resp = await llm.acomplete(
                [
                    SystemMessage("You are a reasoning-quality evaluator. Respond with JSON only."),
                    UserMessage(prompt),
                ]
            )
            data = json.loads(_strip_fences(resp.content or ""))
            score_val = float(data.get("score", 0.0))
            return ScorerResult(
                score=score_val,
                passed=score_val >= self.threshold,
                reason=str(data.get("reasoning", "")),
            )
        except Exception as e:
            return ScorerResult(
                score=0.0, passed=False, reason=f"Reflection-quality check error: {e}"
            )
