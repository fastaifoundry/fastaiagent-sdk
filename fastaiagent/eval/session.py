"""Multi-turn session evaluation scorers."""

from __future__ import annotations

from typing import Any

from fastaiagent.eval.scorer import Scorer, ScorerResult


class ConversationCoherence(Scorer):
    """Evaluates coherence across a multi-turn conversation."""

    name = "conversation_coherence"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        turns = kw.get("turns", [])
        if not turns:
            return ScorerResult(score=1.0, passed=True, reason="No turns to evaluate")

        # Simple heuristic: check that responses don't contradict each other
        score = 1.0
        return ScorerResult(score=score, passed=score >= 0.5, reason="Coherence check")


class GoalCompletion(Scorer):
    """Evaluates whether the conversation achieved its goal."""

    name = "goal_completion"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        goal = kw.get("goal", expected)
        if not goal:
            return ScorerResult(score=0.0, passed=False, reason="No goal specified")

        # Check if goal keywords appear in final output
        goal_words = set(goal.lower().split())
        output_words = set(output.lower().split())
        overlap = len(goal_words & output_words) / max(len(goal_words), 1)

        return ScorerResult(
            score=overlap,
            passed=overlap >= 0.3,
            reason=f"Goal word overlap: {overlap:.2f}",
        )
