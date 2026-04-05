"""Multi-turn session evaluation scorers."""

from __future__ import annotations

from typing import Any

from fastaiagent.eval.scorer import Scorer, ScorerResult


class ConversationCoherence(Scorer):
    """Evaluates coherence across a multi-turn conversation.

    Checks for self-contradiction signals and topic drift across turns.
    Pass ``turns`` as a kwarg: a list of dicts with at least a ``"content"`` key.
    """

    name = "conversation_coherence"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        turns = kw.get("turns", [])
        if not turns:
            return ScorerResult(score=1.0, passed=True, reason="No turns to evaluate")

        contents = [
            t.get("content", "") if isinstance(t, dict) else str(t) for t in turns
        ]

        penalties = 0.0
        total_checks = 0

        # 1. Contradiction signals: later turns that negate earlier claims
        negation_phrases = [
            "actually, ",
            "i was wrong",
            "correction:",
            "that is incorrect",
            "that's incorrect",
            "i made an error",
            "let me correct",
            "i apologize, ",
            "sorry, that was wrong",
            "on second thought",
            "i misspoke",
            "disregard my previous",
            "contrary to what i said",
        ]
        for content in contents[1:]:
            total_checks += 1
            lower = content.lower()
            if any(phrase in lower for phrase in negation_phrases):
                penalties += 1.0

        # 2. Topic drift: measure vocabulary overlap between consecutive turns
        for i in range(1, len(contents)):
            total_checks += 1
            prev_words = set(contents[i - 1].lower().split())
            curr_words = set(contents[i].lower().split())
            union = prev_words | curr_words
            if union:
                overlap = len(prev_words & curr_words) / len(union)
                if overlap < 0.05:
                    penalties += 1.0

        if total_checks == 0:
            return ScorerResult(score=1.0, passed=True, reason="No checks applicable")

        score = max(1.0 - (penalties / total_checks), 0.0)
        return ScorerResult(
            score=score,
            passed=score >= 0.5,
            reason=f"Penalties: {penalties:.0f}/{total_checks} checks",
        )


class GoalCompletion(Scorer):
    """Evaluates whether the conversation achieved its goal.

    Compares the goal (from ``goal`` kwarg or ``expected``) against the final
    output using keyword recall, key-phrase matching, and optional
    checklist completion.
    """

    name = "goal_completion"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        goal = kw.get("goal", expected)
        if not goal:
            return ScorerResult(score=0.0, passed=False, reason="No goal specified")

        output_lower = output.lower()
        goal_lower = goal.lower()

        # 1. Keyword recall — filter out stop words for a fairer signal
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "about", "that", "this", "it", "and", "or",
            "but", "if", "not", "no", "so", "up", "out", "then", "than",
        }
        goal_words = {
            w for w in goal_lower.split() if w not in stop_words and len(w) > 1
        }
        if not goal_words:
            goal_words = set(goal_lower.split())

        matched = sum(1 for w in goal_words if w in output_lower)
        keyword_recall = matched / max(len(goal_words), 1)

        # 2. Key-phrase matching — bigrams from the goal
        goal_tokens = goal_lower.split()
        bigrams = [
            f"{goal_tokens[i]} {goal_tokens[i + 1]}"
            for i in range(len(goal_tokens) - 1)
        ]
        if bigrams:
            bigram_hits = sum(1 for bg in bigrams if bg in output_lower)
            phrase_score = bigram_hits / len(bigrams)
        else:
            phrase_score = keyword_recall

        # 3. Checklist detection — if the goal contains numbered/bulleted items
        import re

        checklist_items = re.findall(
            r"(?:^|\n)\s*(?:\d+[.)]\s*|-\s*|\*\s*)(.+)", goal
        )
        checklist_score = 0.0
        if checklist_items:
            items_found = sum(
                1
                for item in checklist_items
                if item.strip().lower() in output_lower
                or all(
                    w in output_lower
                    for w in item.strip().lower().split()
                    if w not in stop_words and len(w) > 1
                )
            )
            checklist_score = items_found / len(checklist_items)
            score = 0.4 * keyword_recall + 0.3 * phrase_score + 0.3 * checklist_score
        else:
            score = 0.6 * keyword_recall + 0.4 * phrase_score

        return ScorerResult(
            score=round(score, 3),
            passed=score >= 0.3,
            reason=(
                f"keyword_recall={keyword_recall:.2f} "
                f"phrase={phrase_score:.2f}"
                + (
                    f" checklist={checklist_score:.2f}"
                    if checklist_items
                    else ""
                )
            ),
        )
