"""Multi-turn session evaluation scorers.

``ConversationCoherence`` and ``GoalCompletion`` default to fast, zero-dependency
heuristics (``mode="heuristic"``). Pass ``mode="llm"`` to score with an LLM judge
instead — the heuristic path is unchanged. ``KnowledgeRetention``, ``RoleAdherence``,
and ``ConversationRelevancy`` are LLM-judged turn metrics.

All scorers take the conversation through a ``turns`` kwarg: a list of dicts with at
least ``"content"`` (and usually ``"role"``), e.g.
``[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]``.
"""

from __future__ import annotations

import json
from typing import Any

from fastaiagent._internal.async_utils import run_sync
from fastaiagent.eval.scorer import Scorer, ScorerResult


def _format_turns(turns: list[Any]) -> str:
    """Render conversation turns as role-prefixed lines (dict- or object-tolerant)."""
    lines = []
    for t in turns:
        if isinstance(t, dict):
            role = t.get("role", "?")
            content = t.get("content", "")
        else:
            role = getattr(t, "role", "?")
            content = getattr(t, "content", str(t))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _judge_score(
    llm: Any, system_prompt: str, user_prompt: str, threshold: float
) -> ScorerResult:
    """Run a single 0-1 judge call and parse ``{"score","reasoning"}`` into a result.

    Shared by every LLM-judged session metric; fails closed with a readable reason
    rather than raising, matching the other eval scorers.
    """
    from fastaiagent.eval.agent_metrics import _strip_fences
    from fastaiagent.llm import SystemMessage, UserMessage

    try:
        resp = await llm.acomplete([SystemMessage(system_prompt), UserMessage(user_prompt)])
        data = json.loads(_strip_fences(resp.content or "{}"))
        score_val = float(data.get("score", 0.0))
        return ScorerResult(
            score=score_val,
            passed=score_val >= threshold,
            reason=str(data.get("reasoning", "")),
        )
    except Exception as e:
        return ScorerResult(score=0.0, passed=False, reason=f"Session judge error: {e}")


class ConversationCoherence(Scorer):
    """Evaluates coherence across a multi-turn conversation.

    ``mode="heuristic"`` (default) checks for self-contradiction signals and topic
    drift across turns — fast, no LLM. ``mode="llm"`` judges coherence with an LLM
    instead (``threshold`` then governs pass/fail on the 0-1 score).

    Pass ``turns`` as a kwarg: a list of dicts with at least a ``"content"`` key.
    """

    name = "conversation_coherence"

    def __init__(self, *, mode: str = "heuristic", llm: Any = None, threshold: float = 0.5):
        if mode not in ("heuristic", "llm"):
            raise ValueError(f"Unknown mode {mode!r}. Use 'heuristic' or 'llm'.")
        self.mode = mode
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if self.mode == "llm":
            return run_sync(self._ascore_llm(input, output, expected, **kw))

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

    async def _ascore_llm(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient

        turns = kw.get("turns", [])
        if not turns:
            return ScorerResult(score=1.0, passed=True, reason="No turns to evaluate")
        llm = self._llm or LLMClient()
        prompt = (
            "Evaluate the coherence of this multi-turn conversation. A coherent "
            "conversation stays on-topic, avoids self-contradiction across turns, and "
            "maintains a consistent stance.\n\n"
            f"Transcript:\n{_format_turns(turns)}\n\n"
            "Score 0.0 (incoherent / self-contradictory) to 1.0 (fully coherent).\n"
            'Respond with JSON only: {"score": <0.0-1.0>, "reasoning": "<short>"}'
        )
        return await _judge_score(
            llm,
            "You evaluate conversational coherence. Respond with JSON only.",
            prompt,
            self.threshold,
        )


class GoalCompletion(Scorer):
    """Evaluates whether the conversation achieved its goal.

    ``mode="heuristic"`` (default) compares the goal (from the ``goal`` kwarg or
    ``expected``) against the final output using keyword recall, key-phrase matching,
    and optional checklist completion — fast, no LLM. ``mode="llm"`` judges goal
    completion with an LLM over the transcript (or final output) instead.
    """

    name = "goal_completion"

    def __init__(self, *, mode: str = "heuristic", llm: Any = None, threshold: float = 0.5):
        if mode not in ("heuristic", "llm"):
            raise ValueError(f"Unknown mode {mode!r}. Use 'heuristic' or 'llm'.")
        self.mode = mode
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if self.mode == "llm":
            return run_sync(self._ascore_llm(input, output, expected, **kw))

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

    async def _ascore_llm(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient

        goal = kw.get("goal", expected)
        if not goal:
            return ScorerResult(score=0.0, passed=False, reason="No goal specified")
        turns = kw.get("turns", [])
        convo = _format_turns(turns) if turns else f"assistant: {output}"
        llm = self._llm or LLMClient()
        prompt = (
            f"Goal: {goal}\n\n"
            "Did this conversation achieve the goal? Consider the final outcome and "
            "whether the user's objective was actually met.\n\n"
            f"Conversation:\n{convo}\n\n"
            "Score 0.0 (goal not met) to 1.0 (fully achieved).\n"
            'Respond with JSON only: {"score": <0.0-1.0>, "reasoning": "<short>"}'
        )
        return await _judge_score(
            llm, "You evaluate goal completion. Respond with JSON only.", prompt, self.threshold
        )


class KnowledgeRetention(Scorer):
    """LLM-judged: does the agent retain and reuse information the user gave earlier?

    Penalizes re-asking for details already provided, or contradicting earlier facts.
    Pass ``turns``. Score 1.0 = perfect retention.

    Example:
        scorer = KnowledgeRetention()
        result = scorer.score(input="", output="", turns=conversation)
    """

    name = "knowledge_retention"

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
        from fastaiagent.llm import LLMClient

        turns = kw.get("turns", [])
        if not turns:
            return ScorerResult(score=1.0, passed=True, reason="No turns to evaluate")
        llm = self._llm or LLMClient()
        prompt = (
            "Across this conversation, does the assistant correctly retain and reuse "
            "information the user provided earlier (names, numbers, preferences) without "
            "re-asking for it or contradicting it?\n\n"
            f"Transcript:\n{_format_turns(turns)}\n\n"
            "Score 0.0 (forgets or re-asks) to 1.0 (perfect retention).\n"
            'Respond with JSON only: {"score": <0.0-1.0>, "reasoning": "<short>"}'
        )
        return await _judge_score(
            llm,
            "You evaluate conversational memory. Respond with JSON only.",
            prompt,
            self.threshold,
        )


class RoleAdherence(Scorer):
    """LLM-judged: does the assistant stay in its assigned role across the conversation?

    Requires a ``role`` (the persona the assistant should maintain) — set it on the
    constructor or pass it as a ``role`` kwarg — plus ``turns``. Score 1.0 = fully in role.

    Example:
        scorer = RoleAdherence(role="a formal banking assistant")
        result = scorer.score(input="", output="", turns=conversation)
    """

    name = "role_adherence"

    def __init__(self, role: str | None = None, llm: Any = None, threshold: float = 0.7):
        self.role = role
        self._llm = llm
        self.threshold = threshold

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        return run_sync(self.ascore(input, output, expected, **kw))

    async def ascore(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        from fastaiagent.llm import LLMClient

        role = kw.get("role", self.role)
        if not role:
            return ScorerResult(score=0.0, passed=False, reason="No role specified")
        turns = kw.get("turns", [])
        if not turns:
            return ScorerResult(score=1.0, passed=True, reason="No turns to evaluate")
        llm = self._llm or LLMClient()
        prompt = (
            f"The assistant is supposed to act strictly as: {role}.\n"
            "Does it stay in this role/persona consistently across the conversation, "
            "without breaking character or violating the role's constraints?\n\n"
            f"Transcript:\n{_format_turns(turns)}\n\n"
            "Score 0.0 (breaks role) to 1.0 (fully adheres).\n"
            'Respond with JSON only: {"score": <0.0-1.0>, "reasoning": "<short>"}'
        )
        return await _judge_score(
            llm, "You evaluate role adherence. Respond with JSON only.", prompt, self.threshold
        )


class ConversationRelevancy(Scorer):
    """LLM-judged: are the assistant's replies relevant to what the user asked each turn?

    Pass ``turns``. Score 1.0 = every reply is on-topic and responsive.

    Example:
        scorer = ConversationRelevancy()
        result = scorer.score(input="", output="", turns=conversation)
    """

    name = "conversation_relevancy"

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
        from fastaiagent.llm import LLMClient

        turns = kw.get("turns", [])
        if not turns:
            return ScorerResult(score=1.0, passed=True, reason="No turns to evaluate")
        llm = self._llm or LLMClient()
        prompt = (
            "Are the assistant's responses relevant and responsive to what the user asked "
            "at each turn (no off-topic, evasive, or non-sequitur replies)?\n\n"
            f"Transcript:\n{_format_turns(turns)}\n\n"
            "Score 0.0 (irrelevant) to 1.0 (fully relevant).\n"
            'Respond with JSON only: {"score": <0.0-1.0>, "reasoning": "<short>"}'
        )
        return await _judge_score(
            llm,
            "You evaluate conversational relevancy. Respond with JSON only.",
            prompt,
            self.threshold,
        )
