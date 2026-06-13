"""Example 82: LLM-judged multi-turn metrics on a REAL conversation.

Drives a real 3-turn conversation, then scores the transcript with the LLM-judged
session metrics (real Agent + real LLM — needs OPENAI_API_KEY):
- ConversationCoherence(mode="llm") : coherence judged by an LLM (vs. the heuristic default)
- GoalCompletion(mode="llm")        : did the conversation meet its goal?
- KnowledgeRetention                : does the agent reuse info the user gave earlier?
- RoleAdherence                     : does the agent stay in its assigned role?
- ConversationRelevancy             : are replies relevant to each user turn?

Session scorers take the conversation through keyword args (`turns` / `goal` / `role`), so
we call `.score(...)` directly. The two upgraded scorers default to `mode="heuristic"`
(no LLM, no key) — `mode="llm"` is opt-in.

Run:
    zsh -lc 'python examples/82_llm_session_metrics.py'
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import (
    ConversationCoherence,
    ConversationRelevancy,
    GoalCompletion,
    KnowledgeRetention,
    RoleAdherence,
)
from fastaiagent.llm import AssistantMessage, UserMessage


def _show(label: str, result) -> None:  # noqa: ANN001 - example brevity
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {label:<24} score={result.score:.2f} — {result.reason}")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    role = "a concise shipping-support agent who stays consistent across the conversation"
    agent = Agent(
        name="shipping-support",
        system_prompt=(
            f"You are {role}. Order #12345 shipped via FedEx and is due Friday. "
            "Remember details the customer gives you."
        ),
        llm=llm,
    )

    user_turns = [
        "Hi, I'm Dana — where is my order #12345?",
        "Which carrier is it with?",
        "Great — when will it arrive, and can you address me by name?",
    ]

    history: list = []
    transcript: list[dict] = []
    for user_text in user_turns:
        result = agent.run(user_text, messages=history)
        history = history + [UserMessage(user_text), AssistantMessage(result.output)]
        transcript.append({"role": "user", "content": user_text})
        transcript.append({"role": "assistant", "content": result.output})

    print("== Conversation ==")
    for turn in transcript:
        print(f"  {turn['role']:>9}: {turn['content']}")

    print("\n== LLM-judged session metrics ==")
    _show(
        "coherence (llm)",
        ConversationCoherence(mode="llm", llm=llm).score(input="", output="", turns=transcript),
    )
    _show(
        "goal_completion (llm)",
        GoalCompletion(mode="llm", llm=llm).score(
            input="",
            output="",
            goal="Tell Dana the carrier and arrival day for order #12345, addressing her by name.",
            turns=transcript,
        ),
    )
    _show(
        "knowledge_retention",
        KnowledgeRetention(llm=llm).score(input="", output="", turns=transcript),
    )
    _show(
        "role_adherence",
        RoleAdherence(role=role, llm=llm).score(input="", output="", turns=transcript),
    )
    _show(
        "conversation_relevancy",
        ConversationRelevancy(llm=llm).score(input="", output="", turns=transcript),
    )


if __name__ == "__main__":
    main()
