"""Example 77: Session scorers on a REAL multi-turn conversation.

Demonstrates (real Agent + real LLM — needs OPENAI_API_KEY):
- ConversationCoherence : self-contradiction + topic-drift across turns
- GoalCompletion        : did the conversation achieve its goal?

We drive a real 3-turn conversation by feeding prior turns back to the agent via
`agent.run(input, messages=history)`, then score the real transcript. Session
scorers take the conversation through keyword arguments (`turns` / `goal`), so we
call `.score(...)` directly — `evaluate()` only forwards `input`/`expected`.

Run:
    zsh -lc 'python examples/77_session_eval.py'
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import ConversationCoherence, GoalCompletion
from fastaiagent.llm import AssistantMessage, UserMessage


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    agent = Agent(
        name="shipping-support",
        system_prompt=(
            "You are a concise shipping-support agent. Order #12345 shipped via FedEx "
            "and is due Friday. Stay consistent across the conversation."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )

    user_turns = [
        "Where is my order #12345?",
        "Which carrier is it with?",
        "Great — when will it arrive?",
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

    print("\n== Session scores ==")
    # GoalCompletion compares goal keywords against the conversation outcome, so
    # score it against all assistant turns joined together.
    outcome = " ".join(t["content"] for t in transcript if t["role"] == "assistant")
    coherence = ConversationCoherence().score(input="", output="", turns=transcript)
    goal = GoalCompletion().score(
        input="",
        output=outcome,
        goal="The order shipped via FedEx and arrives Friday",
    )
    for name, r in [("conversation_coherence", coherence), ("goal_completion", goal)]:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {name:<22} score={r.score:.2f} — {r.reason}")


if __name__ == "__main__":
    main()
