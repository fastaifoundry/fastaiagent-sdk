"""Agent Hardening & Scorecard — the close-the-loop eval workflow.

    generate_scenarios → simulate → Scorecard → harden → (apply + repeat)

* generate_scenarios()  — auto-create test scenarios by introspecting the agent
* TaskCompletion / Hallucination / ReflectionQuality — named eval metrics
* Scorecard             — roll up any eval/sim run into a per-metric panel
* harden()              — turn failures into concrete fix recommendations

The Scorecard demo runs anywhere (no key). The full loop runs when OPENAI_API_KEY
is set; it reuses your LLMClient.

Run:
    zsh -lc 'python examples/74_agent_hardening.py'
"""

from __future__ import annotations

import os

from fastaiagent import (
    Agent,
    LLMClient,
    Scorecard,
    generate_scenarios,
    harden,
    simulate,
)
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.scorer import ScorerResult


def demo_scorecard_offline() -> None:
    """Scorecard aggregation needs no LLM."""
    print("== Scorecard (offline) ==")
    results = EvalResults(
        scores={
            "task_completion": [
                ScorerResult(score=1.0, passed=True),
                ScorerResult(score=0.4, passed=False),
            ],
            "faithfulness": [
                ScorerResult(score=0.9, passed=True),
                ScorerResult(score=0.95, passed=True),
            ],
        }
    )
    print(Scorecard.from_eval_results(results, label="support-v1").summary())


def demo_full_loop() -> None:
    """generate → simulate → score → harden, against a live model."""
    llm = LLMClient(provider="openai", model="gpt-4o-mini")

    # A deliberately thin agent so the loop finds something to fix.
    agent = Agent(
        name="support",
        system_prompt="You are a support agent. Answer briefly.",
        llm=llm,
    )

    print("\n== 1. generate_scenarios ==")
    scenarios = generate_scenarios(agent, n=3, llm=llm, focus="refunds and order status")
    for s in scenarios:
        print(f"  • {s.name}: {s.user.persona}")

    print("\n== 2. simulate ==")
    results = simulate(scenarios, agent)
    print(results.summary())

    print("\n== 3. Scorecard ==")
    print(Scorecard.from_simulation(results).summary())

    print("\n== 4. harden (recommend-only) ==")
    report = harden(agent, results, llm=llm)
    print(report.summary())


def main() -> None:
    demo_scorecard_offline()
    if os.environ.get("OPENAI_API_KEY"):
        demo_full_loop()
    else:
        print("\n(Set OPENAI_API_KEY to run the full generate→simulate→score→harden loop.)")


if __name__ == "__main__":
    main()
