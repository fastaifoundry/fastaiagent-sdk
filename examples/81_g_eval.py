"""Example 81: G-Eval judge (evaluation steps + rubric + chain-of-thought) on a REAL agent.

Demonstrates (real Agent + real LLM — needs OPENAI_API_KEY):
- GEval        : criteria + explicit evaluation_steps + a score-band rubric, scored with
                 chain-of-thought and normalized to 0-1.
- Separation   : the same judge scores a correct answer higher than a wrong one.
- Auto-CoT     : a GEval with NO evaluation_steps derives them from the criteria.
- Legacy path  : plain LLMJudge(criteria=...) still works, unchanged.

G-Eval is the richer, DeepEval-style judge. It's a normal scorer, so it also works inside
`evaluate(scorers=[GEval(...)])` and renders in the Local UI like any other scorer.

Run:
    zsh -lc 'python examples/81_g_eval.py'
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import GEval, LLMJudge


def _show(label: str, result) -> None:  # noqa: ANN001 - example brevity
    status = "PASS" if result.passed else "FAIL"
    print(f"  [{status}] {label:<22} score={result.score:.2f} — {result.reason}")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    agent = Agent(
        name="geography-tutor",
        system_prompt="You are a concise geography tutor. Answer in one short sentence.",
        llm=llm,
    )

    question = "What is the capital of France?"
    good = agent.run(question)
    print("== Real agent answer ==")
    print(f"  Q: {question}")
    print(f"  A: {good.output!r}")

    # A G-Eval judge: criteria + explicit steps + a 1-5 score-band rubric.
    judge = GEval(
        name="correctness",
        criteria="Is the answer factually correct and complete?",
        evaluation_steps=[
            "Identify the factual claim the answer makes.",
            "Compare it against the expected answer.",
            "Penalize fabricated, missing, or contradicted facts.",
        ],
        rubric=[(1, "Mostly incorrect"), (3, "Partially correct"), (5, "Fully correct")],
        scale="1-5",
        llm=llm,
    )

    print("\n== G-Eval (steps + rubric + chain-of-thought) ==")
    _show("correct answer", judge.score(input=question, output=good.output, expected="Paris"))
    _show(
        "wrong answer",
        judge.score(input=question, output="The capital of France is Berlin.", expected="Paris"),
    )

    # Auto-CoT: omit evaluation_steps and GEval derives them from the criteria.
    auto = GEval(
        name="helpfulness",
        criteria="Does the response directly and helpfully answer the user's question?",
        scale="1-5",
        llm=llm,
    )
    print("\n== G-Eval with auto-generated steps (Auto-CoT) ==")
    _show("helpful answer", auto.score(input=question, output=good.output))
    _show("evasive answer", auto.score(input=question, output="Geography is fascinating."))
    print(f"  (derived steps: {auto.evaluation_steps})")

    # The legacy single-call judge is unchanged.
    print("\n== Legacy LLMJudge (criteria only) ==")
    legacy = LLMJudge(criteria="correctness", llm=llm)
    _show("legacy correctness", legacy.score(input=question, output=good.output, expected="Paris"))


if __name__ == "__main__":
    main()
