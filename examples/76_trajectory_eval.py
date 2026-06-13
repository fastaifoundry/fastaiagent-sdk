"""Example 76: Trajectory scorers on a REAL agent run.

Demonstrates (real Agent + real LLM + real tools — needs OPENAI_API_KEY):
- ToolUsageAccuracy  : did the agent use the right set of tools?
- StepEfficiency     : did it solve the task in the expected number of steps?
- PathCorrectness    : did it follow the right order (LCS-based)?
- CycleEfficiency    : did it avoid repeated consecutive tool calls?
- ToolCallCorrectness: did each call have the right name AND arguments?

The agent really decides which tools to call; we read the actual sequence off
`AgentResult.tool_calls` (each entry is `{"tool_name", "arguments", ...}`) and
score it against the expected trajectory. Scores reflect the real run.

Run:
    zsh -lc 'python examples/76_trajectory_eval.py'
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, LLMClient, tool
from fastaiagent.eval import (
    CycleEfficiency,
    PathCorrectness,
    StepEfficiency,
    ToolCallCorrectness,
    ToolUsageAccuracy,
)


@tool(description="Get the recommended restaurant tip percentage for a country.")
def tip_rate(country: str) -> str:
    rates = {"us": "20", "usa": "20", "united states": "20", "france": "10"}
    return rates.get(country.strip().lower(), "15")


@tool(description="Compute a tip amount given a bill total and a tip percentage.")
def compute_tip(bill: float, percent: float) -> str:
    return f"{bill * percent / 100:.2f}"


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    agent = Agent(
        name="tip-helper",
        system_prompt=(
            "You help compute restaurant tips. First look up the tip rate for the "
            "country with the tip_rate tool, then compute the tip with compute_tip. "
            "Use the tools rather than doing the math yourself."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[tip_rate, compute_tip],
    )

    result = agent.run("I had a $200 dinner in the United States. What tip should I leave?")

    # Real trajectory captured from the run.
    actual = [tc["tool_name"] for tc in result.tool_calls]
    actual_calls = [
        {"name": tc["tool_name"], "arguments": tc["arguments"]} for tc in result.tool_calls
    ]
    expected = ["tip_rate", "compute_tip"]

    print(f"output:   {result.output}")
    print(f"expected: {expected}")
    print(f"actual:   {actual}\n")

    checks = [
        (
            "tool_usage_accuracy",
            ToolUsageAccuracy().score(
                input="", output="", actual_trajectory=actual, expected_trajectory=expected
            ),
        ),
        (
            "path_correctness",
            PathCorrectness().score(
                input="", output="", actual_trajectory=actual, expected_trajectory=expected
            ),
        ),
        (
            "cycle_efficiency",
            CycleEfficiency().score(input="", output="", actual_trajectory=actual),
        ),
        (
            "step_efficiency",
            StepEfficiency().score(
                input="", output="", actual_steps=len(actual), expected_steps=len(expected)
            ),
        ),
        (
            "tool_call_correctness",
            ToolCallCorrectness().score(
                input="",
                output="",
                actual_tool_calls=actual_calls,
                expected_tool_calls=[
                    {"name": "tip_rate", "arguments": {"country": "United States"}},
                    {"name": "compute_tip", "arguments": {"bill": 200, "percent": 20}},
                ],
            ),
        ),
    ]

    for name, r in checks:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {name:<22} score={r.score:.2f} — {r.reason}")


if __name__ == "__main__":
    main()
