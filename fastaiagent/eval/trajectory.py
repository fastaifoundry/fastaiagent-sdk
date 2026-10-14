"""Trajectory evaluation scorers.

Three states (1.89.0), so a trajectory scorer never reports a verdict it did not
reach. Before, every one returned **pass 1.0** when there was nothing to
compare, and ``evaluate()`` never passed the agent's tool calls at all, so
``tool_usage_accuracy`` read 1.00 on every case of every run:

* **no expectation** (``expected_trajectory`` / ``expected_tool_calls`` /
  ``expected_steps`` absent) → skipped: the check does not apply;
* **no record of what the agent did** (``actual_*`` absent, e.g. ``agent_fn``
  returned a plain string) → errored: the check could not run;
* **an observed empty list** (the agent called no tool) → a real verdict.

``evaluate()`` fills the ``actual_*`` keys from ``AgentResult.tool_calls``, and
a dataset row may carry the ``expected_*`` keys per case.
"""

from __future__ import annotations

from typing import Any

from fastaiagent.eval.scorer import Scorer, ScorerResult

#: Why an ``actual_*`` key is missing, and what to do about it.
NO_RECORD = (
    "agent_fn returned no tool-call record — return an AgentResult "
    "(agent.run / agent.arun) to score trajectories"
)


def _names(calls: list[Any]) -> list[str]:
    out: list[str] = []
    for c in calls:
        if isinstance(c, dict):
            out.append(str(c.get("name", "?")))
        else:
            out.append(str(c))
    return out


class ToolUsageAccuracy(Scorer):
    """Evaluates if the correct tools were used."""

    name = "tool_usage_accuracy"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        expected_traj = kw.get("expected_trajectory")
        if not expected_traj:
            return ScorerResult.not_applicable("No expected trajectory")
        if "actual_trajectory" not in kw:
            return ScorerResult.error(NO_RECORD)
        actual = list(kw["actual_trajectory"] or [])

        expected_tools = set(expected_traj)
        actual_tools = set(actual)
        correct = len(expected_tools & actual_tools)
        score = correct / max(len(expected_tools), 1)

        return ScorerResult(
            score=score,
            passed=score >= 0.5,
            reason=f"Correct tools: {correct}/{len(expected_tools)}",
            details={"expected": list(expected_traj), "actual": actual},
        )


class StepEfficiency(Scorer):
    """Evaluates how efficiently the agent solved the problem."""

    name = "step_efficiency"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if kw.get("expected_steps") is None:
            return ScorerResult.not_applicable("No expected step count")
        if "actual_steps" not in kw:
            return ScorerResult.error(NO_RECORD)
        actual_steps = int(kw["actual_steps"] or 0)
        expected_steps = int(kw["expected_steps"])
        if actual_steps == 0:
            # Took no step at all: nothing was wasted.
            return ScorerResult(
                score=1.0,
                passed=True,
                reason=f"Steps: 0 (expected: {expected_steps})",
                details={"expected": expected_steps, "actual": 0},
            )

        score = min(expected_steps / actual_steps, 1.0)
        return ScorerResult(
            score=score,
            passed=score >= 0.5,
            reason=f"Steps: {actual_steps} (expected: {expected_steps})",
            details={"expected": expected_steps, "actual": actual_steps},
        )


class PathCorrectness(Scorer):
    """Evaluates if the agent followed the correct path using LCS."""

    name = "path_correctness"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        expected_traj = kw.get("expected_trajectory")
        if not expected_traj:
            return ScorerResult.not_applicable("No expected trajectory")
        if "actual_trajectory" not in kw:
            return ScorerResult.error(NO_RECORD)
        actual = list(kw["actual_trajectory"] or [])

        lcs_len = self._lcs_length(actual, list(expected_traj))
        score = lcs_len / max(len(expected_traj), 1)
        return ScorerResult(
            score=score,
            passed=score >= 0.5,
            reason=f"Path LCS: {lcs_len}/{len(expected_traj)}",
            details={"expected": list(expected_traj), "actual": actual},
        )

    @staticmethod
    def _lcs_length(a: list[Any], b: list[Any]) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]


class CycleEfficiency(Scorer):
    """Evaluates whether unnecessary cycles occurred."""

    name = "cycle_efficiency"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        if "actual_trajectory" not in kw:
            return ScorerResult.error(NO_RECORD)
        actual = list(kw["actual_trajectory"] or [])
        if not actual:
            return ScorerResult(score=1.0, passed=True, reason="No tool calls: no cycles")

        # Count repeated consecutive tool calls
        cycles = 0
        for i in range(1, len(actual)):
            if actual[i] == actual[i - 1]:
                cycles += 1

        score = 1.0 - (cycles / max(len(actual), 1))
        return ScorerResult(
            score=max(score, 0.0),
            passed=score >= 0.5,
            reason=f"Cycles: {cycles}/{len(actual)}",
            details={"actual": actual},
        )


class ToolCallCorrectness(Scorer):
    """Validates tool calls by name AND arguments (deep equality).

    Stricter than ``ToolUsageAccuracy`` which only checks tool names as sets.
    This scorer verifies that each expected tool call has a matching actual call
    with the same name and identical arguments.

    Pass via kwargs:
        - ``actual_tool_calls``: list of dicts ``[{"name": "...", "arguments": {...}}, ...]``
        - ``expected_tool_calls``: list of dicts with the same structure

    Example:
        scorer = ToolCallCorrectness()
        result = scorer.score(
            input="", output="",
            actual_tool_calls=[{"name": "search", "arguments": {"query": "Paris"}}],
            expected_tool_calls=[{"name": "search", "arguments": {"query": "Paris"}}],
        )
        # score=1.0, passed=True
    """

    name = "tool_call_correctness"

    def score(
        self, input: str, output: str, expected: str | None = None, **kw: Any
    ) -> ScorerResult:
        expected_calls = kw.get("expected_tool_calls")
        if not expected_calls:
            return ScorerResult.not_applicable("No expected tool calls")
        if "actual_tool_calls" not in kw:
            return ScorerResult.error(NO_RECORD)
        actual = list(kw["actual_tool_calls"] or [])

        # Greedy matching: for each expected call, find a matching actual call
        remaining = list(actual)
        matched = 0
        for exp in expected_calls:
            for i, act in enumerate(remaining):
                if (
                    act.get("name") == exp.get("name")
                    and act.get("arguments") == exp.get("arguments")
                ):
                    matched += 1
                    remaining.pop(i)
                    break

        score = matched / len(expected_calls)
        # Tool names only in details: arguments can carry case content.
        return ScorerResult(
            score=round(score, 4),
            passed=score >= 0.5,
            reason=f"Correct calls: {matched}/{len(expected_calls)} (name+args match)",
            details={"expected": _names(expected_calls), "actual": _names(actual)},
        )
