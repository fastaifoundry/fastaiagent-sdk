"""Live e2e: Agent CI gate over a real LLM (marked ``e2e``; needs OPENAI_API_KEY).

One small dataset, a real Agent on gpt-4o-mini, gated via the same
``gate()`` the pytest plugin and CLI use — proving the full loop
(evaluate → persist → gate → compare) against a live provider.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from fastaiagent.eval.results import EvalResults

pytestmark = pytest.mark.e2e

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

DATASET = [
    {"input": "Reply with exactly the word: ping", "expected_output": "ping"},
    {"input": "Reply with exactly the word: pong", "expected_output": "pong"},
]

# One transient provider error (429/5xx/timeout) on either of the two
# concurrent calls puts error_rate at exactly the 0.5 gate ceiling and fails
# the build. Retry rather than widen the threshold: ``errored == 0`` is the
# assertion worth keeping — it is what proves the evaluate→gate loop ran
# clean — and a genuine break fails every attempt, so retrying costs only
# flakiness, not signal.
_MAX_EVAL_ATTEMPTS = 3


def _evaluate_clean(fn: Any, dataset: list[dict[str, Any]]) -> EvalResults:
    """Run the eval, retrying while any case errors. Returns the last attempt."""
    from fastaiagent.eval import evaluate

    results = None
    for attempt in range(1, _MAX_EVAL_ATTEMPTS + 1):
        results = evaluate(fn, dataset, scorers=["contains"], persist=False, concurrency=2)
        if results.errored_count == 0:
            return results
        print(
            f"[attempt {attempt}/{_MAX_EVAL_ATTEMPTS}] "
            f"{results.errored_count}/{len(results.cases)} cases errored; "
            f"{_case_errors(results)}"
        )
    assert results is not None
    return results


def _case_errors(results: EvalResults) -> str:
    """Per-case error text — ``evaluate`` records exceptions instead of raising,
    so without this a failure reports only a count and is undiagnosable in CI."""
    errored = [f"case[{i}]: {c.error}" for i, c in enumerate(results.cases) if c.error]
    return "; ".join(errored) or "<no per-case error recorded>"


@requires_openai
def test_agent_ci_gate_live(tmp_path) -> None:
    from fastaiagent.agent import Agent
    from fastaiagent.eval import compare_runs
    from fastaiagent.eval.gate import gate
    from fastaiagent.llm import LLMClient

    agent = Agent(
        name="ci-e2e",
        llm=LLMClient(provider="openai", model="gpt-4o-mini", temperature=0),
        system_prompt="Follow the instruction literally. Output only the requested word.",
    )

    results = _evaluate_clean(agent.run, DATASET)
    baseline_id = results.persist_local(db_path=tmp_path / "local.db", run_name="main")

    report = gate(results, fail_under=["contains.pass_rate=0.5"], max_error_rate=0.5)
    assert report.outcome == "passed", report.describe()
    assert report.errored == 0, (
        f"{_MAX_EVAL_ATTEMPTS} attempts all errored — not a transient blip.\n"
        f"{report.describe()}\n{_case_errors(results)}"
    )

    # Second run compares clean against the first (same agent, same data).
    results2 = _evaluate_clean(agent.run, DATASET)
    results2.persist_local(db_path=tmp_path / "local.db", run_name="pr")
    comparison = compare_runs("main", "pr", db_path=tmp_path / "local.db")
    assert comparison.run_a["run_id"] == baseline_id
    assert not comparison.has_regressions or comparison.pass_rate_delta >= -0.5
