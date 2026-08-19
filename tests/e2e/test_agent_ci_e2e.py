"""Live e2e: Agent CI gate over a real LLM (marked ``e2e``; needs OPENAI_API_KEY).

One small dataset, a real Agent on gpt-4o-mini, gated via the same
``gate()`` the pytest plugin and CLI use — proving the full loop
(evaluate → persist → gate → compare) against a live provider.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.e2e

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

DATASET = [
    {"input": "Reply with exactly the word: ping", "expected_output": "ping"},
    {"input": "Reply with exactly the word: pong", "expected_output": "pong"},
]


@requires_openai
def test_agent_ci_gate_live(tmp_path) -> None:
    from fastaiagent.agent import Agent
    from fastaiagent.eval import compare_runs, evaluate
    from fastaiagent.eval.gate import gate
    from fastaiagent.llm import LLMClient

    agent = Agent(
        name="ci-e2e",
        llm=LLMClient(provider="openai", model="gpt-4o-mini", temperature=0),
        system_prompt="Follow the instruction literally. Output only the requested word.",
    )

    results = evaluate(agent.run, DATASET, scorers=["contains"], persist=False, concurrency=2)
    baseline_id = results.persist_local(db_path=tmp_path / "local.db", run_name="main")

    report = gate(results, fail_under=["contains.pass_rate=0.5"], max_error_rate=0.5)
    assert report.outcome == "passed", report.describe()
    assert report.errored == 0

    # Second run compares clean against the first (same agent, same data).
    results2 = evaluate(agent.run, DATASET, scorers=["contains"], persist=False, concurrency=2)
    results2.persist_local(db_path=tmp_path / "local.db", run_name="pr")
    comparison = compare_runs("main", "pr", db_path=tmp_path / "local.db")
    assert comparison.run_a["run_id"] == baseline_id
    assert not comparison.has_regressions or comparison.pass_rate_delta >= -0.5
