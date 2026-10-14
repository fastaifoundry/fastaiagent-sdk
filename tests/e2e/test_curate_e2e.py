"""Trace -> dataset curation — real-LLM end-to-end (no mocking).

Runs a real Agent against a live provider so real ``agent.input``/``agent.output``
spans are captured, then curates them and evaluates the curated set. This is the
only test that proves the trace->dataset contract against the real Agent code
path (the unit tests assert the curation logic given that contract).

    zsh -lc 'pytest tests/e2e/test_curate_e2e.py -q'
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import Dataset, evaluate
from fastaiagent.trace import otel

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]


def test_curate_real_agent_then_evaluate(isolated_local_db: Any, tmp_path: Any) -> None:
    otel.reset()  # rebuild the tracer against the temp DB from isolated_local_db
    try:
        agent = Agent(
            name="e2e-curate",
            system_prompt="You are a concise FAQ bot. Answer in one short sentence.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        )
        question = "What is the capital of France?"
        run = agent.run(question)
        assert "paris" in run.output.lower()

        # Curate the real trace we just produced.
        ds = Dataset.from_traces(filter="all", agent="e2e-curate")
        matches = [it for it in ds if it["input"] == question]
        assert matches, "curated dataset should contain the agent run we just did"
        item = matches[0]
        assert item["expected_output"] == run.output  # captured output is the gold answer
        assert not item.get("needs_review")
        assert item["trace_id"] == run.trace_id

        # Round-trip to JSONL and re-evaluate with a real scorer.
        out = tmp_path / "curated.jsonl"
        ds.to_jsonl(out)
        results = evaluate(
            agent_fn=agent.run,
            dataset=str(out),
            scorers=["answer_relevancy"],
            persist=False,
        )
        scores = results.scores["answer_relevancy"]
        assert scores and all(0.0 <= s.score <= 1.0 for s in scores)
    finally:
        otel.reset()


@pytest.mark.parametrize("record", ["rows", "legacy"])
def test_curate_guardrail_filter_live(
    isolated_local_db: Any, monkeypatch: pytest.MonkeyPatch, record: str
) -> None:
    """G8 (1.89.0): a real warn-rule firing is curated, from its event row, and
    from its guardrail span alone (a run recorded before 1.89.0 has no row).
    Before, the filter matched an outcome nothing writes and returned nothing
    either way."""
    import sqlite3

    from fastaiagent._internal.config import get_config, reset_config
    from fastaiagent.eval import curate_from_traces
    from fastaiagent.guardrail.builtins import no_pii

    reset_config()
    otel.reset()
    try:
        rule = no_pii()
        rule.action = "warn"  # record and carry on: the run still finishes
        agent = Agent(
            name="e2e-pii",
            system_prompt="Repeat the user's message back exactly, nothing else.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
            guardrails=[rule],
        )
        run = agent.run("Repeat exactly: my email is a@b.com")
        assert "a@b.com" in run.output, "the model did not echo the email; rerun"
        if record == "legacy":
            with sqlite3.connect(get_config().local_db_path) as conn:
                conn.execute("DELETE FROM guardrail_events")

        items = curate_from_traces(filter="guardrail", limit=None)
        mine = [it for it in items if it["trace_id"] == run.trace_id]
        assert mine, "the run where no_pii fired was not curated"
        assert "guardrail 'no_pii' fired" in mine[0]["reason"]
        assert mine[0]["needs_review"] is True
    finally:
        otel.reset()
