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
