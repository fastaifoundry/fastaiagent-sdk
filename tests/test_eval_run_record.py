"""What an eval run records about itself (1.89.0).

* ``agent_version`` was always NULL. It is now a fingerprint of the agent's
  definition (D19): the same agent gives the same value, a changed prompt a
  different one, and a plain function ``None`` rather than a made-up value.
* ``started_at`` and ``finished_at`` were both the save time; they are the
  run's own times now.
* ``kind`` says what produced the run.
* ``scorer_config`` (each scorer's ``describe()``) lands in the run's metadata,
  never a key.
* ``EvalResults.compare()`` averaged in errored/skipped placeholders.

Real ``evaluate()``, real ``Agent``, real SQLite; no mocks.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from fastaiagent.agent import Agent
from fastaiagent.eval import evaluate
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.results import EvalResults
from fastaiagent.eval.scorer import ScorerResult
from fastaiagent.llm import LLMClient
from fastaiagent.testing import TestModel


def _row(db: Path, run_id: str) -> dict:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)).fetchone())


def test_agent_version_is_a_fingerprint() -> None:
    def make(system_prompt: str) -> Agent:
        return Agent(name="fp", system_prompt=system_prompt, llm=TestModel(response="x"))

    data = [{"input": "q", "expected": "x"}]
    v1 = evaluate(make("Be brief.").run, data, persist=False).agent_version
    v1_again = evaluate(make("Be brief.").run, data, persist=False).agent_version
    v2 = evaluate(make("Be thorough.").run, data, persist=False).agent_version
    assert v1 and v1.startswith("sha256:") and len(v1) == len("sha256:") + 16
    assert v1 == v1_again
    assert v1 != v2
    assert evaluate(lambda q: "x", data, persist=False).agent_version is None


def test_run_times_kind_and_version_are_persisted(isolated_local_db: Path) -> None:
    agent = Agent(name="timed", llm=TestModel(response="x", delay_ms=30))
    results = evaluate(agent.run, [{"input": "q", "expected": "x"}], concurrency=1)
    row = _row(isolated_local_db, results.run_id)
    assert row["kind"] == "evaluate"
    assert row["agent_version"] == results.agent_version
    assert row["started_at"] < row["finished_at"], "both were the save time"


def test_scorer_config_lands_in_metadata_without_a_key(isolated_local_db: Path) -> None:
    secret = "sk-proj-SHOULDNEVERBESTORED0123456789"
    judge = LLMJudge(
        name="tone",
        criteria="polite",
        llm=LLMClient(provider="openai", model="gpt-4o-mini", api_key=secret),
    )
    results = evaluate(lambda q: "x", [{"input": "q"}], scorers=["exact_match"], persist=False)
    results.scorer_config["tone"] = judge.describe()
    run_id = results.persist_local()
    meta = json.loads(_row(isolated_local_db, run_id)["metadata"])
    cfg = meta["scorer_config"]
    assert cfg["exact_match"] == {"type": "ExactMatch"}
    assert cfg["tone"]["model"] == "gpt-4o-mini" and cfg["tone"]["criteria"] == "polite"
    assert secret not in json.dumps(meta)


def test_results_compare_ignores_placeholders() -> None:
    a = EvalResults()
    a.add("j", ScorerResult(score=0.8, passed=True))
    b = EvalResults()
    b.add("j", ScorerResult(score=0.8, passed=True))
    b.add("j", ScorerResult.error("judge down"))
    assert "0.80 → 0.80" in a.compare(b), "an outage read as a score drop"


def test_started_at_precedes_the_agent_calls() -> None:
    before = time.time()
    results = evaluate(lambda q: "x", [{"input": "q"}], persist=False)
    from datetime import datetime

    assert results.started_at and results.finished_at
    assert datetime.fromisoformat(results.started_at).timestamp() >= before - 1
    assert results.started_at <= results.finished_at
