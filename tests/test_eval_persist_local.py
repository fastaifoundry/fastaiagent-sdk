"""Tests for EvalResults.persist_local + aevaluate() auto-persistence.

Hits real SQLite via temp paths. No mocks for the storage layer.
The agent function is a plain Python callable — not an LLM — so the
test is deterministic and hermetic.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.eval.evaluate import aevaluate, evaluate
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import ScorerResult


def _uppercase_agent(text: str) -> str:
    return text.upper()


def _wrong_agent(text: str) -> str:
    return "wrong"


class TestPersistLocal:
    def test_writes_run_and_cases(self, temp_dir):
        results = EvalResults()
        results.add("exact_match", ScorerResult(score=1.0, passed=True))
        results.add("exact_match", ScorerResult(score=0.0, passed=False))
        results.add_case(
            EvalCaseRecord(
                input="q1",
                expected_output="A",
                actual_output="A",
                trace_id="t1",
                per_scorer={"exact_match": {"passed": True, "score": 1.0, "reason": None}},
            )
        )
        results.add_case(
            EvalCaseRecord(
                input="q2",
                expected_output="B",
                actual_output="b",
                trace_id="t2",
                per_scorer={"exact_match": {"passed": False, "score": 0.0, "reason": None}},
            )
        )

        db_path = temp_dir / "local.db"
        run_id = results.persist_local(
            db_path=db_path,
            run_name="smoke",
            dataset_name="cases.jsonl",
            agent_name="my-agent",
        )
        assert run_id

        with SQLiteHelper(db_path) as db:
            run_row = db.fetchone(
                "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
            )
            case_rows = db.fetchall(
                "SELECT * FROM eval_cases WHERE run_id = ? ORDER BY ordinal",
                (run_id,),
            )

        assert run_row is not None
        assert run_row["run_name"] == "smoke"
        assert run_row["dataset_name"] == "cases.jsonl"
        assert run_row["agent_name"] == "my-agent"
        assert run_row["pass_count"] == 1
        assert run_row["fail_count"] == 1
        assert run_row["pass_rate"] == 0.5

        assert len(case_rows) == 2
        assert case_rows[0]["trace_id"] == "t1"
        assert case_rows[1]["trace_id"] == "t2"


class TestAEvaluateAutoPersists:
    @pytest.fixture(autouse=True)
    def _local_db(self, monkeypatch, temp_dir):
        monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(temp_dir / "local.db"))
        reset_config()
        yield
        reset_config()

    def test_auto_persist_default(self, temp_dir):
        dataset = [
            {"input": "hello", "expected_output": "HELLO"},
            {"input": "world", "expected_output": "WORLD"},
        ]
        asyncio.run(
            aevaluate(
                _uppercase_agent,
                dataset,
                scorers=["exact_match"],
                agent_name="uppercaser",
                run_name="unit",
            )
        )

        with SQLiteHelper(temp_dir / "local.db") as db:
            runs = db.fetchall("SELECT * FROM eval_runs")
            cases = db.fetchall("SELECT * FROM eval_cases ORDER BY ordinal")
        assert len(runs) == 1
        run = runs[0]
        assert run["run_name"] == "unit"
        assert run["agent_name"] == "uppercaser"
        assert run["pass_count"] == 2
        assert run["fail_count"] == 0
        assert len(cases) == 2
        assert cases[0]["input"].strip('"') == "hello"

    def test_persist_false_skips_write(self, temp_dir):
        dataset = [{"input": "hello", "expected_output": "HELLO"}]
        asyncio.run(
            aevaluate(
                _uppercase_agent,
                dataset,
                scorers=["exact_match"],
                persist=False,
            )
        )

        # Either the DB wasn't created, or the eval_runs table is empty.
        db_file = temp_dir / "local.db"
        if db_file.exists():
            with SQLiteHelper(db_file) as db:
                rows = db.fetchall(
                    "SELECT name FROM sqlite_master WHERE name='eval_runs'"
                )
                if rows:
                    count = db.fetchall("SELECT COUNT(*) AS n FROM eval_runs")
                    assert count[0]["n"] == 0

    def test_failed_case_recorded(self, temp_dir):
        dataset = [{"input": "hello", "expected_output": "HELLO"}]
        asyncio.run(
            aevaluate(
                _wrong_agent,
                dataset,
                scorers=["exact_match"],
            )
        )
        with SQLiteHelper(temp_dir / "local.db") as db:
            runs = db.fetchall("SELECT * FROM eval_runs")
            cases = db.fetchall("SELECT * FROM eval_cases")
        assert runs[0]["fail_count"] == 1
        assert cases[0]["actual_output"].strip('"') == "wrong"


class TestSyncEvaluateAutoPersists:
    def test_sync_path_persists_too(self, monkeypatch, temp_dir):
        monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(temp_dir / "local.db"))
        reset_config()
        try:
            evaluate(
                _uppercase_agent,
                [{"input": "a", "expected_output": "A"}],
                scorers=["exact_match"],
                run_name="sync-run",
            )
        finally:
            reset_config()

        with SQLiteHelper(temp_dir / "local.db") as db:
            rows = db.fetchall("SELECT run_name FROM eval_runs")
        assert any(r["run_name"] == "sync-run" for r in rows)
