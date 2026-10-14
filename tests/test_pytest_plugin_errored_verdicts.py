"""The pytest plugin records a case whose scorer could not run (1.89.0).

Before, a judge that raised escaped ``evaluate_one``: the inner test errored and
the case never reached the session run, so a 4-case suite persisted a 3-case
run and its pass rate read as if the outage case had not existed. Now the case
is recorded as errored, the test fails with a message saying the check could
not run (not that the agent was wrong), and a gate on the judged metric is
**invalid**. A case nothing applied to also fails instead of passing.

Real inner pytest sessions (pytester), real ``local.db``; no mocks.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _isolate_config():  # noqa: ANN202
    from fastaiagent._internal.config import reset_config

    yield
    os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
    reset_config()


_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.agent import Agent
from fastaiagent.eval import case
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.testing import FunctionModel, TestModel

def _down(messages):
    raise RuntimeError("judge provider down, key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")

@case(input="a", expected="a")
def test_one(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="a")).run, scorers=["exact_match"])

@case(input="b", expected="b")
def test_two(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="b")).run, scorers=["exact_match"])

@case(input="c", expected="c")
def test_three(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="c")).run, scorers=["exact_match"])

@case(input="d", expected="d")
def test_judge_outage(evaluate_one):
    judge = LLMJudge(name="exact_match", llm=FunctionModel(_down))
    evaluate_one(Agent(name="g", llm=TestModel(response="d")).run, scorers=[judge])

@case(input="e")
def test_nothing_applies(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="e")).run, scorers=["exact_match"])
"""


def _run(db_path) -> dict:  # noqa: ANN001
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM eval_runs").fetchall()
        assert len(rows) == 1, rows
        return dict(rows[0])
    finally:
        conn.close()


def test_errored_case_is_recorded_and_fails_the_test(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "-rf")
    result.assert_outcomes(passed=3, failed=2)
    out = result.stdout.str()
    assert "scorer could not run, or none applied" in out
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456" not in out, "the key in the judge error leaked"

    run = _run(db)
    assert run["case_count"] == 5, "the outage case went missing from the run"
    assert (run["case_pass_count"], run["case_fail_count"], run["case_error_count"]) == (3, 0, 2)
    assert run["check_error_count"] == 1 and run["check_skip_count"] == 1
    assert (run["pass_count"], run["fail_count"]) == (3, 0)
    assert run["kind"] == "pytest"

    conn = sqlite3.connect(db)
    try:
        stored = [r[0] for r in conn.execute("SELECT per_scorer FROM eval_cases")]
    finally:
        conn.close()
    assert not any("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456" in s for s in stored), (
        "the key in the judge error reached local.db"
    )


def test_a_gate_on_the_outage_metric_is_invalid(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-fail-under", "exact_match.pass_rate=0.5")
    assert result.ret != 0
    out = result.stdout.str()
    assert "GATE: INVALID" in out and "GATE: FAILED" not in out, out
    meta = _run(db)["metadata"]
    assert '"gate_outcome": "invalid"' in meta
