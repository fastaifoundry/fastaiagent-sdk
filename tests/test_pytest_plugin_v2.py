"""Agent CI pytest-plugin v2: session aggregation + gates (pytester, no mocks).

Each test boots a real inner pytest session over generated test files, then
inspects the real ``local.db`` the inner session wrote.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _isolate_config():
    """pytester runs inner sessions in-process: the inner suite's
    FASTAIAGENT_LOCAL_DB + config reset must not leak into other tests."""
    from fastaiagent._internal.config import reset_config

    yield
    os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
    reset_config()


# Inner suites reset the lru_cached config AFTER pointing the env var at the
# per-test DB — get_config() may already be cached from earlier in-process runs.
_PASSING_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.testing import TestModel
from fastaiagent.agent import Agent
from fastaiagent.eval import case

@case(input="hello", expected="hi")
def test_a(evaluate_one):
    agent = Agent(name="g", llm=TestModel(response="hi"))
    evaluate_one(agent.run, scorers=["exact_match"])

@case(input="bye", expected="bye")
def test_b(evaluate_one):
    agent = Agent(name="g", llm=TestModel(response="bye"))
    evaluate_one(agent.run, scorers=["exact_match"])
"""

_MIXED_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.testing import TestModel
from fastaiagent.agent import Agent
from fastaiagent.eval import case

@case(input="hello", expected="hi")
def test_pass(evaluate_one):
    agent = Agent(name="g", llm=TestModel(response="hi"))
    evaluate_one(agent.run, scorers=["exact_match"])

@case(input="ping", expected="pong")
def test_fail(evaluate_one):
    agent = Agent(name="g", llm=TestModel(response="wrong"))
    evaluate_one(agent.run, scorers=["exact_match"], assert_pass=False)
"""

_INFRA_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.eval import case

def _broken(x):
    raise TimeoutError("provider 500")

@case(input="hello", expected="hi")
def test_infra(evaluate_one):
    evaluate_one(_broken, scorers=["exact_match"])
"""


def _runs(db_path) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT run_id, run_name, pass_count, fail_count FROM eval_runs"
        ).fetchall()


def test_session_persists_one_aggregated_run(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_PASSING_SUITE.format(db=db))
    result = pytester.runpytest("--no-header")
    result.assert_outcomes(passed=2)
    rows = _runs(db)
    assert len(rows) == 1, f"expected ONE aggregated run, got {rows}"
    _, run_name, pass_count, fail_count = rows[0]
    assert run_name.startswith("pytest::")
    assert (pass_count, fail_count) == (2, 0)


def test_eval_run_name_option(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_PASSING_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-run-name", "main")
    result.assert_outcomes(passed=2)
    assert _runs(db)[0][1] == "main"


def test_fail_under_gate_passes(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_PASSING_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-fail-under", "overall.pass_rate=0.9")
    assert result.ret == 0
    result.stdout.fnmatch_lines(["*fastaiagent eval*"])


def test_fail_under_gate_fails_session(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_MIXED_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-fail-under", "overall.pass_rate=0.9")
    # Both tests "pass" (assert_pass=False on the miss) but the aggregate
    # gate sees pass_rate 0.5 < 0.9 and fails the session.
    result.assert_outcomes(passed=2)
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*GATE: FAILED*"])


def test_no_gate_options_means_no_gating(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_MIXED_SUITE.format(db=db))
    result = pytester.runpytest("--no-header")
    result.assert_outcomes(passed=2)
    assert result.ret == 0


def test_infra_error_fails_test_and_marks_invalid(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_INFRA_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-max-error-rate", "0.1")
    # The test itself fails loudly (infra), and the run is INVALID, not green.
    result.assert_outcomes(failed=1)
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*errored (infra, not scored)*"])
    result.stdout.fnmatch_lines(["*GATE: INVALID*"])
    # The errored case is persisted with its error string.
    with sqlite3.connect(db) as conn:
        errors = conn.execute("SELECT error FROM eval_cases").fetchall()
    assert errors and "provider 500" in errors[0][0]


def test_baseline_regression_gate(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"

    # Session 1: perfect run, published as baseline "main".
    pytester.makepyfile(_PASSING_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-run-name", "main")
    assert result.ret == 0

    # Session 2: one case regresses; gate against baseline must fail.
    pytester.makepyfile(_MIXED_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-baseline", "main")
    result.assert_outcomes(passed=2)
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*GATE: REGRESSION*"])


def test_baseline_within_tolerance_passes(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_PASSING_SUITE.format(db=db))
    assert pytester.runpytest("--no-header", "--eval-run-name", "main").ret == 0

    pytester.makepyfile(_MIXED_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-baseline", "main", "--eval-tolerance", "0.6")
    assert result.ret == 0


def test_missing_baseline_fails_gate(pytester: pytest.Pytester, tmp_path) -> None:
    db = tmp_path / "local.db"
    pytester.makepyfile(_PASSING_SUITE.format(db=db))
    result = pytester.runpytest("--no-header", "--eval-baseline", "does-not-exist")
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*baseline comparison failed*"])


def test_save_as_test_lines_load_through_dataset(pytester: pytest.Pytester, tmp_path) -> None:
    """`ReplayResult.save_as_test()` JSONL rows drive `@dataset` directly."""
    from fastaiagent.trace.replay import ReplayResult

    cases = tmp_path / "regression.jsonl"
    ReplayResult(
        original_output="hi",
        new_output="hi",
        steps_executed=1,
        trace_id="t1",
    ).save_as_test(cases, input="hello", expected_output="hi", source_trace_id="t0")

    db = tmp_path / "local.db"
    pytester.makepyfile(
        f"""
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.testing import TestModel
from fastaiagent.agent import Agent
from fastaiagent.eval import pytest_dataset as dataset

@dataset(r"{cases}")
def test_regression(eval_case, evaluate_one):
    agent = Agent(name="g", llm=TestModel(response=eval_case["expected_output"]))
    evaluate_one(agent.run, scorers=["exact_match"])
        """
    )
    result = pytester.runpytest("--no-header")
    result.assert_outcomes(passed=1)
