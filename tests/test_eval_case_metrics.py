"""Cases first, checks second: the 1.89.0 eval numbers on a candidate-C-shaped run.

WS4 §1.6. The audit's run 52cfb6b7 had 5 cases; case 4's agent raised (BOOM);
4 cases × 6 scorers = 24 verdicts, and ``tool_usage_accuracy`` said "No expected
trajectory" on all 4. 1.88.0 showed "Pass rate 83%, Passed 20, Failed 4" for a
run in which **one case of five** passed. After: cases 1 / 3 / 1 (25%), checks
16 / 20 (80%), 4 n/a.

Built from real ``ScorerResult`` objects, persisted through the real
``persist_local`` into a real SQLite file, read back through the route helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastaiagent._internal.errors import EvalError
from fastaiagent.eval import Scorecard
from fastaiagent.eval.compare import case_outcome, case_outcome_detail, summarize_cases
from fastaiagent.eval.gate import gate
from fastaiagent.eval.results import SUMMARY_VERSION, EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import ScorerResult

P = ScorerResult(score=1.0, passed=True)
F = ScorerResult(score=0.0, passed=False, reason="mismatch")
NA = ScorerResult.not_applicable("No expected trajectory")


def _geval(score: float) -> ScorerResult:
    return ScorerResult(score=score, passed=score >= 0.5)


def candidate_c() -> EvalResults:
    # Per scored case: exact_match, contains, geval_correctness, latency,
    # cost_under, tool_usage_accuracy. Fails: exact_match on cases 2, 3, 5;
    # contains on case 5. geval scores average 0.875.
    rows = [
        ("q1", [P, P, _geval(1.0), P, P, NA], None),
        ("q2", [F, P, _geval(0.75), P, P, NA], None),
        ("q3", [F, P, _geval(1.0), P, P, NA], None),
        ("q4", None, "RuntimeError: BOOM"),
        ("q5", [F, F, _geval(0.75), P, P, NA], None),
    ]
    names = [
        "exact_match",
        "contains",
        "geval_correctness",
        "latency",
        "cost_under",
        "tool_usage_accuracy",
    ]
    er = EvalResults()
    for q, verdicts, error in rows:
        per: dict[str, dict[str, object]] = {}
        if verdicts is not None:
            for name, v in zip(names, verdicts, strict=True):
                er.add(name, v)
                per[name] = v.stored()
        er.add_case(EvalCaseRecord(input=q, expected_output="x", per_scorer=per, error=error))
    return er


def test_case_and_check_counts() -> None:
    er = candidate_c()
    cc = er.case_counts
    assert (cc["case_pass_count"], cc["case_fail_count"], cc["case_error_count"]) == (1, 3, 1)
    assert cc["case_count"] == 5
    assert cc["case_pass_rate"] == 0.25
    assert er.check_counts == {"passed": 16, "failed": 4, "errored": 0, "skipped": 4}


def test_scorecard_cases_first_checks_second() -> None:
    sc = Scorecard.from_eval_results(candidate_c())
    assert sc.overall_pass_rate == 0.8, "checks: 16 / 20, the n/a verdicts excluded"
    assert sc.cases == {"count": 5, "passed": 1, "failed": 3, "errored": 1, "pass_rate": 0.25}
    assert sc.checks_skipped == 4
    tua = next(m for m in sc.metrics if m.name == "tool_usage_accuracy")
    assert (tua.n, tua.skipped, tua.pass_rate, tua.avg_score) == (0, 4, None, None)
    text = sc.summary()
    assert "not evaluable" in text and "cases: 1 passed, 3 failed, 1 errored" in text


@pytest.mark.parametrize(
    ("spec", "kw", "outcome"),
    [
        (["overall.pass_rate=0.8"], {}, "passed"),
        (["geval_correctness.avg_score=0.7"], {}, "passed"),
        (["tool_usage_accuracy.pass_rate=0.9"], {}, "invalid"),
        (["cases.pass_rate=0.8"], {}, "failed"),
        (["overall.pass_rate=0.8"], {"max_error_rate": 0.1}, "invalid"),
        (["typo_scorer.pass_rate=0.5"], {}, "failed"),
    ],
    ids=["overall", "geval", "tua-n0", "cases", "max-error-rate", "typo"],
)
def test_gate_rows_of_the_worked_example(spec: list[str], kw: dict, outcome: str) -> None:
    report = gate(candidate_c(), fail_under=spec, **kw)
    assert report.outcome == outcome, report.describe()


def test_overall_avg_score_is_not_a_metric() -> None:
    with pytest.raises(EvalError, match="overall.avg_score"):
        gate(candidate_c(), fail_under=["overall.avg_score=0.5"])


def test_a_judge_outage_is_invalid_not_failed() -> None:
    er = EvalResults()
    for q in ("a", "b"):
        r = ScorerResult.error("Judge error: provider down")
        er.add("llm_judge", r)
        er.add_case(EvalCaseRecord(input=q, per_scorer={"llm_judge": r.stored()}))
    report = gate(er, fail_under=["llm_judge.pass_rate=0.5"])
    assert report.outcome == "invalid"
    assert any("could not run" in r for r in report.invalid_reasons)
    # Tolerated outage: max_error_rate allows it, the metric still has no verdict.
    assert gate(er, fail_under=["llm_judge.pass_rate=0.5"], max_error_rate=1.0).outcome == (
        "invalid"
    )


@pytest.mark.parametrize(
    ("exc", "want"),
    [(TimeoutError(), "TimeoutError"), (RuntimeError("provider 500"), "provider 500")],
)
def test_an_errored_case_always_says_why(exc: BaseException, want: str) -> None:
    """1.95.0: an exception with no message (a bare TimeoutError, an
    httpx.ConnectError) was recorded as ``""``, so the errored case had no
    reason in the summary, the UI or the persisted run."""
    from fastaiagent.eval import Dataset, evaluate

    def agent_fn(_: str) -> str:
        raise exc

    results = evaluate(
        agent_fn=agent_fn,
        dataset=Dataset.from_list([{"input": "q", "expected": "a"}]),
        scorers=["exact_match"],
        persist=False,
    )
    assert results.cases[0].error == want


def test_partial_outage_within_max_error_rate_is_gated() -> None:
    er = EvalResults()
    verdicts = [P, P, P, ScorerResult.error("Judge error: 503")]
    for i, r in enumerate(verdicts):
        er.add("llm_judge", r)
        er.add_case(EvalCaseRecord(input=str(i), per_scorer={"llm_judge": r.stored()}))
    assert gate(er, fail_under=["llm_judge.pass_rate=0.9"]).outcome == "invalid"
    report = gate(er, fail_under=["llm_judge.pass_rate=0.9"], max_error_rate=0.3)
    assert report.outcome == "passed", report.describe()


@pytest.mark.parametrize(
    ("case", "want"),
    [
        ({"error": "BOOM", "per_scorer": {}}, ("errored", "agent")),
        ({"per_scorer": {"a": {"passed": False}, "b": {"errored": True}}}, ("failed", None)),
        ({"per_scorer": {"a": {"passed": True}, "b": {"errored": True}}}, ("errored", "scorer")),
        ({"per_scorer": {"a": {"skipped": True}}}, ("errored", "unscored")),
        ({"per_scorer": {}}, ("errored", "unscored")),
        ({"per_scorer": {"a": {"passed": True}, "b": {"skipped": True}}}, ("passed", None)),
    ],
    ids=["agent", "fail-wins", "scorer", "all-skipped", "empty", "pass"],
)
def test_case_outcome_rule(case: dict, want: tuple[str, str | None]) -> None:
    assert case_outcome_detail(case) == want
    assert case_outcome(case) == want[0]


def test_nothing_applied_has_no_rate() -> None:
    """D13: a run where no verdict applied has no pass rate (it read 0.0)."""
    s = summarize_cases([{"per_scorer": {"a": {"skipped": True}}}])
    assert s["case_pass_rate"] is None and s["check_pass_rate"] is None


def test_persisted_columns(isolated_local_db: Path) -> None:
    from fastaiagent._internal.storage import SQLiteHelper

    er = candidate_c()
    run_id = er.persist_local(run_name="cand-c", kind="evaluate")
    with SQLiteHelper(isolated_local_db) as db:
        row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        cases = db.fetchall("SELECT * FROM eval_cases WHERE run_id = ?", (run_id,))
    assert row is not None
    assert (row["pass_count"], row["fail_count"], row["pass_rate"]) == (16, 4, 0.8)
    assert (row["case_count"], row["case_pass_count"], row["case_fail_count"]) == (5, 1, 3)
    assert (row["case_error_count"], row["case_pass_rate"]) == (1, 0.25)
    assert (row["check_error_count"], row["check_skip_count"]) == (0, 4)
    assert row["summary_version"] == SUMMARY_VERSION and row["kind"] == "evaluate"
    assert len(cases) == 5
    stored = json.loads(cases[0]["per_scorer"])
    assert stored["tool_usage_accuracy"]["skipped"] is True
    assert "errored" not in stored["exact_match"], "defaults are not written"


def test_nothing_rated_persists_null_pass_rate(isolated_local_db: Path) -> None:
    from fastaiagent._internal.storage import SQLiteHelper

    er = EvalResults()
    er.add("exact_match", NA)
    er.add_case(EvalCaseRecord(input="q", per_scorer={"exact_match": NA.stored()}))
    run_id = er.persist_local()
    with SQLiteHelper(isolated_local_db) as db:
        row = db.fetchone("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
    assert row is not None and row["pass_rate"] is None and row["case_pass_rate"] is None
    assert row["case_error_count"] == 1


def test_no_errored_or_skipped_run_matches_188_numbers() -> None:
    """The checks rate is identical to 1.88.0's for a run with only verdicts."""
    er = EvalResults()
    for r in (P, P, F):
        er.add("exact_match", r)
        er.add_case(EvalCaseRecord(input="q", per_scorer={"exact_match": r.stored()}))
    assert Scorecard.from_eval_results(er).overall_pass_rate == round(2 / 3, 4)
    assert er.case_pass_rate == round(2 / 3, 4)
