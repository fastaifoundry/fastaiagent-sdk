"""Infra-failure visibility across results → scorecard → persistence (1.48.0).

The false-green defect: before 1.48.0 a run where most cases infrastructure-
failed reported ``pass_rate == 1.0`` because errored cases are unscored and
nothing surfaced the error count. These tests pin the fix end-to-end against
real SQLite — no mocks.
"""

from __future__ import annotations

import pytest

from fastaiagent.eval import evaluate
from fastaiagent.eval.gate import gate, parse_threshold
from fastaiagent.eval.results import EvalCaseRecord, EvalResults, Scorecard
from fastaiagent.ui.db import init_local_db


def _good(x: str) -> str:
    return "hi"


def _flaky(x: str) -> str:
    raise TimeoutError("provider 500")


def test_errored_count_and_error_rate_properties() -> None:
    results = EvalResults()
    results.add_case(EvalCaseRecord(input="a", error="boom"))
    results.add_case(EvalCaseRecord(input="b", per_scorer={"exact_match": {"passed": True}}))
    assert results.errored_count == 1
    assert results.error_rate == 0.5
    assert "errored: 1/2" in results.summary()


def test_scorecard_carries_errored_count() -> None:
    results = evaluate(
        _flaky,
        [{"input": "a", "expected": "hi"}] * 3 + [{"input": "b", "expected": "hi"}] * 0,
        scorers=["exact_match"],
        persist=False,
    )
    sc = Scorecard.from_eval_results(results)
    assert sc.errored == 3
    assert "errored (unscored)=3" in sc.summary()
    assert sc.to_dict()["errored"] == 3


def test_false_green_run_is_invalid_not_passing() -> None:
    """190/200-crashed scenario: gate must say INVALID, never green."""
    dataset = [{"input": f"q{i}", "expected": "hi"} for i in range(10)]

    calls = {"n": 0}

    def mostly_broken(x: str) -> str:
        calls["n"] += 1
        if calls["n"] <= 9:
            raise ConnectionError("upstream 503")
        return "hi"

    results = evaluate(
        mostly_broken, dataset, scorers=["exact_match"], persist=False, concurrency=1
    )
    # The one scored case passed — the old pass_rate is a perfect 1.0 …
    assert Scorecard.from_eval_results(results).overall_pass_rate == 1.0
    # … but the gate sees 90% infra errors and refuses to call it green.
    report = gate(results, fail_under=["overall.pass_rate=0.9"], max_error_rate=0.1)
    assert report.outcome == "invalid"
    assert report.errored == 9


def test_error_persists_to_eval_cases_column(tmp_path) -> None:
    results = evaluate(
        _flaky, [{"input": "a", "expected": "hi"}], scorers=["exact_match"], persist=False
    )
    run_id = results.persist_local(db_path=tmp_path / "local.db", run_name="r")
    db = init_local_db(tmp_path / "local.db")
    try:
        row = db.fetchone("SELECT error FROM eval_cases WHERE run_id = ?", (run_id,))
        assert row is not None and "provider 500" in row["error"]
    finally:
        db.close()


def test_case_outcome_classifies_errored() -> None:
    """The shared classifier — no optional deps needed."""
    from fastaiagent.eval.compare import case_outcome

    assert case_outcome({"error": "boom", "per_scorer": {}}) == "errored"
    assert case_outcome({"per_scorer": {"m": {"passed": True}}}) == "passed"
    assert case_outcome({"per_scorer": {"m": {"passed": False}}}) == "failed"


def test_ui_route_delegates_to_shared_case_outcome() -> None:
    """The UI route is a thin wrapper; needs the optional [ui] extra."""
    pytest.importorskip("fastapi")
    from fastaiagent.ui.routes.evals import _case_outcome

    assert _case_outcome({"error": "boom", "per_scorer": {}}) == "errored"
    assert _case_outcome({"per_scorer": {"m": {"passed": True}}}) == "passed"
    assert _case_outcome({"per_scorer": {"m": {"passed": False}}}) == "failed"


def test_persist_local_metadata_kwarg_and_git_provenance(tmp_path) -> None:
    results = evaluate(
        _good, [{"input": "a", "expected": "hi"}], scorers=["exact_match"], persist=False
    )
    run_id = results.persist_local(
        db_path=tmp_path / "local.db", run_name="r", metadata={"suite": "smoke"}
    )
    import json

    db = init_local_db(tmp_path / "local.db")
    try:
        row = db.fetchone("SELECT metadata FROM eval_runs WHERE run_id = ?", (run_id,))
        meta = json.loads(row["metadata"])
    finally:
        db.close()
    assert meta["suite"] == "smoke"
    # Running inside this repo: git provenance is best-effort captured.
    assert "git_sha" in meta and len(meta["git_sha"]) >= 7


def test_threshold_grammar() -> None:
    t = parse_threshold("overall.pass_rate=0.9")
    assert (t.metric, t.field, t.minimum) == ("overall", "pass_rate", 0.9)
    t = parse_threshold("geval.avg_score=0.7")
    assert (t.metric, t.field) == ("geval", "avg_score")
    t = parse_threshold("exact_match=0.85")
    assert (t.metric, t.field, t.minimum) == ("exact_match", "pass_rate", 0.85)

    from fastaiagent._internal.errors import EvalError

    with pytest.raises(EvalError):
        parse_threshold("no-equals-sign")
    with pytest.raises(EvalError):
        parse_threshold("overall.pass_rate=not-a-number")


def test_gate_fails_on_missing_metric() -> None:
    """A typo'd scorer name must fail the gate, never silently pass."""
    results = evaluate(
        _good, [{"input": "a", "expected": "hi"}], scorers=["exact_match"], persist=False
    )
    report = gate(results, fail_under=["tpyo_scorer.pass_rate=0.5"])
    assert report.outcome == "failed"
    assert report.checks[0].actual is None
