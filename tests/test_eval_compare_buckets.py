"""Compare puts every case somewhere (1.89.0, E14).

Before, ``match_cases`` dropped any case it could not pair, ``bucket_cases``
dropped every pair where either side errored, and two dataset rows with the same
input both paired with the *last* duplicate in the other run. A comparison could
report "no regressions" while a third of the cases were silently missing. Now:
``errored`` (with the side), ``only_in_a`` / ``only_in_b``, score ``drifted``
pairs, a cases-rate delta, and a warning when a run predates 1.89.0's rates.

Real SQLite via ``persist_local``, the real UI route; no mocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fastaiagent.eval.compare import (
    align_cases,
    bucket_cases,
    compare_cases,
    compare_runs,
    match_cases,
    scorer_deltas,
)
from fastaiagent.eval.results import EvalCaseRecord, EvalResults
from fastaiagent.eval.scorer import ScorerResult


def _case(i: int, inp: str, score: float | None, *, error: str | None = None) -> dict[str, Any]:
    if error:
        return {"ordinal": i, "input": inp, "per_scorer": {}, "error": error}
    if score is None:
        return {"ordinal": i, "input": inp, "per_scorer": {"j": {"errored": True, "passed": False}}}
    return {
        "ordinal": i,
        "input": inp,
        "per_scorer": {"j": {"passed": score >= 0.5, "score": score}},
    }


A = [
    _case(0, "keep-pass", 0.9),
    _case(1, "regress", 0.9),
    _case(2, "improve", 0.1),
    _case(3, "outage-b", 0.9),
    _case(4, "drift", 0.95),
    _case(5, "dropped", 0.9),
]
B = [
    _case(0, "keep-pass", 0.9),
    _case(1, "regress", 0.1),
    _case(2, "improve", 0.9),
    _case(3, "outage-b", None),
    _case(4, "drift", 0.6),
    _case(5, "added", 0.9),
]


def test_every_case_is_in_exactly_one_bucket() -> None:
    c = compare_cases(A, B)
    assert [e["a"]["input"] for e in c.regressed] == ["regress"]
    assert [e["a"]["input"] for e in c.improved] == ["improve"]
    assert (c.unchanged_pass, c.unchanged_fail) == (2, 0)
    assert [(e["a"]["input"], e["side"]) for e in c.errored] == [("outage-b", "b")]
    assert [x["input"] for x in c.only_in_a] == ["dropped"]
    assert [x["input"] for x in c.only_in_b] == ["added"]
    placed = (
        len(c.regressed)
        + len(c.improved)
        + c.unchanged_pass
        + c.unchanged_fail
        + len(c.errored)
        + len(c.only_in_a)
    )
    assert placed == len(A)


def test_drift_lists_only_the_scorers_that_moved() -> None:
    c = compare_cases(A, B, drift_threshold=0.1)
    assert [e["a"]["input"] for e in c.drifted] == ["drift"]
    (d,) = c.drifted[0]["scorer_deltas"]
    assert (d["score_before"], d["score_after"], d["score_delta"]) == (0.95, 0.6, -0.35)
    assert compare_cases(A, B, drift_threshold=0.5).drifted == []


def test_errored_side_both() -> None:
    a = [_case(0, "x", None)]
    b = [_case(0, "x", None, error="BOOM")]
    (e,) = compare_cases(a, b).errored
    assert e["side"] == "both"


def test_duplicate_inputs_pair_in_order_not_all_with_the_last() -> None:
    a = [_case(0, "same", 0.9), _case(1, "same", 0.1)]
    b = [_case(7, "same", 0.9), _case(8, "same", 0.1)]  # ordinals do not line up
    pairs, only_a, only_b = align_cases(a, b)
    assert [(x["ordinal"], y["ordinal"]) for x, y in pairs] == [(0, 7), (1, 8)]
    assert only_a == [] and only_b == []
    assert match_cases(a, b) == pairs


def test_bucket_cases_keeps_its_old_shape() -> None:
    regressed, improved, up, uf = bucket_cases(A, B)
    assert (len(regressed), len(improved), up, uf) == (1, 1, 2, 0)


def test_scorer_deltas_outcomes_and_no_placeholder_scores() -> None:
    (d,) = scorer_deltas(_case(0, "x", 0.9), _case(0, "x", None))
    assert (d["outcome_before"], d["outcome_after"]) == ("passed", "errored")
    assert d["score_after"] is None and d["score_delta"] is None, "0.0 is a placeholder"
    (d,) = scorer_deltas({"per_scorer": {"j": {"passed": True, "score": 1}}}, {"per_scorer": {}})
    assert d["outcome_after"] is None


def _persist(db: Path, name: str, scores: list[float | None], *, legacy: bool = False) -> str:
    er = EvalResults()
    for i, s in enumerate(scores):
        r = (
            ScorerResult.error("judge down")
            if s is None
            else ScorerResult(score=s, passed=s >= 0.5)
        )
        er.add("j", r)
        er.add_case(EvalCaseRecord(input=f"q{i}", per_scorer={"j": r.stored()}))
    run_id = er.persist_local(db_path=db, run_name=name)
    if legacy:
        import sqlite3

        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE eval_runs SET summary_version = NULL WHERE run_id = ?", (run_id,))
    return run_id


def test_compare_runs_fields_and_legacy_warning(tmp_path: Path) -> None:
    db = tmp_path / "local.db"
    _persist(db, "base", [0.9, 0.1, 0.9, 0.9])
    _persist(db, "new", [0.9, 0.9, None, 0.9])
    cmp = compare_runs("base", "new", db_path=db)
    assert len(cmp.errored) == 1 and len(cmp.improved) == 1
    assert cmp.case_pass_rate_delta == pytest.approx(1.0 - 0.75)
    assert cmp.pass_rate_comparable is True
    assert not any("warning" in line for line in cmp.describe())

    _persist(db, "old", [0.9, 0.1, 0.9, 0.9], legacy=True)
    cmp = compare_runs("old", "new", db_path=db)
    assert cmp.pass_rate_comparable is False
    assert any("persisted before 1.89.0" in line for line in cmp.describe())


def test_ui_compare_route_carries_the_new_buckets(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    db = tmp_path / "local.db"
    a = _persist(db, "base", [0.9, 0.1, 0.9])
    b = _persist(db, "new", [0.9, 0.9, None, 0.9])
    app = build_app(db_path=str(db), auth_path=tmp_path / "auth.json", no_auth=True)
    with TestClient(app) as client:
        body = client.get(f"/api/evals/compare?a={a}&b={b}").json()
    assert len(body["errored"]) == 1 and body["errored"][0]["side"] == "b"
    assert [c["input"] for c in body["only_in_b"]] == ["q3"]
    assert body["only_in_a"] == [] and body["drifted"] == []
    assert body["pass_rate_comparable"] is True
    assert body["case_pass_rate_delta"] == pytest.approx(1.0 - 2 / 3, abs=1e-4)
