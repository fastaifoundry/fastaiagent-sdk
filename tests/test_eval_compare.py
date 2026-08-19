"""``fastaiagent.eval.compare`` — load_run / compare_runs over real SQLite.

Also pins parity with the UI route, which now delegates its matching and
bucketing to this module.
"""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import EvalError
from fastaiagent.eval import evaluate
from fastaiagent.eval.compare import bucket_cases, compare_runs, load_run, match_cases

DATASET = [
    {"input": "greet", "expected": "hi"},
    {"input": "farewell", "expected": "bye"},
]


def _perfect(x: str) -> str:
    return {"greet": "hi", "farewell": "bye"}[x]


def _regressed(x: str) -> str:
    return {"greet": "hi", "farewell": "WRONG"}[x]


def _persist(agent_fn, tmp_path, run_name: str) -> str:
    results = evaluate(agent_fn, DATASET, scorers=["exact_match"], persist=False, concurrency=1)
    return results.persist_local(db_path=tmp_path / "local.db", run_name=run_name)


def test_load_run_by_id_and_by_name(tmp_path) -> None:
    run_id = _persist(_perfect, tmp_path, "main")
    by_id = load_run(run_id, db_path=tmp_path / "local.db")
    by_name = load_run("main", db_path=tmp_path / "local.db")
    assert by_id["run"]["run_id"] == by_name["run"]["run_id"] == run_id
    assert len(by_id["cases"]) == 2
    assert isinstance(by_id["cases"][0]["per_scorer"], dict)  # JSON unpacked


def test_load_run_latest_name_wins(tmp_path) -> None:
    _persist(_perfect, tmp_path, "main")
    second = _persist(_regressed, tmp_path, "main")
    assert load_run("main", db_path=tmp_path / "local.db")["run"]["run_id"] == second


def test_load_run_not_found_raises(tmp_path) -> None:
    _persist(_perfect, tmp_path, "main")
    with pytest.raises(EvalError, match="not found"):
        load_run("nope", db_path=tmp_path / "local.db")


def test_compare_runs_buckets_regression(tmp_path) -> None:
    _persist(_perfect, tmp_path, "main")
    _persist(_regressed, tmp_path, "pr")
    cmp = compare_runs("main", "pr", db_path=tmp_path / "local.db")
    assert len(cmp.regressed) == 1
    assert cmp.regressed[0]["a"]["input"] == "farewell"
    assert cmp.regressed[0]["scorer_deltas"][0]["changed"] is True
    assert cmp.unchanged_pass == 1
    assert cmp.pass_rate_delta == -0.5
    assert cmp.has_regressions
    assert any("regressed:" in line for line in cmp.describe())


def test_match_cases_falls_back_to_input() -> None:
    """Ordinal matches first; a missing ordinal falls back to input equality."""
    a = [{"ordinal": 0, "input": "x"}, {"ordinal": 1, "input": "y"}]
    b = [{"ordinal": 0, "input": "x"}, {"ordinal": 5, "input": "y"}]
    pairs = [(pa["input"], pb["input"]) for pa, pb in match_cases(a, b)]
    # "x" pairs by ordinal 0; "y" has no ordinal-1 partner and pairs by input.
    assert pairs == [("x", "x"), ("y", "y")]


def test_errored_cases_are_non_signal_in_buckets() -> None:
    passed = {"ordinal": 0, "input": "q", "per_scorer": {"m": {"passed": True}}}
    errored = {"ordinal": 0, "input": "q", "per_scorer": {}, "error": "503"}
    regressed, improved, up, uf = bucket_cases([passed], [errored])
    assert (regressed, improved, up, uf) == ([], [], 0, 0)


def test_ui_route_parity(tmp_path) -> None:
    """The route's compare endpoint delegates here — same buckets by construction.

    This test locks the *shape* the route depends on: entry dicts carry
    ``a`` / ``b`` / ``scorer_deltas`` keys.
    """
    _persist(_perfect, tmp_path, "main")
    _persist(_regressed, tmp_path, "pr")
    a = load_run("main", db_path=tmp_path / "local.db")
    b = load_run("pr", db_path=tmp_path / "local.db")
    regressed, improved, up, uf = bucket_cases(a["cases"], b["cases"])
    assert {"a", "b", "scorer_deltas"} <= set(regressed[0].keys())
    assert up == 1 and uf == 0 and improved == []
