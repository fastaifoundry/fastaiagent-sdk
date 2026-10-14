"""A scorer that raises is one errored verdict, not a lost run (1.89.0).

Before, in ``evaluate()`` one raising scorer aborted the whole run and nothing
was persisted; in the pytest plugin the case never reached the session run. Two
scorers with one name silently overwrote each other's verdicts. A built-in that
needs arguments (``regex_match``) crashed the Local UI's run-eval with a 500.

Real ``evaluate()``, real SQLite, real FastAPI ``TestClient``; no mocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent._internal.errors import EvalError
from fastaiagent.eval import evaluate
from fastaiagent.eval.builtins import ExactMatch
from fastaiagent.eval.compare import load_run
from fastaiagent.eval.scorer import Scorer


@Scorer.code("explodes")
def explodes(input, output, expected=None):  # noqa: ANN001, ANN201
    raise ZeroDivisionError("scorer bug")


@Scorer.code("returns_none")
def returns_none(input, output, expected=None):  # noqa: ANN001, ANN201
    return None


class _NotAResult(Scorer):
    name = "not_a_result"

    def score(self, input, output, expected=None, **kw):  # noqa: ANN001, ANN201
        return {"passed": True}


DATA = [{"input": "a", "expected": "a"}, {"input": "b", "expected": "b"}]


def test_a_raising_scorer_is_errored_and_the_run_is_persisted(isolated_local_db: Path) -> None:
    results = evaluate(lambda q: q, DATA, scorers=["exact_match", explodes], run_name="guard")
    assert results.run_id, "the run was persisted"
    for case in results.cases:
        assert case.per_scorer["exact_match"]["passed"] is True, "the others still ran"
        bad = case.per_scorer["explodes"]
        assert bad["errored"] is True and bad["passed"] is False
        assert "ZeroDivisionError: scorer bug" in bad["reason"]
    stored = load_run("guard")
    assert stored["run"]["check_error_count"] == 2
    assert stored["run"]["case_error_count"] == 2, "a scorer that could not run errors the case"


def test_a_result_that_is_not_a_scorer_result_is_errored() -> None:
    results = evaluate(lambda q: q, DATA[:1], scorers=[_NotAResult()], persist=False)
    v = results.cases[0].per_scorer["not_a_result"]
    assert v["errored"] is True and "not a ScorerResult" in v["reason"]


def test_a_code_scorer_returning_none_is_a_fail_not_a_crash() -> None:
    """Unchanged contract: CodeScorer turns an unexpected return into a fail."""
    results = evaluate(lambda q: q, DATA[:1], scorers=[returns_none], persist=False)
    assert results.cases[0].per_scorer["returns_none"]["passed"] is False


def test_duplicate_scorer_names_raise_before_any_agent_call() -> None:
    calls: list[str] = []

    def agent(q: str) -> str:
        calls.append(q)
        return q

    with pytest.raises(EvalError, match="Two scorers are named 'exact_match'"):
        evaluate(agent, DATA, scorers=["exact_match", ExactMatch()], persist=False)
    assert calls == []


def test_a_builtin_that_needs_arguments_raises_a_clear_error() -> None:
    with pytest.raises(EvalError, match="regex_match.*pattern"):
        evaluate(lambda q: q, DATA, scorers=["regex_match"], persist=False)


def test_an_unknown_scorer_name_raises() -> None:
    with pytest.raises(EvalError, match="Unknown scorer 'nope'"):
        evaluate(lambda q: q, DATA, scorers=["nope"], persist=False)


# --------------------------------------------------------------------------- #
# The Local UI's run-eval: a bad scorer list is the request's fault (422)
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path: Path):  # noqa: ANN201
    # Skips only the UI cells when the [ui] extra is absent, not the module.
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.db import init_local_db
    from fastaiagent.ui.server import build_app

    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    app = build_app(db_path=str(db_path), auth_path=tmp_path / "auth.json", no_auth=True)
    with TestClient(app) as c:
        c.post("/api/datasets", json={"name": "g"})
        c.post("/api/datasets/g/cases", json={"input": "x", "expected_output": "x"})
        yield c


@pytest.mark.parametrize(
    ("scorers", "needle"),
    [
        (["regex_match"], "pattern"),
        (["no_such_scorer"], "Unknown scorer"),
        (["exact_match", "exact_match"], "Two scorers"),
    ],
    ids=["needs-args", "unknown", "duplicate"],
)
def test_run_eval_rejects_a_bad_scorer_list_with_422(client, scorers, needle) -> None:  # noqa: ANN001
    r = client.post("/api/datasets/g/run-eval", json={"scorers": scorers})
    assert r.status_code == 422, r.text
    assert needle in r.json()["detail"]


def test_run_eval_records_kind(client) -> None:  # noqa: ANN001
    r = client.post("/api/datasets/g/run-eval", json={"scorers": ["exact_match"]})
    assert r.status_code == 200, r.text
    run = client.get(f"/api/evals/{r.json()['run_id']}").json()["run"]
    assert run["kind"] == "ui-echo"
