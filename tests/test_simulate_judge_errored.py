"""A simulation judge that could not run is not a verdict (1.89.0, E13).

Before, ``_judge_transcript`` inverted every *failure* criterion's result: "the
judge said the failure happened" became a fail, and anything else, including a
judge that errored, became "the failure did not happen", a **pass**. A
scenario whose judge was down could pass on its failure criteria alone. Now an
errored criterion is ``errored=True, passed=False`` whatever its kind, and the
scenario carries ``error="judge could not run on K criteria: …"``.

D15: the denominator stays every scenario; errored scenarios are counted.
Every span of a simulated conversation is stamped ``fastaiagent.source=simulation``.

Real ``Agent``, real ``LLMJudge`` over ``FunctionModel``/``TestModel``, real
SQLite; no mocks.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.agent.agent import Agent
from fastaiagent.eval import Scorecard
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.eval.simulate import Scenario, SimulatedUser, simulate
from fastaiagent.testing.models import FunctionModel, TestModel


def _down(messages: Any) -> str:
    raise RuntimeError("judge provider down")


def _scenario(*, success: list[str], failure: list[str]) -> Scenario:
    return Scenario(
        name="refund",
        user=SimulatedUser(script=["I want a refund"]),
        success_criteria=success,
        failure_criteria=failure,
        max_turns=2,
    )


def _agent() -> Agent:
    return Agent(name="bot", llm=TestModel(response="Refunds take 5 days."))


def test_an_errored_failure_criterion_is_not_inverted_into_a_pass() -> None:
    judge = LLMJudge(llm=FunctionModel(_down))
    scenario = _scenario(success=[], failure=["The agent promised an instant refund."])
    results = simulate(scenario, _agent(), judge=judge, persist=False)
    r = results.results[0]
    (v,) = r.verdicts
    assert v.kind == "failure" and v.errored is True and v.passed is False
    assert r.passed is False, "a judge outage passed the scenario"
    assert r.error and r.error.startswith("judge could not run on 1 criteria")
    assert v.to_dict()["errored"] is True


def test_an_errored_success_criterion_is_errored_too() -> None:
    judge = LLMJudge(llm=FunctionModel(_down))
    results = simulate(
        _scenario(success=["The agent states the refund window."], failure=[]),
        _agent(),
        judge=judge,
        persist=False,
    )
    r = results.results[0]
    assert r.verdicts[0].errored is True and r.passed is False and r.error


def test_a_real_verdict_is_unchanged() -> None:
    """The control: a judge that answers 0.0 on a failure criterion means the
    failure did not happen, which is still a pass."""
    judge = LLMJudge(llm=TestModel(response=json.dumps({"score": 0.0, "reasoning": "no"})))
    results = simulate(
        _scenario(success=[], failure=["The agent was rude."]), _agent(), judge=judge, persist=False
    )
    r = results.results[0]
    assert r.passed is True and r.error is None and r.verdicts[0].errored is False
    assert "errored" not in r.verdicts[0].to_dict(), "default not written"


def test_errored_scenarios_keep_the_denominator(tmp_path: Path) -> None:
    ok = LLMJudge(llm=TestModel(response=json.dumps({"score": 1.0, "reasoning": "ok"})))
    good = simulate(
        _scenario(success=["The agent replied."], failure=[]), _agent(), judge=ok, persist=False
    )
    bad = simulate(
        _scenario(success=["The agent replied."], failure=[]),
        _agent(),
        judge=LLMJudge(llm=FunctionModel(_down)),
        persist=False,
    )
    results = good
    results.results.extend(bad.results)
    assert (results.pass_count, results.errored_count, len(results.results)) == (1, 1, 2)
    assert results.pass_rate == 0.5, "D15: errored stays in the denominator"
    assert "could not be judged" in results.summary()
    sc = Scorecard.from_simulation(results)
    assert sc.errored == 1 and sc.metrics[0].errored == 1 and sc.overall_pass_rate == 0.5

    run_id = results.persist_local(db_path=tmp_path / "local.db")
    with sqlite3.connect(tmp_path / "local.db") as conn:
        meta = conn.execute("SELECT metadata FROM sim_runs WHERE run_id = ?", (run_id,)).fetchone()
        errs = [r[0] for r in conn.execute("SELECT error FROM sim_cases ORDER BY ordinal")]
    assert json.loads(meta[0]) == {"errored_count": 1}
    assert errs[0] is None and errs[1].startswith("judge could not run")


def test_ui_filters_errored_scenarios(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    ok = LLMJudge(llm=TestModel(response=json.dumps({"score": 1.0, "reasoning": "ok"})))
    results = simulate(
        _scenario(success=["The agent replied."], failure=[]), _agent(), judge=ok, persist=False
    )
    bad = simulate(
        _scenario(success=["The agent replied."], failure=[]),
        _agent(),
        judge=LLMJudge(llm=FunctionModel(_down)),
        persist=False,
    )
    results.results.extend(bad.results)
    db = tmp_path / "local.db"
    run_id = results.persist_local(db_path=db)
    app = build_app(db_path=str(db), auth_path=tmp_path / "auth.json", no_auth=True)
    with TestClient(app) as client:
        errored = client.get(f"/api/simulations/{run_id}?outcome=errored")
        failed = client.get(f"/api/simulations/{run_id}?outcome=failed")
    assert errored.status_code == 200, errored.text
    assert len(errored.json()["cases"]) == 1
    assert failed.json()["cases"] == [], "an errored scenario is not a failed one"


def test_simulation_spans_are_stamped(isolated_local_db: Path) -> None:
    from fastaiagent.trace import otel

    otel.reset()
    try:
        ok = LLMJudge(llm=TestModel(response=json.dumps({"score": 1.0, "reasoning": "ok"})))
        results = simulate(
            _scenario(success=["The agent replied."], failure=[]),
            _agent(),
            judge=ok,
            persist=False,
        )
        trace_id = results.results[0].trace_id
        assert trace_id
        with sqlite3.connect(isolated_local_db) as conn:
            rows = conn.execute(
                "SELECT name, attributes FROM spans WHERE trace_id = ?", (trace_id,)
            ).fetchall()
    finally:
        otel.reset()
    assert rows, "no spans recorded for the simulated conversation"
    sources = {json.loads(a).get("fastaiagent.source") for _, a in rows}
    assert sources == {"simulation"}, rows
