"""Eval traces are labelled and grouped (1.89.0, E3 + D10).

Before, every judge call ran after the agent span had ended, with no enclosing
span, so each became its own loose root ``llm.*`` trace (the audit's DB had 26),
and the case's agent trace looked exactly like production traffic: it counted
in Analytics, Agents, curation and ``learn``.

Now:

* each scored case gets one root ``eval.score`` span, stamped
  ``fastaiagent.source=eval`` with ``fastaiagent.eval.{run_id, run_name, kind,
  case_ordinal, subject_trace_id}``, an ``eval.scorer`` child per scorer
  (structural attributes only, never the reason), and the judge's ``llm.*``
  call beneath it; ``eval_cases.scoring_trace_id`` points at it;
* every span of the case's agent trace is stamped ``source=eval`` (``optimize``
  for an AutoLLM run), carrying the run id and ordinal on its root.

Swept across evaluate / aevaluate and the pytest plugin. Real tracing, real
SQLite; no mocks.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.agent import Agent
from fastaiagent.eval import evaluate
from fastaiagent.eval.evaluate import aevaluate
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.testing import FunctionModel, TestModel

pytest_plugins = ["pytester"]

JUDGE_REPLY = json.dumps({"score": 0.9, "reasoning": "the answer quotes $10 correctly"})


@pytest.fixture
def traced(isolated_local_db: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent.trace import otel

    monkeypatch.delenv("FASTAIAGENT_TRACE_ENABLED", raising=False)
    otel.reset()
    yield isolated_local_db
    otel.reset()


def _spans(db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT trace_id, span_id, parent_span_id, name, status, attributes FROM spans"
        ).fetchall()
    return [{**dict(r), "attributes": json.loads(r["attributes"] or "{}")} for r in rows]


def _cases(db: Path, run_id: str) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(r)
            for r in conn.execute(
                "SELECT ordinal, trace_id, scoring_trace_id FROM eval_cases WHERE run_id = ? "
                "ORDER BY ordinal",
                (run_id,),
            )
        ]


def _judge() -> LLMJudge:
    return LLMJudge(name="helpful", llm=FunctionModel(lambda m: JUDGE_REPLY))


DATA = [{"input": "price?", "expected": "$10"}, {"input": "color?", "expected": "red"}]


def _run(mode: str, **kw: Any) -> Any:
    agent = Agent(name="shop", llm=TestModel(response="It is $10."))
    if mode == "evaluate":
        return evaluate(agent.run, DATA, scorers=["contains", _judge()], **kw)
    return asyncio.run(aevaluate(agent.arun, DATA, scorers=["contains", _judge()], **kw))


@pytest.mark.parametrize("mode", ["evaluate", "aevaluate"])
def test_one_scoring_root_per_case_with_the_judge_beneath(traced: Path, mode: str) -> None:
    results = _run(mode, run_name="spans")
    spans = _spans(traced)
    cases = _cases(traced, results.run_id)
    assert len(cases) == 2 and all(c["scoring_trace_id"] for c in cases)

    for case in cases:
        trace = [s for s in spans if s["trace_id"] == case["scoring_trace_id"]]
        (root,) = [s for s in trace if s["parent_span_id"] is None]
        assert root["name"] == "eval.score"
        a = root["attributes"]
        assert a["fastaiagent.source"] == "eval"
        assert a["fastaiagent.eval.run_id"] == results.run_id
        assert a["fastaiagent.eval.run_name"] == "spans"
        assert a["fastaiagent.eval.kind"] == "evaluate"
        assert a["fastaiagent.eval.case_ordinal"] == case["ordinal"]
        assert a["fastaiagent.eval.subject_trace_id"] == case["trace_id"]
        scorers = [s for s in trace if s["name"] == "eval.scorer"]
        assert {s["attributes"]["fastaiagent.eval.scorer"] for s in scorers} == {
            "contains",
            "helpful",
        }
        assert all(s["parent_span_id"] == root["span_id"] for s in scorers)
        llm = [s for s in trace if s["name"].startswith("llm.")]
        assert llm, "the judge's call is not under the scoring span"
        assert all(s["attributes"].get("fastaiagent.source") == "eval" for s in trace)
        # Structural only: the judge's reasoning is not on the scorer spans.
        assert "$10" not in json.dumps([s["attributes"] for s in scorers])

    loose = [s for s in spans if s["parent_span_id"] is None and s["name"].startswith("llm.")]
    assert loose == [], "a judge call is a loose root trace again"


def test_the_agent_trace_is_stamped_eval(traced: Path) -> None:
    results = _run("evaluate", run_name="agent-stamp")
    spans = _spans(traced)
    for case in _cases(traced, results.run_id):
        trace = [s for s in spans if s["trace_id"] == case["trace_id"]]
        assert trace, "no agent trace recorded"
        assert {s["attributes"].get("fastaiagent.source") for s in trace} == {"eval"}
        (root,) = [s for s in trace if s["parent_span_id"] is None]
        assert root["attributes"]["fastaiagent.eval.run_id"] == results.run_id
        assert root["attributes"]["fastaiagent.eval.case_ordinal"] == case["ordinal"]
        assert "fastaiagent.eval.subject_trace_id" not in root["attributes"]


def test_an_autollm_run_is_stamped_optimize(traced: Path) -> None:
    results = _run("evaluate", run_kind="optimize")
    sources = {
        s["attributes"].get("fastaiagent.source")
        for s in _spans(traced)
        if s["trace_id"] in {c["trace_id"] for c in _cases(traced, results.run_id)}
    }
    assert sources == {"optimize"}


def test_an_errored_scorer_span_is_error(traced: Path) -> None:
    down = LLMJudge(name="down", llm=FunctionModel(lambda m: (_ for _ in ()).throw(RuntimeError)))
    agent = Agent(name="shop", llm=TestModel(response="ok"))
    results = evaluate(agent.run, DATA[:1], scorers=[down])
    (case,) = _cases(traced, results.run_id)
    spans = [s for s in _spans(traced) if s["trace_id"] == case["scoring_trace_id"]]
    (scorer,) = [s for s in spans if s["name"] == "eval.scorer"]
    assert scorer["status"] == "ERROR"
    assert scorer["attributes"]["fastaiagent.eval.outcome"] == "errored"


def test_tracing_off_stores_no_scoring_trace(
    isolated_local_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "0")
    otel.reset()
    try:
        results = _run("evaluate")
        cases = _cases(isolated_local_db, results.run_id)
    finally:
        otel.reset()
    assert [c["scoring_trace_id"] for c in cases] == [None, None]
    assert results.cases[0].per_scorer["helpful"]["passed"] is True, "scoring still ran"


def test_production_views_leave_eval_traces_out(traced: Path, tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    Agent(name="shop", llm=TestModel(response="prod")).run("a real request")
    _run("evaluate")
    app = build_app(db_path=str(traced), auth_path=tmp_path / "auth.json", no_auth=True)
    with TestClient(app) as client:
        agents = client.get("/api/agents").json()
        everything = client.get("/api/agents?include_experiments=true").json()

    def runs(payload: Any) -> int:
        return sum(int(a.get("run_count") or 0) for a in payload["agents"])

    assert runs(agents) == 1, agents
    assert runs(everything) > runs(agents)


_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()
from fastaiagent.agent import Agent
from fastaiagent.eval import case
from fastaiagent.testing import TestModel

@case(input="a", expected="a")
def test_one(evaluate_one):
    evaluate_one(Agent(name="g", llm=TestModel(response="a")).run, scorers=["exact_match"])
"""


def test_pytest_plugin_cases_carry_the_session_run(
    pytester: pytest.Pytester, tmp_path: Path
) -> None:
    import os

    from fastaiagent._internal.config import reset_config

    db = tmp_path / "local.db"
    pytester.makepyfile(_SUITE.format(db=db))
    try:
        pytester.runpytest("--no-header", "--eval-run-name", "ci").assert_outcomes(passed=1)
    finally:
        os.environ.pop("FASTAIAGENT_LOCAL_DB", None)
        reset_config()
    with sqlite3.connect(db) as conn:
        run_id = conn.execute("SELECT run_id FROM eval_runs").fetchone()[0]
        (scoring_trace,) = conn.execute("SELECT scoring_trace_id FROM eval_cases").fetchone()
    root = next(
        s for s in _spans(db) if s["trace_id"] == scoring_trace and s["parent_span_id"] is None
    )
    assert root["attributes"]["fastaiagent.eval.run_id"] == run_id
    assert root["attributes"]["fastaiagent.eval.kind"] == "pytest"
    assert root["attributes"]["fastaiagent.eval.run_name"] == "ci"
