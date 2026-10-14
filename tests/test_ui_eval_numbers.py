"""The Local UI's eval numbers (1.89.0, WS4 4B), over real runs and real spans.

* **Cost** lived on the child ``llm.*`` spans and every reader priced the root
  (or the ``agent.*`` span) only: eval runs showed "—", every agent on the
  Agents page and every workflow $0.00. Eval runs now split **agent cost**
  (the cases' traces) from **eval cost** (the judges' scoring traces).
* **Tokens**: the Traces summary added the root's ``agent.tokens_used`` roll-up
  *and* every call's tokens, so each trace reported twice its tokens.
* The runs list filters by kind, gate and name; the trend and the Overview
  read cases first and leave dataset checks and AutoLLM candidates out.
* A run carries its Agent-CI evidence (``run.gate``); a trace says when it is
  an eval case's scoring or a simulated conversation; a simulation run counts
  its errored scenarios; ``harden()`` saves its report onto the run.

Real ``evaluate()`` / ``simulate()`` over ``TestModel``/``FunctionModel``,
real SQLite, real FastAPI ``TestClient``; no mocks.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.agent import Agent  # noqa: E402
from fastaiagent.eval import evaluate  # noqa: E402
from fastaiagent.eval.gate import gate  # noqa: E402
from fastaiagent.eval.harden import harden  # noqa: E402
from fastaiagent.eval.llm_judge import LLMJudge  # noqa: E402
from fastaiagent.eval.results import record_gate_result  # noqa: E402
from fastaiagent.testing import FunctionModel, TestModel  # noqa: E402
from fastaiagent.ui.pricing import compute_cost_usd  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

AGENT_USAGE = (1000, 200)
JUDGE_REPLY = json.dumps({"score": 0.9, "reasoning": "ok"})


@pytest.fixture
def env(isolated_local_db: Path, tmp_path: Path) -> Any:
    from fastaiagent.trace import otel

    otel.reset()
    app = build_app(db_path=str(isolated_local_db), auth_path=tmp_path / "a.json", no_auth=True)
    yield isolated_local_db, app
    otel.reset()


def _priced_agent(name: str = "shop") -> Agent:
    return Agent(
        name=name, llm=TestModel(response="It is $10.", model="gpt-4o-mini", usage=AGENT_USAGE)
    )


def _judge() -> LLMJudge:
    llm = FunctionModel(lambda m: JUDGE_REPLY, model="gpt-4o-mini")
    return LLMJudge(name="helpful", llm=llm)


DATA = [{"input": "price?", "expected": "$10"}, {"input": "color?", "expected": "$10"}]


def test_agent_and_eval_cost_are_known_and_apart(env: Any) -> None:
    db, app = env
    results = evaluate(_priced_agent().run, DATA, scorers=["contains", _judge()], run_name="c")
    with TestClient(app) as client:
        run = client.get(f"/api/evals/{results.run_id}").json()["run"]
        listed = next(r for r in client.get("/api/evals").json()["rows"] if r["run_name"] == "c")
    per_case = compute_cost_usd("gpt-4o-mini", *AGENT_USAGE)
    assert per_case and per_case > 0
    assert run["agent_cost_usd"] == pytest.approx(2 * per_case, rel=1e-6)
    assert run["agent_cost_known"] is True
    assert run["cost_usd"] == pytest.approx(run["agent_cost_usd"], rel=1e-4), "alias"
    assert run["traced_case_count"] == 2
    # The judge's calls are separate; FunctionModel reports no tokens, so it
    # cannot be priced: unknown, never a silent $0.
    assert run["eval_cost_known"] is False
    assert listed["agent_cost_usd"] == pytest.approx(run["agent_cost_usd"])


def test_an_untraced_run_has_unknown_cost_not_zero(env: Any) -> None:
    db, app = env
    results = evaluate(lambda q: "$10", DATA, scorers=["contains"])
    with TestClient(app) as client:
        run = client.get(f"/api/evals/{results.run_id}").json()["run"]
    assert run["agent_cost_usd"] is None and run["agent_cost_known"] is False


def test_trace_tokens_are_not_double_counted(env: Any) -> None:
    db, app = env
    result = _priced_agent().run("x")
    with TestClient(app) as client:
        trace = client.get(f"/api/traces/{result.trace_id}").json()
    assert trace["total_tokens"] == sum(AGENT_USAGE), "the root roll-up and calls were both added"
    with TestClient(app) as client:
        listed = client.get("/api/traces").json()
    (row,) = [t for t in listed["rows"] if t["trace_id"] == result.trace_id]
    assert row["total_tokens"] == sum(AGENT_USAGE)


def test_agents_show_real_cost_each_their_own(env: Any) -> None:
    from fastaiagent.llm.message import MessageRole
    from fastaiagent.tool import tool

    db, app = env
    _priced_agent("solo").run("x")
    helper = _priced_agent("helper")

    @tool()
    def ask_helper(question: str) -> str:
        """Ask the helper agent."""
        return helper.run(question).output

    def lead_turn(messages: list[Any]) -> Any:
        if messages[-1].role == MessageRole.tool:
            return ("done", [])
        return ("", [{"name": "ask_helper", "arguments": {"question": "price?"}}])

    Agent(name="lead", llm=FunctionModel(lead_turn), tools=[ask_helper]).run("go")
    with sqlite3.connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM spans")}
    assert {"agent.lead", "agent.helper", "tool.ask_helper"} <= names, names

    with TestClient(app) as client:
        agents = {a["agent_name"]: a for a in client.get("/api/agents").json()["agents"]}
    per_call = compute_cost_usd("gpt-4o-mini", *AGENT_USAGE)
    assert agents["solo"]["avg_cost_usd"] == pytest.approx(per_call, rel=1e-6)
    assert agents["helper"]["avg_cost_usd"] == pytest.approx(per_call, rel=1e-6)
    # The lead's own calls are unpriced (FunctionModel); the helper's call
    # under it is the helper's, not counted again for the lead.
    assert agents["lead"]["avg_cost_usd"] == 0.0


def test_runs_list_filters_and_trend(env: Any) -> None:
    db, app = env
    a = evaluate(lambda q: "$10", DATA, scorers=["contains"], run_name="eval-run")
    b = evaluate(lambda q: q, DATA, scorers=["exact_match"], run_name="echo", run_kind="ui-echo")
    c = evaluate(lambda q: "$10", DATA, scorers=["contains"], run_name="cand", run_kind="optimize")
    record_gate_result(a.run_id, gate_outcome="failed", thresholds={"overall.pass_rate": 1.0})
    with TestClient(app) as client:

        def names(**params: Any) -> list[str]:
            body = client.get("/api/evals", params=params).json()
            return sorted(r["run_name"] for r in body["rows"])

        assert names(kind="evaluate") == ["eval-run"]
        assert names(exclude_kind="ui-echo,optimize") == ["eval-run"]
        assert names(kind="ui-echo") == ["echo"]
        assert names(gate="failed") == ["eval-run"]
        assert names(gate="none") == ["cand", "echo"]
        assert names(gate="failed,invalid") == ["eval-run"]
        assert names(gate="failed,none") == ["cand", "echo", "eval-run"]
        assert names(q="can") == ["cand"]
        paged = client.get("/api/evals", params={"page_size": 2}).json()
        assert paged["total"] == 3 and len(paged["rows"]) == 2

        points = client.get("/api/evals/trend").json()["points"]
        assert [p["run_id"] for p in points] == [a.run_id], "dataset checks / AutoLLM excluded"
        p = points[0]
        assert p["case_pass_rate"] == 1.0 and p["gate_outcome"] == "failed"
        assert {"kind", "series", "pass_rate", "run_name"} <= set(p)
        everything = client.get("/api/evals/trend", params={"include": "all"}).json()["points"]
        assert {q["run_id"] for q in everything} == {a.run_id, b.run_id, c.run_id}


def test_compare_checks_delta_is_null_when_a_run_has_no_rate(env: Any) -> None:
    db, app = env
    rated = evaluate(lambda q: "$10", DATA, scorers=["contains"])
    nothing = evaluate(lambda q: "$10", [{"input": "q"}, {"input": "r"}], scorers=["contains"])
    with TestClient(app) as client:
        body = client.get(f"/api/evals/compare?a={rated.run_id}&b={nothing.run_id}").json()
    assert body["pass_rate_delta"] is None, "0.0 read as 'no change'"


def test_overview_reads_cases_first(env: Any) -> None:
    db, app = env
    good = evaluate(lambda q: "$10", DATA, scorers=["contains"], dataset_name="shop.jsonl")
    record_gate_result(good.run_id, gate_outcome="passed", thresholds={})
    echo = evaluate(lambda q: q, DATA, scorers=["exact_match"], run_kind="ui-echo")
    # A dataset check's gate is not a quality gate: not counted.
    record_gate_result(echo.run_id, gate_outcome="failed", thresholds={})
    # A later outage on the same dataset (the one judged case failed, 0%) is
    # reported as a gate failure, but must not become the headline.
    outage = evaluate(lambda q: "nope", DATA[:1], scorers=["contains"], dataset_name="shop.jsonl")
    record_gate_result(outage.run_id, gate_outcome="invalid", thresholds={})
    with TestClient(app) as client:
        o = client.get("/api/overview").json()
    assert o["eval_case_pass_rate_7d"] == 1.0, "an echo run or an outage dragged the headline"
    assert (o["eval_runs_counted_7d"], o["eval_runs_excluded_7d"]) == (1, 2)
    assert o["eval_gate_failures_7d"] == 1
    assert "avg_pass_rate_last_7d" in o, "the legacy key stays"
    recent = o["recent_eval_runs"][0]
    assert {"case_pass_rate", "kind", "gate_outcome"} <= set(recent)


def test_run_carries_its_agent_ci_evidence(env: Any) -> None:
    db, app = env
    results = evaluate(lambda q: "$10", DATA, scorers=["contains"], run_name="ci")
    report = gate(results, fail_under=["contains.pass_rate=0.9"], max_error_rate=0.1)
    record_gate_result(
        results.run_id,
        gate_outcome=report.outcome,
        thresholds={"contains.pass_rate": 0.9},
        baseline={"run_id": "base", "pass_rate_delta": 0.0, "regressed_count": 0},
        report=report,
        baseline_detail={"run_name": "main", "tolerance": 0.02},
    )
    with TestClient(app) as client:
        g = client.get(f"/api/evals/{results.run_id}").json()["run"]["gate"]
    assert g["outcome"] == "passed"
    assert g["checks"] == [
        {"metric": "contains", "field": "pass_rate", "minimum": 0.9, "actual": 1.0, "passed": True}
    ]
    assert g["error"]["max_error_rate"] == 0.1 and g["error"]["errored"] == 0
    assert g["baseline"]["run_name"] == "main" and g["baseline"]["regressed_count"] == 0

    from fastaiagent.eval.platform_export import build_payloads

    payload = build_payloads(run_id=results.run_id, db_path=db)[0]
    assert set(payload["baseline"]) == {"run_id", "pass_rate_delta", "regressed_count"}, (
        "the local detail must not reach the wire"
    )
    assert "gate_checks" not in json.dumps(payload)


def test_trace_scores_explain_scoring_and_simulation_traces(env: Any) -> None:
    from fastaiagent.eval.simulate import Scenario, SimulatedUser, simulate

    db, app = env
    results = evaluate(_priced_agent().run, DATA[:1], scorers=[_judge()], run_name="s")
    sim = simulate(
        Scenario(
            name="hello",
            user=SimulatedUser(script=["hi"]),
            success_criteria=["The agent replied."],
            max_turns=2,
        ),
        _priced_agent(),
        judge=LLMJudge(llm=TestModel(response=JUDGE_REPLY)),
    )
    with sqlite3.connect(db) as conn:
        scoring_trace = conn.execute(
            "SELECT scoring_trace_id FROM eval_cases WHERE run_id = ?", (results.run_id,)
        ).fetchone()[0]
    with TestClient(app) as client:
        scoring = client.get(f"/api/traces/{scoring_trace}/scores").json()
        conversation = client.get(f"/api/traces/{sim.results[0].trace_id}/scores").json()
        sim_run = client.get(f"/api/simulations/{sim.run_id}").json()["run"]
    (row,) = scoring["eval_scoring"]
    assert row["run_id"] == results.run_id and row["ordinal"] == 0
    (case,) = conversation["simulation_cases"]
    assert case["scenario_name"] == "hello" and case["passed"] == 1
    assert (sim_run["errored_count"], sim_run["judged_count"]) == (0, 1)


def test_case_media_endpoint(env: Any) -> None:
    pytest.importorskip("PIL")
    from fastaiagent.multimodal.image import Image
    from tests.test_eval_multimodal_persist import _png

    db, app = env
    results = evaluate(
        lambda q: "x",
        [{"input": Image.from_bytes(_png(32, 32), "image/png"), "expected": "x"}],
        scorers=["exact_match"],
    )
    with sqlite3.connect(db) as conn:
        case_id = conn.execute(
            "SELECT case_id FROM eval_cases WHERE run_id = ?", (results.run_id,)
        ).fetchone()[0]
    with TestClient(app) as client:
        ok = client.get(f"/api/evals/{results.run_id}/cases/{case_id}/media/0")
        missing = client.get(f"/api/evals/{results.run_id}/cases/{case_id}/media/7")
        wrong_run = client.get(f"/api/evals/other-run/cases/{case_id}/media/0")
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg"
    assert ok.content[:2] == b"\xff\xd8"
    assert missing.status_code == 404 and wrong_run.status_code == 404


@pytest.mark.parametrize("kind", ["eval", "simulation"])
def test_harden_saves_its_report_on_the_run(env: Any, kind: str) -> None:
    from fastaiagent.eval.simulate import Scenario, SimulatedUser, simulate

    db, app = env
    if kind == "eval":
        results = evaluate(lambda q: "wrong", DATA, scorers=["contains"], run_name="h")
        table = "eval_runs"
    else:
        results = simulate(
            Scenario(
                name="x",
                user=SimulatedUser(script=["hi"]),
                success_criteria=["The agent says the price."],
                max_turns=2,
            ),
            _priced_agent(),
            judge=LLMJudge(llm=TestModel(response=json.dumps({"score": 0.0, "reasoning": "no"}))),
        )
        table = "sim_runs"
    reply = json.dumps(
        {"recommendations": [{"target": "instructions", "recommendation": "say the price"}]}
    )
    report = harden(_priced_agent(), results, llm=FunctionModel(lambda m: reply))
    assert report.recommendations
    with sqlite3.connect(db) as conn:
        meta = json.loads(
            conn.execute(
                f"SELECT metadata FROM {table} WHERE run_id = ?", (results.run_id,)
            ).fetchone()[0]
        )
        spans = conn.execute(
            "SELECT attributes FROM spans WHERE name = 'eval.harden' AND parent_span_id IS NULL"
        ).fetchall()
    h = meta["hardening"]
    assert h["failure_count"] == report.failure_count
    assert h["recommendations"][0]["recommendation"] == "say the price"
    assert spans and json.loads(spans[-1][0])["fastaiagent.source"] == "eval"


def test_harden_persist_false_writes_nothing(env: Any) -> None:
    db, app = env
    results = evaluate(lambda q: "wrong", DATA, scorers=["contains"])
    reply = json.dumps({"recommendations": [{"target": "tools", "recommendation": "x"}]})
    harden(object(), results, llm=FunctionModel(lambda m: reply), persist=False)
    with sqlite3.connect(db) as conn:
        meta = json.loads(
            conn.execute(
                "SELECT metadata FROM eval_runs WHERE run_id = ?", (results.run_id,)
            ).fetchone()[0]
        )
    assert "hardening" not in meta
