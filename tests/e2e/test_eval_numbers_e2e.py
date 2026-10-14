"""Live e2e: 1.89.0's eval numbers against a real provider (marked ``e2e``).

Ports the audit's ``run_evals.py`` shape: a real gpt-4o-mini agent with a tool,
per-row ``expected_trajectory``, one case whose agent raises (BOOM), and a judge
outage (a model that does not exist). Swept across ``evaluate`` / ``aevaluate``
× ``agent.run`` / ``agent.arun``; the pytest plugin, the CLI and ``simulate``
with a live judge. Needs ``OPENAI_API_KEY``; no mocks.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

MODEL = "gpt-4o-mini"


def _agent() -> Any:
    from fastaiagent.agent import Agent
    from fastaiagent.llm import LLMClient
    from fastaiagent.tool import tool

    @tool()
    def lookup_order(order_id: str) -> str:
        """Look up the shipping status of an order by its id."""
        return f"Order {order_id} shipped on Monday via FedEx."

    return Agent(
        name="tip-helper",
        system_prompt=(
            "You answer order questions. Always call lookup_order for any order id "
            "before answering. Never call it when no order id is given."
        ),
        llm=LLMClient(provider="openai", model=MODEL, temperature=0),
        tools=[lookup_order],
    )


DATASET = [
    {
        "input": "Where is order A17?",
        "expected_output": "FedEx",
        "expected_trajectory": ["lookup_order"],
    },
    {
        "input": "What are your opening hours?",
        "expected_output": "hours",
        "expected_trajectory": [],
    },
    {"input": "BOOM", "expected_output": "x"},
    {"input": "Where is order B42?", "expected_output": "Monday"},
]


def _with_boom(fn: Any) -> Any:
    if asyncio.iscoroutinefunction(fn):

        async def afn(q: str) -> Any:
            if q == "BOOM":
                raise RuntimeError("BOOM")
            return await fn(q)

        return afn

    def sfn(q: str) -> Any:
        if q == "BOOM":
            raise RuntimeError("BOOM")
        return fn(q)

    return sfn


@requires_openai
@pytest.mark.parametrize("mode", ["evaluate/run", "aevaluate/arun"])
def test_live_case_and_check_numbers(mode: str) -> None:
    from fastaiagent.eval import Scorecard, evaluate
    from fastaiagent.eval.evaluate import aevaluate
    from fastaiagent.eval.gate import gate
    from fastaiagent.eval.trajectory import ToolUsageAccuracy

    agent = _agent()
    scorers: list[Any] = ["contains", ToolUsageAccuracy()]
    if mode == "evaluate/run":
        results = evaluate(_with_boom(agent.run), DATASET, scorers=scorers, persist=False)
    else:
        results = asyncio.run(
            aevaluate(_with_boom(agent.arun), DATASET, scorers=scorers, persist=False)
        )

    tua = [c.per_scorer.get("tool_usage_accuracy") for c in results.cases]
    assert tua[0] and tua[0]["passed"] is True, tua[0]
    assert tua[0]["details"]["actual"] == ["lookup_order"], "the live tool call was captured"
    assert results.cases[2].error and "BOOM" in results.cases[2].error
    assert tua[3] and tua[3].get("skipped") is True, "no expectation: does not apply"

    cc = results.case_counts
    assert cc["case_count"] == 4 and cc["case_error_count"] >= 1
    sc = Scorecard.from_eval_results(results)
    assert sc.cases["count"] == 4
    assert sc.checks_skipped >= 1
    assert gate(results, fail_under=["tool_usage_accuracy.pass_rate=0.5"]).outcome != "invalid"


@requires_openai
def test_live_judge_outage_is_invalid_and_scrubbed() -> None:
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.gate import gate
    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.llm import LLMClient

    judge = LLMJudge(
        name="correctness",
        llm=LLMClient(provider="openai", model="gpt-does-not-exist-xyz", max_retries=0),
    )
    results = evaluate(
        lambda q: "Paris",
        [{"input": "Capital of France?", "expected": "Paris"}],
        scorers=[judge],
        persist=False,
    )
    v = results.cases[0].per_scorer["correctness"]
    assert v["errored"] is True and v["passed"] is False, v
    assert os.environ["OPENAI_API_KEY"] not in (v["reason"] or "")
    report = gate(results, fail_under=["correctness.pass_rate=0.5"])
    assert report.outcome == "invalid", report.describe()


@requires_openai
def test_live_agent_and_eval_cost_reach_the_ui(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Phase 7: both costs are real and apart, on a live provider."""
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent._internal.config import reset_config
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.llm import LLMClient
    from fastaiagent.trace import otel
    from fastaiagent.ui.server import build_app

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    otel.reset()
    try:
        judge = LLMJudge(
            name="correctness", llm=LLMClient(provider="openai", model=MODEL, temperature=0)
        )
        results = evaluate(
            _agent().run,
            [{"input": "Where is order A17?", "expected": "FedEx"}],
            scorers=[judge],
        )
        app = build_app(db_path=str(db), auth_path=tmp_path / "auth.json", no_auth=True)
        with TestClient(app) as client:
            run = client.get(f"/api/evals/{results.run_id}").json()["run"]
            scoring_trace = client.get(f"/api/evals/{results.run_id}").json()["cases"][0][
                "scoring_trace_id"
            ]
            scores = client.get(f"/api/traces/{scoring_trace}/scores").json()
    finally:
        otel.reset()
        reset_config()
    assert run["agent_cost_known"] is True and run["agent_cost_usd"] > 0, run
    assert run["eval_cost_known"] is True and run["eval_cost_usd"] > 0, run
    assert scores["eval_scoring"][0]["run_id"] == results.run_id


@requires_openai
def test_live_ui_run_eval_with_a_judge_spec(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Phase 8: the Local UI's run-eval, a real agent, an llm_judge spec."""
    pytest.importorskip("fastapi")
    pytest.importorskip("itsdangerous")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.db import init_local_db
    from fastaiagent.ui.server import build_app

    fa = tmp_path / ".fastaiagent"
    (fa / "datasets").mkdir(parents=True)
    db = fa / "local.db"
    init_local_db(db).close()
    (fa / "datasets" / "orders.jsonl").write_text(
        json.dumps(
            {
                "input": "Where is order A17?",
                "expected_output": "FedEx",
                "expected_trajectory": ["lookup_order"],
            }
        )
        + "\n"
    )
    app = build_app(
        db_path=str(db), auth_path=tmp_path / "a.json", no_auth=True, runners=[_agent()]
    )
    with TestClient(app) as client:
        options = client.get("/api/datasets/orders/run-eval/options").json()
        assert {"provider": "openai", "model": MODEL} in options["judge_models"]
        r = client.post(
            "/api/datasets/orders/run-eval",
            json={
                "agent_name": "tip-helper",
                "scorers": [
                    "contains",
                    "tool_usage_accuracy",
                    {
                        "type": "llm_judge",
                        "name": "helpful",
                        "criteria": "Does the reply tell the customer how the order shipped?",
                        "model": {"provider": "openai", "model": MODEL},
                    },
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        run = client.get(f"/api/evals/{body['run_id']}").json()
    assert body["kind"] == "ui" and body["cases"]["errored"] == 0, body
    per = run["cases"][0]["per_scorer"]
    assert per["helpful"]["passed"] is True and "errored" not in per["helpful"], per
    assert per["tool_usage_accuracy"]["passed"] is True, per
    assert run["run"]["metadata"]["scorer_config"]["helpful"]["model"] == MODEL


@requires_openai
def test_live_judge_real_verdict() -> None:
    from fastaiagent.eval import evaluate
    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.llm import LLMClient

    judge = LLMJudge(
        name="correctness", llm=LLMClient(provider="openai", model=MODEL, temperature=0)
    )
    results = evaluate(
        lambda q: "Paris is the capital of France.",
        [{"input": "Capital of France?", "expected": "Paris"}],
        scorers=[judge],
        persist=False,
    )
    v = results.cases[0].per_scorer["correctness"]
    assert v["passed"] is True and "errored" not in v, v


@requires_openai
def test_live_cli_judge_outage_exits_3(tmp_path: Path) -> None:
    """A judge whose key is rejected: the CLI says INVALID (3), not FAILED (1)."""
    (tmp_path / "target.py").write_text("def echo(x):\n    return 'The order shipped.'\n")
    (tmp_path / "cases.jsonl").write_text(json.dumps({"input": "Where is A1?"}) + "\n")
    env = {**os.environ, "OPENAI_API_KEY": "sk-invalid-key-for-the-outage-test-000000"}
    env.pop("FASTAIAGENT_API_KEY", None)
    env.pop("FASTAIAGENT_TARGET", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastaiagent.cli.main",
            "eval",
            "run",
            "--agent",
            f"{tmp_path / 'target.py'}:echo",
            "--dataset",
            str(tmp_path / "cases.jsonl"),
            "--scorers",
            "task_completion",
            "--fail-under",
            "task_completion.avg_score=0.5",
            "--db",
            str(tmp_path / "local.db"),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 3, out
    assert "sk-invalid-key-for-the-outage-test-000000" not in out


@requires_openai
def test_live_simulation_judge(tmp_path: Path) -> None:
    from fastaiagent.eval.llm_judge import LLMJudge
    from fastaiagent.eval.simulate import Scenario, SimulatedUser, simulate
    from fastaiagent.llm import LLMClient

    scenario = Scenario(
        name="order-status",
        user=SimulatedUser(script=["Where is order A17?"]),
        success_criteria=["The assistant states how the order shipped."],
        failure_criteria=["The assistant refuses to help."],
        max_turns=2,
    )
    live = LLMJudge(llm=LLMClient(provider="openai", model=MODEL, temperature=0))
    ok = simulate(scenario, _agent(), judge=live, persist=False).results[0]
    assert ok.error is None and not any(v.errored for v in ok.verdicts), ok.verdicts

    down = LLMJudge(llm=LLMClient(provider="openai", model="gpt-does-not-exist-xyz", max_retries=0))
    bad = simulate(scenario, _agent(), judge=down, persist=False).results[0]
    assert bad.passed is False and bad.error and bad.error.startswith("judge could not run")
    assert all(v.errored for v in bad.verdicts)


_PYTEST_SUITE = """
import os
os.environ["FASTAIAGENT_LOCAL_DB"] = r"{db}"
from fastaiagent._internal.config import reset_config
reset_config()

from fastaiagent.eval import case
from fastaiagent.eval.llm_judge import LLMJudge
from fastaiagent.llm import LLMClient

good = LLMJudge(name="correctness", llm=LLMClient(provider="openai", model="{model}"))
down = LLMJudge(
    name="correctness",
    llm=LLMClient(provider="openai", model="gpt-does-not-exist-xyz", max_retries=0),
)

@case(input="Capital of France?", expected="Paris")
def test_live(evaluate_one):
    evaluate_one(lambda q: "Paris", scorers=[good])

@case(input="Capital of Italy?", expected="Rome")
def test_outage(evaluate_one):
    evaluate_one(lambda q: "Rome", scorers=[down])
"""


@requires_openai
def test_live_pytest_plugin_records_the_outage_case(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "local.db"
    suite = tmp_path / "test_live_suite.py"
    suite.write_text(_PYTEST_SUITE.format(db=db, model=MODEL))
    env = {k: v for k, v in os.environ.items() if k not in ("FASTAIAGENT_API_KEY",)}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(suite)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=300,
    )
    assert "1 failed, 1 passed" in proc.stdout, proc.stdout + proc.stderr
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT case_count, case_pass_count, case_error_count, check_error_count FROM eval_runs"
        ).fetchone()
    assert row == (2, 1, 1, 1), row
