"""End-to-end tests for the simulation UI endpoints.

Real FastAPI + real SQLite. The DB is seeded by running a deterministic
``simulate()`` with ``TestModel`` (no hand-written SQL, no mocks), then the
list + detail endpoints are exercised over that real ``local.db``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.agent.agent import Agent  # noqa: E402
from fastaiagent.eval.llm_judge import LLMJudge  # noqa: E402
from fastaiagent.eval.simulate import Scenario, SimulatedUser, simulate  # noqa: E402
from fastaiagent.testing.models import TestModel  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _seed(db_path: Path) -> str:
    """Run a deterministic simulation and persist it; return the run_id."""
    agent = Agent(name="seed-bot", llm=TestModel(response="canned reply"))
    judge = LLMJudge(llm=TestModel(response=json.dumps({"score": 1.0, "reasoning": "ok"})))
    scenarios = [
        Scenario(
            name="happy-path",
            user=SimulatedUser(script=["hello", "thanks"]),
            success_criteria=["The agent replied politely."],
            max_turns=6,
        ),
        Scenario(
            name="edge-case",
            user=SimulatedUser(script=["weird input"]),
            success_criteria=["The agent handled it."],
            failure_criteria=["The agent crashed."],
            max_turns=4,
        ),
    ]
    results = simulate(scenarios, agent, judge=judge, persist=False)
    return results.persist_local(db_path=db_path, run_name="ui-seed")


@pytest.fixture
def app_db(temp_dir: Path):
    fa_dir = temp_dir / ".fastaiagent"
    fa_dir.mkdir(parents=True, exist_ok=True)
    db_path = fa_dir / "local.db"
    init_local_db(db_path).close()
    run_id = _seed(db_path)
    app = build_app(db_path=str(db_path), no_auth=True)
    return app, run_id


@pytest.fixture
def client(app_db):
    app, _ = app_db
    return TestClient(app)


def test_list_simulations(client: TestClient, app_db) -> None:
    _, run_id = app_db
    r = client.get("/api/simulations")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total"] == 1
    row = body["rows"][0]
    assert row["run_id"] == run_id
    assert row["run_name"] == "ui-seed"
    assert row["scenario_count"] == 2
    assert row["agent_name"] == "seed-bot"
    assert 0.0 <= row["pass_rate"] <= 1.0


def test_get_simulation_detail(client: TestClient, app_db) -> None:
    _, run_id = app_db
    r = client.get(f"/api/simulations/{run_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run"]["run_id"] == run_id
    assert body["total_cases"] == 2

    cases = body["cases"]
    names = {c["scenario_name"] for c in cases}
    assert names == {"happy-path", "edge-case"}

    happy = next(c for c in cases if c["scenario_name"] == "happy-path")
    # Transcript parsed from JSON into a list of turn dicts.
    assert isinstance(happy["transcript"], list)
    assert happy["transcript"][0]["role"] == "user"
    assert happy["transcript"][0]["content"] == "hello"
    # Criteria + per-criterion parsed too.
    assert happy["criteria"]["success"] == ["The agent replied politely."]
    assert isinstance(happy["per_criterion"], list)
    assert happy["per_criterion"][0]["kind"] == "success"


def test_detail_outcome_filter(client: TestClient, app_db) -> None:
    _, run_id = app_db
    r = client.get(f"/api/simulations/{run_id}", params={"outcome": "passed"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert all(c["passed"] == 1 for c in body["cases"])
    # total_cases reflects the unfiltered count.
    assert body["total_cases"] == 2


def test_get_missing_run_404(client: TestClient) -> None:
    r = client.get("/api/simulations/does-not-exist")
    assert r.status_code == 404
