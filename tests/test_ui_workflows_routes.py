"""Route tests for the /api/workflows directory.

Seeds chain/swarm/supervisor root spans into a temp local.db, hits the
FastAPI app via TestClient. No mocks of the subject under test — real
SQLite, real router wiring.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _insert_root(
    db: SQLiteHelper,
    trace_id: str,
    runner_type: str,
    workflow_name: str,
    status: str = "OK",
    ago_minutes: int = 5,
    node_count: int = 3,
    cost_usd: float | None = None,
) -> None:
    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(minutes=ago_minutes)).isoformat()
    end = now.isoformat()
    attrs: dict[str, object] = {
        f"{runner_type}.name": workflow_name,
        "fastaiagent.runner.type": runner_type,
        f"{runner_type}.node_count": node_count,
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 50,
        "gen_ai.request.model": "gpt-4o-mini",
    }
    if cost_usd is not None:
        attrs["fastaiagent.cost.total_usd"] = cost_usd
    db.execute(
        """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                               start_time, end_time, status, attributes, events)
           VALUES (?, ?, NULL, ?, ?, ?, ?, ?, '[]')""",
        (
            f"s-{trace_id}",
            trace_id,
            f"{runner_type}.{workflow_name}",
            start,
            end,
            status,
            json.dumps(attrs),
        ),
    )


@pytest.fixture
def app_env(tmp_path):
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    with SQLiteHelper(db_path) as db:
        _insert_root(db, "t-chain-1", "chain", "support-flow", "OK", 5)
        _insert_root(db, "t-chain-2", "chain", "support-flow", "ERROR", 10)
        _insert_root(db, "t-chain-3", "chain", "billing-flow", "OK", 15)
        _insert_root(db, "t-swarm-1", "swarm", "research-team", "OK", 20)
        _insert_root(db, "t-sup-1", "supervisor", "triage", "OK", 25)
        # Agent-only roots must be excluded from the workflow directory.
        db.execute(
            """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                                   start_time, end_time, status, attributes, events)
               VALUES ('s-agent-only', 't-agent-1', NULL, 'agent.lonely',
                       '2026-04-21T00:00:00+00:00', '2026-04-21T00:00:01+00:00',
                       'OK', '{"agent.name":"lonely"}', '[]')"""
        )
    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
    )
    return app, db_path


def test_list_all_workflows(app_env):
    app, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/workflows")
    assert r.status_code == 200
    names = {(w["runner_type"], w["workflow_name"]) for w in r.json()["workflows"]}
    assert names == {
        ("chain", "support-flow"),
        ("chain", "billing-flow"),
        ("swarm", "research-team"),
        ("supervisor", "triage"),
    }
    # Agent-only roots shouldn't leak in.
    assert not any(w["runner_type"] == "agent" for w in r.json()["workflows"])


def test_list_filter_by_runner_type(app_env):
    app, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/workflows?runner_type=chain")
    assert r.status_code == 200
    rows = r.json()["workflows"]
    assert {w["workflow_name"] for w in rows} == {"support-flow", "billing-flow"}


def test_list_rejects_bad_runner_type(app_env):
    app, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/workflows?runner_type=nope")
    assert r.status_code == 400


def test_list_includes_registered_runners_without_spans(tmp_path):
    """Regression: a runner registered via build_app(runners=[...]) must
    appear in /api/workflows even before it has produced any spans —
    otherwise the topology view for it is undiscoverable.
    """
    from fastaiagent import Agent, LLMClient
    from fastaiagent.agent.swarm import Swarm
    from fastaiagent.agent.team import Supervisor, Worker

    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()

    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    researcher = Agent(name="researcher", llm=llm)
    writer = Agent(name="writer", llm=llm)
    supervisor = Supervisor(
        name="planner-2",
        llm=llm,
        workers=[
            Worker(agent=researcher, role="researcher"),
            Worker(agent=writer, role="writer"),
        ],
    )
    swarm = Swarm(
        name="customer-router",
        agents=[Agent(name="triage", llm=llm), Agent(name="billing", llm=llm)],
        entrypoint="triage",
    )
    app = build_app(db_path=str(db_path), no_auth=True, runners=[supervisor, swarm])
    with TestClient(app) as client:
        r = client.get("/api/workflows")
    assert r.status_code == 200
    body = r.json()
    pairs = {(w["runner_type"], w["workflow_name"]) for w in body["workflows"]}
    assert ("supervisor", "planner-2") in pairs
    assert ("swarm", "customer-router") in pairs
    # Stub stats are zero (no spans yet) — verify the supervisor's
    # node_count was discovered from the runner.
    sup = next(
        w for w in body["workflows"] if w["workflow_name"] == "planner-2"
    )
    assert sup["run_count"] == 0
    assert sup["node_count"] == 2  # 2 workers


def test_list_filter_includes_registered(tmp_path):
    """The runner_type query param should also apply to registered stubs."""
    from fastaiagent import Agent, LLMClient
    from fastaiagent.agent.swarm import Swarm

    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()

    llm = LLMClient(provider="openai", model="gpt-4o-mini")
    swarm = Swarm(
        name="only-swarm",
        agents=[Agent(name="a", llm=llm), Agent(name="b", llm=llm)],
        entrypoint="a",
    )
    app = build_app(db_path=str(db_path), no_auth=True, runners=[swarm])
    with TestClient(app) as client:
        # Filtered to chain → swarm-registered runner should NOT appear.
        r_chain = client.get("/api/workflows?runner_type=chain")
        # Filtered to swarm → it should appear.
        r_swarm = client.get("/api/workflows?runner_type=swarm")
    assert r_chain.json()["workflows"] == []
    swarm_names = {w["workflow_name"] for w in r_swarm.json()["workflows"]}
    assert "only-swarm" in swarm_names


def test_detail_computes_success_rate(app_env):
    app, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/workflows/chain/support-flow")
    assert r.status_code == 200
    body = r.json()
    assert body["runner_type"] == "chain"
    assert body["workflow_name"] == "support-flow"
    assert body["run_count"] == 2
    assert body["error_count"] == 1
    assert body["success_rate"] == pytest.approx(0.5)
    assert body["node_count"] == 3


def test_detail_404_for_unknown_workflow(app_env):
    app, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/workflows/chain/ghost")
    assert r.status_code == 404


def test_detail_400_for_bad_runner_type(app_env):
    app, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/workflows/retrieval/nope")
    assert r.status_code == 400


def test_trace_list_runner_name_drill_down(app_env):
    """A workflow detail page links to /traces?runner_type=X&runner_name=Y."""
    app, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/traces?runner_type=chain&runner_name=support-flow")
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert len(rows) == 2
    assert all(row["runner_type"] == "chain" for row in rows)
    assert all(row["runner_name"] == "support-flow" for row in rows)
