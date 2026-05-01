"""E2E tests for the cost breakdown endpoint.

Hand-seeds a real SQLite DB with spans across two models and one chain
with two named nodes, then asserts the three group_by modes return the
expected aggregates. No mocking — real DB, real FastAPI app.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.pricing import compute_cost_usd  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def cost_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    now = datetime.now(tz=timezone.utc)
    with SQLiteHelper(db_path) as db:
        # Two LLM spans with gpt-4o + gpt-4o-mini, both under researcher
        # and inside the support-flow chain.
        for i, (model, in_t, out_t, node) in enumerate(
            [
                ("gpt-4o", 1000, 200, "research"),
                ("gpt-4o-mini", 500, 100, "process"),
                ("gpt-4o-mini", 800, 150, "research"),
            ]
        ):
            attrs = {
                "gen_ai.request.model": model,
                "gen_ai.usage.input_tokens": in_t,
                "gen_ai.usage.output_tokens": out_t,
                "agent.name": "researcher",
                "chain.name": "support-flow",
                "chain.node_id": node,
            }
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
                (
                    f"sp-{i}",
                    f"trace-{i}",
                    "root-span",
                    f"llm.openai.{model}",
                    _iso(now - timedelta(minutes=i)),
                    _iso(now - timedelta(minutes=i, seconds=-2)),
                    json.dumps(attrs),
                ),
            )
        # One root agent span for the agent breakdown to count runs.
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]')""",
            (
                "root-span",
                "trace-root",
                "agent.researcher",
                _iso(now - timedelta(minutes=10)),
                _iso(now - timedelta(minutes=10, seconds=-3)),
                json.dumps({"agent.name": "researcher"}),
            ),
        )
    return db_path


@pytest.fixture
def client(cost_db: Path) -> TestClient:
    return TestClient(build_app(db_path=str(cost_db), no_auth=True))


def test_cost_breakdown_by_model(client: TestClient) -> None:
    r = client.get("/api/analytics/costs?group_by=model&period=30d")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["group_by"] == "model"
    by_model = {row["model"]: row for row in body["rows"]}
    assert "gpt-4o" in by_model
    assert "gpt-4o-mini" in by_model

    # Token roll-ups match the seed values.
    assert by_model["gpt-4o"]["calls"] == 1
    assert by_model["gpt-4o"]["input_tokens"] == 1000
    assert by_model["gpt-4o"]["output_tokens"] == 200

    assert by_model["gpt-4o-mini"]["calls"] == 2
    assert by_model["gpt-4o-mini"]["input_tokens"] == 1300
    assert by_model["gpt-4o-mini"]["output_tokens"] == 250

    # Cost matches compute_cost_usd, no drift.
    expected_4o = compute_cost_usd("gpt-4o", 1000, 200)
    assert abs(by_model["gpt-4o"]["cost_usd"] - expected_4o) < 1e-6


def test_cost_breakdown_by_agent(client: TestClient) -> None:
    r = client.get("/api/analytics/costs?group_by=agent")
    assert r.status_code == 200
    by_agent = {row["agent"]: row for row in r.json()["rows"]}
    assert "researcher" in by_agent
    # 1 root agent span + 3 LLM children → runs=1, total cost > 0.
    assert by_agent["researcher"]["runs"] == 1
    assert by_agent["researcher"]["total_cost_usd"] > 0


def test_cost_breakdown_by_node_requires_chain(client: TestClient) -> None:
    r = client.get("/api/analytics/costs?group_by=node")
    assert r.status_code == 400
    assert "chain_name" in r.text


def test_cost_breakdown_by_node(client: TestClient) -> None:
    r = client.get(
        "/api/analytics/costs?group_by=node&chain_name=support-flow&period=30d"
    )
    assert r.status_code == 200
    by_node = {row["node"]: row for row in r.json()["rows"]}
    assert {"research", "process"} <= set(by_node.keys())
    # research has 2 LLM calls, process has 1 → research has more executions.
    assert by_node["research"]["executions"] == 2
    assert by_node["process"]["executions"] == 1
    # percent_of_total adds to ~100.
    total_pct = sum(r["percent_of_total"] for r in r.json()["rows"])
    assert abs(total_pct - 100.0) < 1.0


def test_cost_breakdown_invalid_group_by(client: TestClient) -> None:
    r = client.get("/api/analytics/costs?group_by=lol")
    assert r.status_code == 422  # FastAPI Query pattern enforcement
