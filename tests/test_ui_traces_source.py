"""The Traces list must surface and filter on ``fastaiagent.source``.

``playground.py`` has always claimed "playground traces are filterable in the
Traces page", but the API never exposed the attribute: ``TraceRow`` carried
``framework`` and nothing else, so a Playground experiment was
indistinguishable from production traffic in the list.

Hand-seeded spans, real FastAPI app, no network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    now = datetime.now(tz=timezone.utc)
    seed = [
        ("pg-1", "playground.run", {"fastaiagent.source": "playground"}),
        ("pg-2", "playground.run", {"fastaiagent.source": "playground"}),
        ("prod-1", "agent.support", {"agent.name": "support"}),
        ("other-1", "agent.batch", {"fastaiagent.source": "batch-job"}),
    ]
    with SQLiteHelper(db_path) as db:
        for i, (trace_id, name, attrs) in enumerate(seed):
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time,
                    end_time, status, attributes, events)
                   VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]')""",
                (
                    f"root-{i}",
                    trace_id,
                    name,
                    _iso(now - timedelta(minutes=i)),
                    _iso(now - timedelta(minutes=i, seconds=-1)),
                    json.dumps(attrs),
                ),
            )
    return TestClient(build_app(db_path=str(db_path), no_auth=True))


def rows(client: TestClient, query: str = "") -> list[dict]:
    r = client.get(f"/api/traces?limit=50{query}")
    assert r.status_code == 200, r.text
    body = r.json()
    return body["rows"] if isinstance(body, dict) else body


class TestSourceIsSurfaced:
    def test_playground_traces_carry_their_source(self, client: TestClient) -> None:
        by_id = {r["trace_id"]: r for r in rows(client)}
        assert by_id["pg-1"]["source"] == "playground"
        assert by_id["pg-2"]["source"] == "playground"

    def test_traces_without_the_attribute_report_none(
        self, client: TestClient
    ) -> None:
        by_id = {r["trace_id"]: r for r in rows(client)}
        assert by_id["prod-1"]["source"] is None

    def test_source_is_free_text_not_an_enum(self, client: TestClient) -> None:
        """Any value a caller stamps survives — this isn't playground-only."""
        by_id = {r["trace_id"]: r for r in rows(client)}
        assert by_id["other-1"]["source"] == "batch-job"


class TestSourceFilter:
    def test_filters_to_matching_traces(self, client: TestClient) -> None:
        got = {r["trace_id"] for r in rows(client, "&source=playground")}
        assert got == {"pg-1", "pg-2"}

    def test_non_matching_source_returns_nothing(self, client: TestClient) -> None:
        assert rows(client, "&source=nope") == []

    def test_absent_filter_returns_everything(self, client: TestClient) -> None:
        assert len(rows(client)) == 4

    @pytest.mark.parametrize("bad", ["%", "a%b", "'; DROP TABLE spans--", "-leading"])
    def test_malformed_source_is_rejected_not_interpolated(
        self, client: TestClient, bad: str
    ) -> None:
        """Same pattern guard as ``framework`` — no LIKE wildcards get through."""
        r = client.get("/api/traces", params={"source": bad})
        assert r.status_code == 422, r.text
