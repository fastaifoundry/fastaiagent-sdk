"""End-to-end tests for ``GET /api/traces/compare``.

Real FastAPI + real SQLite. No mocks — per the no-mocking rule, the
endpoint is exercised against a real seeded ``local.db`` through the
test client.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _insert_span(
    db,
    *,
    trace_id: str,
    span_id: str,
    name: str,
    start: datetime,
    end: datetime,
    parent_span_id: str | None = None,
    status: str = "OK",
    attributes: dict | None = None,
    events: list | None = None,
    project_id: str = "",
) -> None:
    db.execute(
        """INSERT INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events, project_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            span_id,
            trace_id,
            parent_span_id,
            name,
            _iso(start),
            _iso(end),
            status,
            json.dumps(attributes or {}),
            json.dumps(events or []),
            project_id,
        ),
    )


def _seed_trace(
    db,
    *,
    trace_id: str,
    started: datetime,
    spans: list[tuple[str, int, dict | None]],
    project_id: str = "",
) -> None:
    """Helper: drop a root span + child spans into ``spans``.

    ``spans`` is a list of ``(span_name, duration_ms, attributes)`` tuples.
    The first entry is the root; later entries are children of the root.
    """
    cursor = started
    root_span_id: str | None = None
    for i, (name, duration_ms, attrs) in enumerate(spans):
        span_id = f"{trace_id}-s{i}"
        if i == 0:
            root_span_id = span_id
        end = cursor + timedelta(milliseconds=duration_ms)
        _insert_span(
            db,
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            start=cursor,
            end=end,
            parent_span_id=None if i == 0 else root_span_id,
            attributes=attrs,
            project_id=project_id,
        )
        cursor = end


@pytest.fixture
def app_db(temp_dir: Path):
    """Build a no-auth FastAPI app pointing at a fresh local.db."""
    db_path = temp_dir / "local.db"
    db = init_local_db(db_path)
    db.close()
    app = build_app(db_path=str(db_path), no_auth=True)
    return app, db_path


@pytest.fixture
def client(app_db):
    app, _ = app_db
    return TestClient(app)


def _open(db_path: Path):
    return init_local_db(db_path)


# ---------------------------------------------------------------------------
# 1. Identical traces — alignment is all "same", deltas are zero.
# ---------------------------------------------------------------------------


class TestIdenticalTraces:
    def test_alignment_all_same_deltas_zero(
        self, client: TestClient, app_db
    ) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            t0 = datetime.now(tz=timezone.utc)
            shared_spans: list[tuple[str, int, dict | None]] = [
                ("agent.demo", 100, {"agent.name": "demo"}),
                ("retrieval.support-kb", 50, None),
                (
                    "llm.openai.gpt-4o",
                    200,
                    {
                        "gen_ai.usage.input_tokens": 30,
                        "gen_ai.usage.output_tokens": 20,
                        "gen_ai.response.text": "ok",
                    },
                ),
            ]
            _seed_trace(db, trace_id="trace-a", started=t0, spans=shared_spans)
            _seed_trace(
                db,
                trace_id="trace-b",
                started=t0 + timedelta(minutes=12),
                spans=shared_spans,
            )
        finally:
            db.close()

        r = client.get("/api/traces/compare", params={"a": "trace-a", "b": "trace-b"})
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["trace_a"]["trace_id"] == "trace-a"
        assert body["trace_b"]["trace_id"] == "trace-b"
        assert len(body["alignment"]) == 3
        assert all(row["match"] == "same" for row in body["alignment"])
        # Both traces have the same shape so every delta is zero.
        assert body["summary"]["spans_delta"] == 0
        assert body["summary"]["duration_delta_ms"] == 0
        # 12 minutes apart → 720s
        assert body["summary"]["time_apart_seconds"] == pytest.approx(720, abs=1)


# ---------------------------------------------------------------------------
# 2. Trace B has an extra span — emits ``new_in_b``.
# ---------------------------------------------------------------------------


class TestExtraSpanInB:
    def test_extra_span_marked_new_in_b(self, client: TestClient, app_db) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            t0 = datetime.now(tz=timezone.utc)
            spans_a = [
                ("agent.demo", 100, {"agent.name": "demo"}),
                ("retrieval.kb", 50, None),
                ("llm.gpt", 200, None),
            ]
            spans_b = spans_a + [("tool.validate_input", 30, None)]
            _seed_trace(db, trace_id="ta", started=t0, spans=spans_a)
            _seed_trace(db, trace_id="tb", started=t0, spans=spans_b)
        finally:
            db.close()

        r = client.get("/api/traces/compare", params={"a": "ta", "b": "tb"})
        assert r.status_code == 200
        rows = r.json()["alignment"]
        assert len(rows) == 4
        assert [r["match"] for r in rows[:3]] == ["same", "same", "same"]
        assert rows[3]["match"] == "new_in_b"
        assert rows[3]["span_a"] is None
        assert rows[3]["span_b"]["name"] == "tool.validate_input"
        assert r.json()["summary"]["spans_delta"] == 1


# ---------------------------------------------------------------------------
# 3. Significant duration regression — match = "slower".
# ---------------------------------------------------------------------------


class TestSlowerSpan:
    def test_llm_span_slower_in_b(self, client: TestClient, app_db) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            t0 = datetime.now(tz=timezone.utc)
            _seed_trace(
                db,
                trace_id="fast",
                started=t0,
                spans=[
                    ("agent.demo", 100, None),
                    ("llm.gpt", 1000, None),
                ],
            )
            _seed_trace(
                db,
                trace_id="slow",
                started=t0,
                spans=[
                    ("agent.demo", 100, None),
                    ("llm.gpt", 1700, None),  # +700ms — over the 500ms threshold
                ],
            )
        finally:
            db.close()

        body = client.get(
            "/api/traces/compare", params={"a": "fast", "b": "slow"}
        ).json()
        rows = body["alignment"]
        slower_row = next(r for r in rows if r["span_a"]["name"] == "llm.gpt")
        assert slower_row["match"] == "slower"
        assert slower_row["delta_ms"] == 700


# ---------------------------------------------------------------------------
# 4. Different output text — match = "different_output".
# ---------------------------------------------------------------------------


class TestDifferentOutput:
    def test_response_text_differs(self, client: TestClient, app_db) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            t0 = datetime.now(tz=timezone.utc)
            _seed_trace(
                db,
                trace_id="terse",
                started=t0,
                spans=[
                    ("agent.demo", 100, None),
                    (
                        "llm.gpt",
                        200,
                        {"gen_ai.response.text": "Refunds in 14 days."},
                    ),
                ],
            )
            _seed_trace(
                db,
                trace_id="verbose",
                started=t0,
                spans=[
                    ("agent.demo", 100, None),
                    (
                        "llm.gpt",
                        200,
                        {
                            "gen_ai.response.text": (
                                "We process refunds within 7 business days."
                            )
                        },
                    ),
                ],
            )
        finally:
            db.close()

        body = client.get(
            "/api/traces/compare", params={"a": "terse", "b": "verbose"}
        ).json()
        rows = body["alignment"]
        llm_row = next(r for r in rows if r["span_a"]["name"] == "llm.gpt")
        assert llm_row["match"] == "different_output"


# ---------------------------------------------------------------------------
# 5. Project scoping — trace from another project is invisible (404).
# ---------------------------------------------------------------------------


class TestProjectScoping:
    def test_other_project_trace_returns_404(
        self, temp_dir: Path
    ) -> None:
        db_path = temp_dir / "local.db"
        db = init_local_db(db_path)
        try:
            t0 = datetime.now(tz=timezone.utc)
            _seed_trace(
                db,
                trace_id="t-p1",
                started=t0,
                spans=[("agent.x", 100, None)],
                project_id="p1",
            )
            _seed_trace(
                db,
                trace_id="t-p2",
                started=t0,
                spans=[("agent.y", 100, None)],
                project_id="p2",
            )
        finally:
            db.close()

        # Build app scoped to project p1 — t-p2 must be 404 from this scope.
        app = build_app(db_path=str(db_path), no_auth=True, project_id="p1")
        client = TestClient(app)

        # Both inside p1 → trivially fine: only one trace exists; pair
        # against itself just to prove the route works.
        same = client.get(
            "/api/traces/compare", params={"a": "t-p1", "b": "t-p1"}
        )
        assert same.status_code == 200

        cross = client.get(
            "/api/traces/compare", params={"a": "t-p1", "b": "t-p2"}
        )
        assert cross.status_code == 404


# ---------------------------------------------------------------------------
# 6. Summary deltas — cost / tokens / duration computed via real
#    ``_summarize_trace``.
# ---------------------------------------------------------------------------


class TestSummaryDeltas:
    def test_cost_tokens_duration_deltas_are_real(
        self, client: TestClient, app_db
    ) -> None:
        _, db_path = app_db
        db = _open(db_path)
        try:
            t0 = datetime.now(tz=timezone.utc)
            _seed_trace(
                db,
                trace_id="cheap",
                started=t0,
                spans=[
                    (
                        "agent.demo",
                        500,
                        {
                            "agent.name": "demo",
                            "fastaiagent.cost.total_usd": 0.01,
                            "gen_ai.usage.input_tokens": 100,
                            "gen_ai.usage.output_tokens": 50,
                        },
                    ),
                ],
            )
            _seed_trace(
                db,
                trace_id="pricey",
                started=t0,
                spans=[
                    (
                        "agent.demo",
                        1200,
                        {
                            "agent.name": "demo",
                            "fastaiagent.cost.total_usd": 0.07,
                            "gen_ai.usage.input_tokens": 250,
                            "gen_ai.usage.output_tokens": 175,
                        },
                    ),
                ],
            )
        finally:
            db.close()

        body = client.get(
            "/api/traces/compare", params={"a": "cheap", "b": "pricey"}
        ).json()
        s = body["summary"]
        assert s["duration_delta_ms"] == 700
        assert s["tokens_delta"] == 275  # (250+175) - (100+50)
        assert s["cost_delta_usd"] == pytest.approx(0.06, abs=1e-9)
        assert s["spans_delta"] == 0


# ---------------------------------------------------------------------------
# 7. Missing trace → 404 with a useful detail.
# ---------------------------------------------------------------------------


class TestMissingTrace:
    def test_unknown_trace_id_returns_404(
        self, client: TestClient
    ) -> None:
        r = client.get(
            "/api/traces/compare", params={"a": "no-such", "b": "also-no"}
        )
        assert r.status_code == 404
        assert "no-such" in r.json()["detail"]
