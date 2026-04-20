"""Integration tests for analytics, scores, threads, and trace deletion."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.pricing import compute_cost_usd  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _seed(db_path: Path) -> None:
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc)
    try:
        for i, (trace_id, thread_id, status, dur_ms, cost, tokens, when_ago) in enumerate(
            [
                ("t-fast-ok", "thread-1", "OK", 300.0, 0.0015, 200, 30),
                ("t-slow-ok", "thread-1", "OK", 5000.0, 0.0200, 1200, 60),
                ("t-error", None, "ERROR", 1000.0, 0.0030, 400, 90),
            ]
        ):
            start = now - timedelta(minutes=when_ago)
            end = start + timedelta(milliseconds=dur_ms)
            attrs = {
                "agent.name": "probe-agent" if i < 2 else "flaky-agent",
                "fastaiagent.cost.total_usd": cost,
                "gen_ai.usage.input_tokens": tokens // 2,
                "gen_ai.usage.output_tokens": tokens // 2,
                "gen_ai.request.model": "gpt-4o-mini",
            }
            if thread_id:
                attrs["fastaiagent.thread.id"] = thread_id
            db.execute(
                """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                                       start_time, end_time, status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]')""",
                (
                    f"s-root-{trace_id}",
                    trace_id,
                    None,
                    "agent.root",
                    start.isoformat(),
                    end.isoformat(),
                    status,
                    json.dumps(attrs),
                ),
            )

        # Guardrail event tied to the failing trace
        db.execute(
            """INSERT INTO guardrail_events
               (event_id, trace_id, span_id, guardrail_name, guardrail_type,
                position, outcome, score, message, agent_name, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                "t-error",
                f"s-root-t-error",
                "no_pii",
                "regex",
                "output",
                "blocked",
                0.0,
                "SSN pattern matched",
                "flaky-agent",
                now.isoformat(),
                "{}",
            ),
        )

        # Eval run + case pointing at the healthy trace
        run_id = "run-analytics-fixture"
        db.execute(
            """INSERT INTO eval_runs
               (run_id, run_name, dataset_name, agent_name, agent_version,
                scorers, started_at, finished_at, pass_count, fail_count,
                pass_rate, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                "probe-smoke",
                "probe.jsonl",
                "probe-agent",
                "v1",
                json.dumps(["exact_match"]),
                now.isoformat(),
                now.isoformat(),
                1,
                0,
                1.0,
                "{}",
            ),
        )
        db.execute(
            """INSERT INTO eval_cases
               (case_id, run_id, ordinal, input, expected_output, actual_output,
                trace_id, per_scorer)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "case-linked",
                run_id,
                0,
                json.dumps("hi"),
                json.dumps("HI"),
                json.dumps("HI"),
                "t-fast-ok",
                json.dumps(
                    {"exact_match": {"passed": True, "score": 1.0, "reason": None}}
                ),
            ),
        )
    finally:
        db.close()


@pytest.fixture
def client(temp_dir: Path) -> TestClient:
    db = temp_dir / "local.db"
    _seed(db)
    app = build_app(db_path=str(db), no_auth=True)
    app.state.test_db_path = db
    return TestClient(app)


class TestPricing:
    def test_matches_longest_prefix(self):
        assert compute_cost_usd("gpt-4o-mini-2024-07-18", 1000, 2000) == pytest.approx(
            (1000 * 0.15 + 2000 * 0.60) / 1_000_000
        )

    def test_unknown_model_returns_none(self):
        assert compute_cost_usd("unknown-model-9000", 1000, 500) is None

    def test_zero_tokens_returns_none(self):
        assert compute_cost_usd("gpt-4o", 0, 0) is None


class TestAnalyticsEndpoint:
    def test_returns_summary_with_percentiles(self, client: TestClient):
        r = client.get("/api/analytics?hours=24")
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["trace_count"] == 3
        assert body["summary"]["error_count"] == 1
        assert body["summary"]["error_rate"] == pytest.approx(1 / 3)
        # Durations were 300, 1000, 5000 ms → p50 must fall in the middle bucket.
        assert body["summary"]["p50_ms"] == pytest.approx(1000.0, rel=0.01)
        assert body["summary"]["p95_ms"] >= body["summary"]["p50_ms"]

    def test_top_agents_ordered(self, client: TestClient):
        r = client.get("/api/analytics?hours=24")
        body = r.json()
        slow_names = [a["agent_name"] for a in body["top_slowest_agents"]]
        assert slow_names[0] == "probe-agent"  # 5s run outpaces the 1s error
        price_names = [a["agent_name"] for a in body["top_priciest_agents"]]
        assert price_names[0] == "probe-agent"

    def test_granularity_day(self, client: TestClient):
        r = client.get("/api/analytics?hours=24&granularity=day")
        assert r.status_code == 200
        assert r.json()["granularity"] == "day"


class TestScoresEndpoint:
    def test_returns_guardrail_events_for_trace(self, client: TestClient):
        r = client.get("/api/traces/t-error/scores")
        assert r.status_code == 200
        body = r.json()
        assert len(body["guardrail_events"]) == 1
        assert body["guardrail_events"][0]["outcome"] == "blocked"

    def test_returns_linked_eval_cases(self, client: TestClient):
        r = client.get("/api/traces/t-fast-ok/scores")
        body = r.json()
        assert len(body["eval_cases"]) == 1
        assert body["eval_cases"][0]["run_name"] == "probe-smoke"
        assert body["eval_cases"][0]["per_scorer"]["exact_match"]["passed"] is True

    def test_empty_when_nothing_links(self, client: TestClient):
        r = client.get("/api/traces/t-slow-ok/scores")
        body = r.json()
        assert body["guardrail_events"] == []
        assert body["eval_cases"] == []


class TestThreadEndpoint:
    def test_returns_traces_sharing_thread_id(self, client: TestClient):
        r = client.get("/api/threads/thread-1")
        assert r.status_code == 200
        body = r.json()
        assert {t["trace_id"] for t in body["traces"]} == {"t-fast-ok", "t-slow-ok"}

    def test_empty_for_unknown_thread(self, client: TestClient):
        r = client.get("/api/threads/missing")
        body = r.json()
        assert body["traces"] == []


class TestDeleteTrace:
    def test_delete_single_trace_cascades(self, client: TestClient):
        r = client.delete("/api/traces/t-error")
        assert r.status_code == 200
        assert r.json() == {"deleted": 1}

        # Spans gone
        with SQLiteHelper(client.app.state.test_db_path) as db:
            spans = db.fetchall("SELECT * FROM spans WHERE trace_id = 't-error'")
            events = db.fetchall(
                "SELECT * FROM guardrail_events WHERE trace_id = 't-error'"
            )
        assert spans == []
        assert events == []

        # Second delete returns 404.
        r2 = client.delete("/api/traces/t-error")
        assert r2.status_code == 404

    def test_bulk_delete(self, client: TestClient):
        r = client.post(
            "/api/traces/bulk-delete",
            json={"trace_ids": ["t-fast-ok", "t-slow-ok", "missing-id"]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["requested"] == 3
        assert body["deleted"] == 2

    def test_eval_cases_are_detached_not_deleted(self, client: TestClient):
        client.delete("/api/traces/t-fast-ok")
        with SQLiteHelper(client.app.state.test_db_path) as db:
            rows = db.fetchall(
                "SELECT case_id, trace_id FROM eval_cases WHERE case_id = 'case-linked'"
            )
        assert len(rows) == 1
        assert rows[0]["trace_id"] is None
