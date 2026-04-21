"""Enhanced evals route tests.

Covers the 0.9.3 additions:
  - list returns per-run cost_usd + avg_latency_ms aggregates
  - run detail supports scorer/outcome/q filters + scorer_summary
  - compare returns regressed / improved / unchanged counts and
    per-scorer deltas, matching cases by ordinal

No mocks: real SQLite, real FastAPI TestClient, real aggregator code.
"""

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
from fastaiagent.ui.server import build_app  # noqa: E402


def _insert_trace_root(
    db: SQLiteHelper,
    trace_id: str,
    cost_usd: float,
    latency_ms: int,
    *,
    agent_name: str = "test-agent",
) -> None:
    now = datetime.now(tz=timezone.utc)
    end = now
    start = now - timedelta(milliseconds=latency_ms)
    attrs = {
        "agent.name": agent_name,
        "fastaiagent.cost.total_usd": cost_usd,
    }
    db.execute(
        """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                               start_time, end_time, status, attributes, events)
           VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]')""",
        (
            f"s-{trace_id}",
            trace_id,
            f"agent.{agent_name}",
            start.isoformat(),
            end.isoformat(),
            json.dumps(attrs),
        ),
    )


def _insert_run(
    db: SQLiteHelper,
    *,
    run_id: str,
    run_name: str,
    dataset: str,
    scorers: list[str],
    started_at: datetime,
    pass_count: int,
    fail_count: int,
) -> None:
    pr = pass_count / max(pass_count + fail_count, 1)
    db.execute(
        """INSERT INTO eval_runs
           (run_id, run_name, dataset_name, agent_name, agent_version, scorers,
            started_at, finished_at, pass_count, fail_count, pass_rate, metadata)
           VALUES (?, ?, ?, 'test-agent', 'v1', ?, ?, ?, ?, ?, ?, '{}')""",
        (
            run_id,
            run_name,
            dataset,
            json.dumps(scorers),
            started_at.isoformat(),
            started_at.isoformat(),
            pass_count,
            fail_count,
            pr,
        ),
    )


def _insert_case(
    db: SQLiteHelper,
    *,
    run_id: str,
    ordinal: int,
    input_val: str,
    expected: str,
    actual: str,
    trace_id: str | None,
    per_scorer: dict[str, dict[str, object]],
) -> None:
    db.execute(
        """INSERT INTO eval_cases
           (case_id, run_id, ordinal, input, expected_output, actual_output,
            trace_id, per_scorer)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            uuid.uuid4().hex,
            run_id,
            ordinal,
            json.dumps(input_val),
            json.dumps(expected),
            json.dumps(actual),
            trace_id,
            json.dumps(per_scorer),
        ),
    )


@pytest.fixture
def app_env(tmp_path: Path):
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    now = datetime.now(tz=timezone.utc)
    with SQLiteHelper(db_path) as db:
        # ── Run A: 2/3 pass, cheap ─────────────────────────────────────────
        _insert_run(
            db,
            run_id="run-a",
            run_name="greet-smoke-a",
            dataset="greetings.jsonl",
            scorers=["exact_match", "style"],
            started_at=now - timedelta(hours=2),
            pass_count=2,
            fail_count=1,
        )
        for i, (inp, actual, em_pass, style_pass, cost, lat) in enumerate(
            [
                ("hi", "hi", True, True, 0.001, 320),
                ("bye", "BYE", False, True, 0.0015, 400),
                ("hello", "hello", True, True, 0.001, 280),
            ]
        ):
            trace_id = f"t-a-{i}"
            _insert_trace_root(db, trace_id, cost_usd=cost, latency_ms=lat)
            _insert_case(
                db,
                run_id="run-a",
                ordinal=i,
                input_val=inp,
                expected=inp,
                actual=actual,
                trace_id=trace_id,
                per_scorer={
                    "exact_match": {"passed": em_pass, "score": 1.0 if em_pass else 0.0},
                    "style": {"passed": style_pass, "score": 1.0},
                },
            )

        # ── Run B: 3/3 pass (fixed the bye case), slightly more expensive ─
        _insert_run(
            db,
            run_id="run-b",
            run_name="greet-smoke-b",
            dataset="greetings.jsonl",
            scorers=["exact_match", "style"],
            started_at=now - timedelta(hours=1),
            pass_count=3,
            fail_count=0,
        )
        for i, (inp, actual, em_pass, style_pass, cost, lat) in enumerate(
            [
                ("hi", "hi", True, True, 0.0012, 320),
                ("bye", "bye", True, True, 0.0016, 410),
                ("hello", "hello", True, True, 0.0011, 275),
            ]
        ):
            trace_id = f"t-b-{i}"
            _insert_trace_root(db, trace_id, cost_usd=cost, latency_ms=lat)
            _insert_case(
                db,
                run_id="run-b",
                ordinal=i,
                input_val=inp,
                expected=inp,
                actual=actual,
                trace_id=trace_id,
                per_scorer={
                    "exact_match": {"passed": em_pass, "score": 1.0},
                    "style": {"passed": style_pass, "score": 1.0},
                },
            )
    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
    )
    return app


def test_list_includes_cost_and_latency_aggregates(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals")
    assert r.status_code == 200
    rows = r.json()["rows"]
    a = next(row for row in rows if row["run_id"] == "run-a")
    # Sum of (0.001 + 0.0015 + 0.001) = 0.0035
    assert a["cost_usd"] == pytest.approx(0.0035, abs=1e-6)
    # Avg of (320 + 400 + 280) = 333.33
    assert a["avg_latency_ms"] == pytest.approx(333.33, abs=1.0)
    assert a["case_count"] == 3


def test_run_detail_scorer_summary(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals/run-a")
    assert r.status_code == 200
    body = r.json()
    summary = body["run"]["scorer_summary"]
    assert summary == {
        "exact_match": {"pass": 2, "fail": 1},
        "style": {"pass": 3, "fail": 0},
    }
    assert body["total_cases"] == 3
    assert len(body["cases"]) == 3


def test_run_detail_filter_by_outcome_failed(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals/run-a?outcome=failed")
    assert r.status_code == 200
    body = r.json()
    assert body["total_cases"] == 3
    assert len(body["cases"]) == 1
    assert body["cases"][0]["actual_output"] == "BYE"


def test_run_detail_filter_by_scorer(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals/run-a?scorer=style")
    assert r.status_code == 200
    assert len(r.json()["cases"]) == 3


def test_run_detail_filter_q(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals/run-a?q=bye")
    assert r.status_code == 200
    cases = r.json()["cases"]
    assert len(cases) == 1
    assert cases[0]["input"] == "bye"


def test_compare_regressed_and_improved(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals/compare?a=run-a&b=run-b")
    assert r.status_code == 200
    body = r.json()
    # Case ordinal 1 ("bye") improved: failed on run-a, passes on run-b.
    assert len(body["improved"]) == 1
    imp = body["improved"][0]
    assert imp["a"]["input"] == "bye"
    # Scorer deltas explain *which* scorer changed.
    deltas = {d["scorer"]: d for d in imp["scorer_deltas"]}
    assert deltas["exact_match"]["changed"] is True
    assert deltas["exact_match"]["passed_before"] is False
    assert deltas["exact_match"]["passed_after"] is True
    assert deltas["style"]["changed"] is False

    # Nothing regressed — run-b is strictly better.
    assert body["regressed"] == []
    # The other two cases were pass → pass.
    assert body["unchanged_pass"] == 2
    assert body["unchanged_fail"] == 0
    # Pass-rate delta: (3/3) - (2/3) = 0.3333
    assert body["pass_rate_delta"] == pytest.approx(1 / 3, abs=1e-3)


def test_compare_404_on_missing_run(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals/compare?a=run-a&b=ghost")
    assert r.status_code == 404


def test_run_detail_404(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/evals/ghost")
    assert r.status_code == 404
