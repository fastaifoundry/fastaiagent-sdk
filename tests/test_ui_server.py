"""End-to-end API tests for the Local UI server.

Real FastAPI + real SQLite + real bcrypt + real itsdangerous cookies. No mocks.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.auth import create_auth_file  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def seeded_db(temp_dir: Path) -> Path:
    """Create local.db with one trace, one eval run, one guardrail event, one prompt."""
    db_path = temp_dir / "local.db"
    db = init_local_db(db_path)
    try:
        now = datetime.now(tz=timezone.utc).isoformat()
        trace_id = "trace-abc"
        root_span_id = "span-root"
        child_span_id = "span-child"
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                root_span_id,
                trace_id,
                None,
                "agent.example",
                now,
                now,
                "OK",
                json.dumps(
                    {
                        "fastai.agent.name": "example-agent",
                        "fastai.cost.total_usd": 0.0012,
                        "gen_ai.usage.input_tokens": 80,
                        "gen_ai.usage.output_tokens": 40,
                        "fastai.thread.id": "t-1",
                        "fastai.prompt.name": "greet",
                        "fastai.prompt.version": "1",
                    }
                ),
                "[]",
            ),
        )
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                child_span_id,
                trace_id,
                root_span_id,
                "llm.chat",
                now,
                now,
                "OK",
                "{}",
                "[]",
            ),
        )

        # Eval run + one case
        run_id = "run-1"
        db.execute(
            """INSERT INTO eval_runs
               (run_id, run_name, dataset_name, agent_name, scorers,
                started_at, finished_at, pass_count, fail_count, pass_rate, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                "smoke",
                "tiny.jsonl",
                "example-agent",
                json.dumps(["exact_match"]),
                now,
                now,
                2,
                1,
                0.6667,
                json.dumps({}),
            ),
        )
        db.execute(
            """INSERT INTO eval_cases
               (case_id, run_id, ordinal, input, expected_output,
                actual_output, trace_id, per_scorer)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "case-1",
                run_id,
                0,
                json.dumps("hello"),
                json.dumps("HELLO"),
                json.dumps("HELLO"),
                trace_id,
                json.dumps({"exact_match": {"passed": True, "score": 1.0}}),
            ),
        )

        # Guardrail event
        db.execute(
            """INSERT INTO guardrail_events
               (event_id, trace_id, span_id, guardrail_name, guardrail_type,
                position, outcome, score, message, agent_name, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                trace_id,
                root_span_id,
                "no_pii",
                "regex",
                "output",
                "passed",
                1.0,
                "clean",
                "example-agent",
                now,
                json.dumps({}),
            ),
        )

        # Prompt with one version
        db.execute(
            """INSERT INTO prompts (slug, latest_version, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            ("greet", "1", now, now),
        )
        db.execute(
            """INSERT INTO prompt_versions
               (slug, version, template, variables, fragments, metadata,
                created_at, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "greet",
                "1",
                "Hello {{name}}!",
                json.dumps(["name"]),
                json.dumps([]),
                json.dumps({}),
                now,
                "code",
            ),
        )
    finally:
        db.close()
    return db_path


@pytest.fixture
def app_no_auth(seeded_db: Path):
    app = build_app(db_path=str(seeded_db), no_auth=True)
    return app, seeded_db


@pytest.fixture
def client_no_auth(app_no_auth):
    app, _ = app_no_auth
    return TestClient(app)


@pytest.fixture
def app_with_auth(seeded_db: Path, temp_dir: Path):
    auth_path = temp_dir / "auth.json"
    create_auth_file("upendra", "correct-horse", path=auth_path)
    app = build_app(
        db_path=str(seeded_db), auth_path=auth_path, no_auth=False
    )
    return app, seeded_db, auth_path


@pytest.fixture
def client_with_auth(app_with_auth):
    app, _, _ = app_with_auth
    return TestClient(app)


class TestAuth:
    def test_status_unauthenticated_initially(self, client_with_auth):
        r = client_with_auth.get("/api/auth/status")
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] is False
        assert body["no_auth"] is False

    def test_login_success_sets_cookie(self, client_with_auth):
        r = client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "correct-horse"},
        )
        assert r.status_code == 200
        assert "fastaiagent_session" in r.cookies

        status_r = client_with_auth.get("/api/auth/status")
        assert status_r.status_code == 200
        assert status_r.json()["authenticated"] is True
        assert status_r.json()["username"] == "upendra"

    def test_login_bad_password(self, client_with_auth):
        r = client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "wrong"},
        )
        assert r.status_code == 401

    def test_logout_clears_session(self, client_with_auth):
        client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "correct-horse"},
        )
        client_with_auth.post("/api/auth/logout")
        assert client_with_auth.get("/api/auth/status").json()["authenticated"] is False

    def test_protected_route_requires_login(self, client_with_auth):
        r = client_with_auth.get("/api/traces")
        assert r.status_code == 401

    def test_no_auth_mode_bypasses_login(self, client_no_auth):
        r = client_no_auth.get("/api/traces")
        assert r.status_code == 200


class TestTraces:
    def test_list_traces(self, client_no_auth):
        r = client_no_auth.get("/api/traces")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["rows"][0]["trace_id"] == "trace-abc"
        assert body["rows"][0]["agent_name"] == "example-agent"
        assert body["rows"][0]["span_count"] == 2
        assert body["rows"][0]["total_tokens"] == 120

    def test_get_trace(self, client_no_auth):
        r = client_no_auth.get("/api/traces/trace-abc")
        assert r.status_code == 200
        body = r.json()
        assert body["agent_name"] == "example-agent"
        assert len(body["spans"]) == 2

    def test_trace_not_found(self, client_no_auth):
        r = client_no_auth.get("/api/traces/missing")
        assert r.status_code == 404

    def test_span_tree(self, client_no_auth):
        r = client_no_auth.get("/api/traces/trace-abc/spans")
        assert r.status_code == 200
        tree = r.json()["tree"]
        assert tree["span"]["span_id"] == "span-root"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["span"]["span_id"] == "span-child"

    def test_note_then_favorite(self, client_no_auth, seeded_db):
        r = client_no_auth.post(
            "/api/traces/trace-abc/notes", json={"note": "interesting"}
        )
        assert r.status_code == 200
        r2 = client_no_auth.post("/api/traces/trace-abc/favorite")
        assert r2.json() == {"favorited": True}
        r3 = client_no_auth.post("/api/traces/trace-abc/favorite")
        assert r3.json() == {"favorited": False}

    def test_filter_by_agent(self, client_no_auth):
        r = client_no_auth.get("/api/traces?agent=example-agent")
        assert r.status_code == 200
        assert r.json()["rows"][0]["agent_name"] == "example-agent"
        r2 = client_no_auth.get("/api/traces?agent=does-not-exist")
        assert r2.json()["rows"] == []


class TestOverview:
    def test_overview_aggregates(self, client_no_auth):
        r = client_no_auth.get("/api/overview")
        assert r.status_code == 200
        body = r.json()
        assert body["eval_runs_last_7d"] == 1
        assert body["traces_last_24h"] >= 1
        assert body["avg_pass_rate_last_7d"] == pytest.approx(0.6667, abs=1e-3)
        assert len(body["recent_traces"]) == 1
        assert len(body["recent_eval_runs"]) == 1


class TestEvals:
    def test_list_runs(self, client_no_auth):
        r = client_no_auth.get("/api/evals")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["rows"][0]["run_id"] == "run-1"

    def test_get_run(self, client_no_auth):
        r = client_no_auth.get("/api/evals/run-1")
        assert r.status_code == 200
        body = r.json()
        assert body["run"]["run_name"] == "smoke"
        assert len(body["cases"]) == 1
        assert body["cases"][0]["input"] == "hello"

    def test_run_not_found(self, client_no_auth):
        r = client_no_auth.get("/api/evals/nonexistent")
        assert r.status_code == 404

    def test_trend(self, client_no_auth):
        r = client_no_auth.get("/api/evals/trend")
        assert r.status_code == 200
        assert r.json()["points"][0]["pass_rate"] == pytest.approx(0.6667, abs=1e-3)


class TestPrompts:
    def test_list_prompts(self, client_no_auth):
        r = client_no_auth.get("/api/prompts")
        assert r.status_code == 200
        body = r.json()
        assert body["rows"][0]["name"] == "greet"
        assert body["rows"][0]["linked_trace_count"] == 1

    def test_get_version(self, client_no_auth):
        r = client_no_auth.get("/api/prompts/greet/versions/1")
        assert r.status_code == 200
        assert r.json()["template"] == "Hello {{name}}!"

    def test_diff(self, client_no_auth, seeded_db):
        # Add a v2 so diff has something to work with.
        with SQLiteHelper(seeded_db) as db:
            db.execute(
                """INSERT INTO prompt_versions
                   (slug, version, template, variables, fragments, metadata,
                    created_at, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                ("greet", "2", "Hi {{name}}!", "[]", "[]", "{}", "", "code"),
            )
        r = client_no_auth.get("/api/prompts/greet/diff?a=1&b=2")
        assert r.status_code == 200
        assert "Hello" in r.json()["diff"]
        assert "Hi" in r.json()["diff"]

    def test_update_creates_new_version(
        self, monkeypatch, seeded_db, temp_dir
    ):
        # The editor gate requires the DB to live inside cwd — mimic a project
        # where .fastaiagent/local.db is beneath the current directory.
        monkeypatch.chdir(temp_dir)
        from fastaiagent.ui.server import build_app

        app = build_app(db_path=str(seeded_db), no_auth=True)
        from fastapi.testclient import TestClient

        client = TestClient(app)
        r = client.put(
            "/api/prompts/greet", json={"template": "Hola {{name}}!"}
        )
        assert r.status_code == 200
        assert r.json()["version"] == 2
        with SQLiteHelper(seeded_db) as db:
            rows = db.fetchall(
                "SELECT version, template FROM prompt_versions WHERE slug='greet'"
            )
        templates = {r["version"]: r["template"] for r in rows}
        assert templates["2"] == "Hola {{name}}!"

    def test_update_rejected_when_registry_external(self, client_no_auth):
        """DB outside cwd → 403 with the documented message."""
        r = client_no_auth.put(
            "/api/prompts/greet", json={"template": "won't save"}
        )
        assert r.status_code == 403
        assert "external" in r.json()["detail"].lower()

    def test_lineage(self, client_no_auth):
        r = client_no_auth.get("/api/prompts/greet/lineage")
        assert r.status_code == 200
        assert "trace-abc" in r.json()["trace_ids"]


class TestGuardrails:
    def test_list_events(self, client_no_auth):
        r = client_no_auth.get("/api/guardrail-events")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["rows"][0]["guardrail_name"] == "no_pii"

    def test_filter_by_outcome(self, client_no_auth):
        r = client_no_auth.get("/api/guardrail-events?outcome=passed")
        assert r.json()["total"] == 1
        r2 = client_no_auth.get("/api/guardrail-events?outcome=blocked")
        assert r2.json()["total"] == 0


class TestAgents:
    def test_list_agents(self, client_no_auth):
        r = client_no_auth.get("/api/agents")
        assert r.status_code == 200
        body = r.json()
        assert body["agents"][0]["agent_name"] == "example-agent"
        assert body["agents"][0]["run_count"] == 1

    def test_get_agent(self, client_no_auth):
        r = client_no_auth.get("/api/agents/example-agent")
        assert r.status_code == 200
        assert r.json()["agent_name"] == "example-agent"

    def test_agent_404(self, client_no_auth):
        r = client_no_auth.get("/api/agents/does-not-exist")
        assert r.status_code == 404


class TestStaticFallback:
    def test_api_catch_all_returns_json_not_html(self, client_no_auth):
        r = client_no_auth.get("/api/doesnotexist")
        assert r.status_code in (404, 503)
