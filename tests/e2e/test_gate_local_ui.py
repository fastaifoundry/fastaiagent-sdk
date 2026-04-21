"""E2E gate — the Local UI server.

Boots the real FastAPI app against a temp ``local.db`` seeded with a
representative snapshot, then drives every REST endpoint with
``httpx.AsyncClient``. Catches import-time or wiring regressions the unit
suite misses (e.g. route missing, Pydantic model drift, SPA fallback
broken).

No mocking. No live API keys required — the UI is a read surface over
SQLite that ships inside the Python wheel.

Runs in CI when:
    pytest tests/e2e/ -v -m e2e

Runs locally with the same command once ``pip install -e '.[all,dev]'``
has been executed so fastapi / bcrypt / itsdangerous are present.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.auth import create_auth_file  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    """Populate local.db with one chain trace, one eval run, one guardrail event.

    Matches the shapes every endpoint expects: a chain root span + child
    agent + child LLM, a prompt + version, and a guardrail event.
    """
    db_path = tmp_path / "local.db"
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    try:
        # Chain root + two children.
        for sid, parent, name, attrs in [
            (
                "s-root",
                None,
                "chain.support",
                {
                    "chain.name": "support",
                    "chain.node_count": 2,
                    "fastaiagent.runner.type": "chain",
                    "fastaiagent.thread.id": "thread-1",
                },
            ),
            (
                "s-agent",
                "s-root",
                "agent.researcher",
                {
                    "agent.name": "researcher",
                    "agent.input": "q",
                    "agent.output": "a",
                    "agent.tokens_used": 40,
                    "agent.latency_ms": 120,
                    "fastaiagent.prompt.name": "greet",
                    "fastaiagent.prompt.version": "1",
                },
            ),
            (
                "s-llm",
                "s-agent",
                "llm.openai.gpt-4o-mini",
                {
                    "gen_ai.request.model": "gpt-4o-mini",
                    "gen_ai.usage.input_tokens": 25,
                    "gen_ai.usage.output_tokens": 15,
                },
            ),
        ]:
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]')""",
                (sid, "trace-gate", parent, name, now, now, "OK", json.dumps(attrs)),
            )

        # Prompt + version.
        db.execute(
            """INSERT INTO prompts (slug, latest_version, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            ("greet", "1", now, now),
        )
        db.execute(
            """INSERT INTO prompt_versions
               (slug, version, template, variables, fragments, metadata, created_at, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("greet", "1", "Hi {{name}}", "[]", "[]", "{}", now, "code"),
        )

        # Eval run + case linked to the trace.
        db.execute(
            """INSERT INTO eval_runs
               (run_id, run_name, dataset_name, agent_name, agent_version,
                scorers, started_at, finished_at, pass_count, fail_count,
                pass_rate, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "run-1",
                "gate",
                "d.jsonl",
                "researcher",
                "v1",
                json.dumps(["exact_match"]),
                now,
                now,
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
                "case-1",
                "run-1",
                0,
                json.dumps("q"),
                json.dumps("a"),
                json.dumps("a"),
                "trace-gate",
                json.dumps({"exact_match": {"passed": True, "score": 1.0}}),
            ),
        )

        # Guardrail event on the trace.
        db.execute(
            """INSERT INTO guardrail_events
               (event_id, trace_id, span_id, guardrail_name, guardrail_type,
                position, outcome, score, message, agent_name, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                "trace-gate",
                "s-agent",
                "no_pii",
                "regex",
                "output",
                "passed",
                1.0,
                "clean",
                "researcher",
                now,
                "{}",
            ),
        )
    finally:
        db.close()
    return db_path


@pytest.fixture
def no_auth_client(seeded_db: Path) -> TestClient:
    app = build_app(db_path=str(seeded_db), no_auth=True)
    return TestClient(app)


@pytest.fixture
def seeded_kb_client(seeded_db: Path, tmp_path: Path, monkeypatch) -> TestClient:
    """Like no_auth_client but also seeds a real LocalKB under FASTAIAGENT_KB_DIR."""
    pytest.importorskip("faiss")
    from fastaiagent.kb.local import LocalKB

    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    monkeypatch.setenv("FASTAIAGENT_KB_DIR", str(kb_root))

    source_dir = tmp_path / "policies"
    source_dir.mkdir()
    (source_dir / "refunds.md").write_text(
        "Refunds are processed within 7 business days after receiving the return.\n"
    )
    (source_dir / "shipping.md").write_text(
        "Shipping is free on orders over $50.\n"
    )
    kb = LocalKB(name="support-docs", path=str(kb_root), chunk_size=120, chunk_overlap=20)
    kb.add(str(source_dir / "refunds.md"))
    kb.add(str(source_dir / "shipping.md"))

    app = build_app(db_path=str(seeded_db), no_auth=True)
    return TestClient(app)


class TestUIServerSurfaces:
    """Every major read surface returns 200 with the shape the UI expects."""

    def test_status_no_auth_mode(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/auth/status")
        assert r.status_code == 200
        assert r.json()["no_auth"] is True

    def test_overview_aggregates(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/overview")
        assert r.status_code == 200
        body = r.json()
        assert body["traces_last_24h"] >= 1
        assert body["eval_runs_last_7d"] >= 1

    def test_traces_list_with_runner_type_filter(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/traces?runner_type=chain")
        assert r.status_code == 200
        rows = r.json()["rows"]
        assert len(rows) >= 1
        assert rows[0]["runner_type"] == "chain"
        assert rows[0]["runner_name"] == "support"

    def test_trace_detail_carries_runner_fields(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/traces/trace-gate")
        assert r.status_code == 200
        body = r.json()
        assert body["runner_type"] == "chain"
        assert body["span_count"] == 3

    def test_spans_form_a_tree(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/traces/trace-gate/spans")
        assert r.status_code == 200
        tree = r.json()["tree"]
        assert tree["span"]["name"] == "chain.support"
        assert tree["children"][0]["span"]["name"] == "agent.researcher"

    def test_trace_scores_aggregates_guardrails_and_evals(
        self, no_auth_client: TestClient
    ):
        r = no_auth_client.get("/api/traces/trace-gate/scores")
        assert r.status_code == 200
        body = r.json()
        assert len(body["guardrail_events"]) == 1
        assert len(body["eval_cases"]) == 1
        assert body["eval_cases"][0]["run_name"] == "gate"

    def test_thread_view_groups_traces(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/threads/thread-1")
        assert r.status_code == 200
        assert len(r.json()["traces"]) == 1

    def test_analytics_returns_percentiles(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/analytics?hours=24")
        assert r.status_code == 200
        body = r.json()
        assert body["summary"]["trace_count"] >= 1
        assert "p50_ms" in body["summary"]

    def test_guardrail_events_filter_works(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/guardrail-events?outcome=passed")
        assert r.status_code == 200
        assert r.json()["total"] == 1

    def test_prompt_lineage_links_to_trace(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/prompts/greet/lineage")
        assert r.status_code == 200
        assert "trace-gate" in r.json()["trace_ids"]

    def test_agents_derived_from_spans(self, no_auth_client: TestClient):
        r = no_auth_client.get("/api/agents")
        assert r.status_code == 200
        assert any(a["agent_name"] == "researcher" for a in r.json()["agents"])

    def test_bulk_delete_cascades(self, no_auth_client: TestClient, seeded_db: Path):
        r = no_auth_client.post(
            "/api/traces/bulk-delete", json={"trace_ids": ["trace-gate"]}
        )
        assert r.status_code == 200
        assert r.json()["deleted"] == 1
        with SQLiteHelper(seeded_db) as db:
            spans = db.fetchall("SELECT * FROM spans WHERE trace_id = 'trace-gate'")
            guards = db.fetchall(
                "SELECT * FROM guardrail_events WHERE trace_id = 'trace-gate'"
            )
            eval_case = db.fetchone(
                "SELECT trace_id FROM eval_cases WHERE case_id = 'case-1'"
            )
        assert spans == []
        assert guards == []
        # Eval case is preserved but detached (trace_id nulled).
        assert eval_case is not None and eval_case["trace_id"] is None


class TestKBBrowser:
    """The read-only KB browser — list, detail, documents, search, lineage."""

    def test_list_includes_seeded_collection(self, seeded_kb_client: TestClient):
        r = seeded_kb_client.get("/api/kb")
        assert r.status_code == 200
        names = [c["name"] for c in r.json()["collections"]]
        assert "support-docs" in names

    def test_collection_detail(self, seeded_kb_client: TestClient):
        r = seeded_kb_client.get("/api/kb/support-docs")
        assert r.status_code == 200
        body = r.json()
        assert body["chunk_count"] >= 2
        assert body["doc_count"] == 2

    def test_documents_list(self, seeded_kb_client: TestClient):
        r = seeded_kb_client.get("/api/kb/support-docs/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2

    def test_search_returns_refund_doc(self, seeded_kb_client: TestClient):
        r = seeded_kb_client.post(
            "/api/kb/support-docs/search",
            json={"query": "refund", "top_k": 3},
        )
        assert r.status_code == 200
        hits = r.json()["results"]
        assert len(hits) >= 1
        assert "refund" in hits[0]["content"].lower()


class TestUIAuthPath:
    """Confirm the bcrypt auth loop works end-to-end."""

    def test_login_logout_round_trip(self, seeded_db: Path, tmp_path: Path):
        auth_path = tmp_path / "auth.json"
        create_auth_file("gate-user", "correct-horse", path=auth_path)
        app = build_app(
            db_path=str(seeded_db), auth_path=auth_path, no_auth=False
        )
        client = TestClient(app)

        unauth = client.get("/api/traces")
        assert unauth.status_code == 401

        login = client.post(
            "/api/auth/login",
            json={"username": "gate-user", "password": "correct-horse"},
        )
        assert login.status_code == 200
        authed = client.get("/api/traces")
        assert authed.status_code == 200

        client.post("/api/auth/logout")
        assert client.get("/api/traces").status_code == 401
