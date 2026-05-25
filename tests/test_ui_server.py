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
# Large-JSONL test below uses multipart upload — see test_datasets_api
# for the same reasoning.
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.auth import create_auth_file  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


class _CSRFAwareTestClient(TestClient):
    """TestClient that auto-injects ``X-CSRF-Token`` from the cookie jar
    on every state-changing request — mirrors what the bundled React UI
    does in production. Without this, every POST/PUT/PATCH/DELETE made
    after login would 403 because of the M4 CSRF middleware.
    """

    _UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        if method.upper() in self._UNSAFE:
            csrf = self.cookies.get("fastaiagent_csrf")
            if csrf:
                headers = kwargs.get("headers")
                if headers is None:
                    headers = {}
                    kwargs["headers"] = headers
                if isinstance(headers, dict):
                    headers.setdefault("X-CSRF-Token", csrf)
        return super().request(method, url, *args, **kwargs)


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
                        "agent.name": "example-agent",
                        "fastaiagent.cost.total_usd": 0.0012,
                        "gen_ai.usage.input_tokens": 80,
                        "gen_ai.usage.output_tokens": 40,
                        "fastaiagent.thread.id": "t-1",
                        "fastaiagent.prompt.name": "greet",
                        "fastaiagent.prompt.version": "1",
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
    return _CSRFAwareTestClient(app)


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

    # -----------------------------------------------------------------
    # security_review_1.md H5 — login throttling
    # -----------------------------------------------------------------

    def test_login_throttles_after_repeated_failures(self, client_with_auth):
        """Five wrong-password attempts must arm the per-IP+user lockout.

        The throttler treats the 5th failure itself as the trigger — so
        the first four attempts return 401 and the fifth onwards returns
        429 with a ``Retry-After`` header.
        """
        from fastaiagent.ui.throttle import get_default_throttler

        # Reset so this test is independent of any earlier cases.
        get_default_throttler().reset()
        for _ in range(4):
            r = client_with_auth.post(
                "/api/auth/login",
                json={"username": "upendra", "password": "wrong"},
            )
            assert r.status_code == 401, r.text
        # Fifth wrong attempt arms the lockout.
        r = client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "wrong"},
        )
        assert r.status_code == 429, r.text
        assert "retry-after" in {h.lower() for h in r.headers.keys()}
        # While locked, even the correct password is rejected.
        r = client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "correct-horse"},
        )
        assert r.status_code == 429

    def test_login_success_resets_throttle(self, client_with_auth):
        """A successful login clears the failure counter for that key."""
        from fastaiagent.ui.throttle import get_default_throttler

        get_default_throttler().reset()
        for _ in range(3):
            client_with_auth.post(
                "/api/auth/login",
                json={"username": "upendra", "password": "wrong"},
            )
        # Correct password — counter should reset.
        ok = client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "correct-horse"},
        )
        assert ok.status_code == 200
        # Now another 4 wrong attempts must NOT trigger lockout (counter
        # was reset by the success above).
        for _ in range(4):
            r = client_with_auth.post(
                "/api/auth/login",
                json={"username": "upendra", "password": "wrong"},
            )
            assert r.status_code == 401, r.text

    # -----------------------------------------------------------------
    # security_review_1.md H3 — derive cookie ``Secure`` from request scheme
    # -----------------------------------------------------------------

    def test_session_cookie_secure_off_on_plain_http(self, client_with_auth):
        """Loopback HTTP keeps ``Secure=False`` so the browser doesn't drop it."""
        from fastaiagent.ui.throttle import get_default_throttler

        get_default_throttler().reset()
        r = client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "correct-horse"},
        )
        assert r.status_code == 200
        set_cookie = r.headers.get("set-cookie", "")
        assert "fastaiagent_session=" in set_cookie
        assert "Secure" not in set_cookie

    def test_session_cookie_secure_on_when_forwarded_https(
        self, client_with_auth
    ):
        """A TLS-terminating proxy advertising ``X-Forwarded-Proto: https``
        flips the cookie to ``Secure``.
        """
        from fastaiagent.ui.throttle import get_default_throttler

        get_default_throttler().reset()
        r = client_with_auth.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "correct-horse"},
            headers={"X-Forwarded-Proto": "https"},
        )
        assert r.status_code == 200
        set_cookie = r.headers.get("set-cookie", "")
        assert "Secure" in set_cookie


# -------------------------------------------------------------------------
# v1.11.0 Medium-batch regression tests
# -------------------------------------------------------------------------


class TestMediumBatch:
    """Regression coverage for the Medium-severity findings shipping in
    1.11.0. Each test maps to a single security_review_1.md ID.
    """

    # ----- M3 — security headers on every response -----

    def test_m3_security_headers_set_on_every_response(self, client_no_auth):
        r = client_no_auth.get("/api/auth/status")
        assert r.status_code == 200
        for h in (
            "Content-Security-Policy",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Referrer-Policy",
            "Permissions-Policy",
        ):
            assert r.headers.get(h), f"missing header: {h}"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]

    # ----- M4 — CSRF double-submit token -----

    def test_m4_csrf_required_for_authed_post(self, app_with_auth):
        """A vanilla TestClient (no CSRF auto-injection) hits 403 on a
        POST after login. The CSRF-aware client we use elsewhere keeps
        working.
        """
        from fastaiagent.ui.throttle import get_default_throttler

        get_default_throttler().reset()
        app, _, _ = app_with_auth
        plain = TestClient(app)
        # Authenticate.
        r = plain.post(
            "/api/auth/login",
            json={"username": "upendra", "password": "correct-horse"},
        )
        assert r.status_code == 200
        # Now POST without echoing the CSRF cookie back as a header → 403.
        r = plain.post("/api/auth/logout")
        assert r.status_code == 403
        body = r.json()
        assert "csrf" in str(body).lower()

    def test_m4_csrf_cookie_issued_on_first_response(self, client_no_auth):
        """The cookie is set even in no_auth mode so the React client
        always has a value to echo back if/when auth is re-enabled.
        """
        # Some httpx versions don't surface set-cookie via .headers for
        # raw access; check the cookie jar.
        client_no_auth.get("/api/auth/status")
        assert client_no_auth.cookies.get("fastaiagent_csrf"), (
            "fastaiagent_csrf cookie should be issued on safe responses"
        )

    # ----- M5 — LLM rate limit on the playground -----

    def test_m5_llm_rate_limit_returns_429_after_burst(
        self, client_no_auth, monkeypatch
    ):
        """30 calls / minute is the default. Reduce to 2 for the test so
        we can verify the 429 fires without making 30 LLM requests.

        The rate-limit check runs before the API-key check; whether the
        first two requests succeed or 400-on-missing-key depends on the
        local environment. The contract under test is that the 3rd
        request — past the limit — must return 429.
        """
        from fastaiagent.ui import throttle as throttle_module

        monkeypatch.setattr(
            throttle_module, "_default_llm_limiter",
            throttle_module.RateLimiter(limit=2, window_seconds=60.0),
        )
        # Force the route to short-circuit at the API-key check by
        # clearing OPENAI_API_KEY for this test only — this keeps the
        # rate-limit assertion independent of network state.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        body = {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "prompt_template": "x",
            "variables": {},
            "parameters": {"temperature": 1.0, "max_tokens": 4, "top_p": 1.0},
        }
        # First two pass the limiter and 400 on missing API key.
        for _ in range(2):
            r = client_no_auth.post("/api/playground/run", json=body)
            assert r.status_code == 400, (r.status_code, r.text)
        # Third hits the rate limit before the key check.
        r = client_no_auth.post("/api/playground/run", json=body)
        assert r.status_code == 429, r.text
        assert r.headers.get("retry-after"), "Retry-After header missing"

    # ----- M6 — file-upload size caps -----

    def test_m6_jsonl_upload_capped(self, client_no_auth):
        """A multi-MB JSONL upload above 5 MiB gets 413 — and is refused
        WITHOUT being read fully into memory (the streaming reader stops
        as soon as the cap is exceeded).
        """
        # 6 MiB of valid JSONL — well above the 5 MiB cap.
        line = '{"input": "hello", "expected_output": "hi"}\n'
        payload = (line * (6 * 1024 * 1024 // len(line))).encode()
        # Create the dataset first so the path validation passes.
        r = client_no_auth.post(
            "/api/datasets",
            json={"name": "huge", "description": "for size cap test"},
        )
        assert r.status_code in (200, 201, 409), r.text
        r = client_no_auth.post(
            "/api/datasets/huge/import",
            files={"file": ("big.jsonl", payload, "application/x-ndjson")},
            data={"mode": "replace"},
        )
        assert r.status_code == 413, r.text


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

    def test_delete_removes_all_versions_and_aliases(
        self, monkeypatch, seeded_db, temp_dir
    ):
        """Real DELETE flow: register multiple versions + an alias, then
        delete the prompt and verify every related row is gone.
        """
        # Editor gate requires DB inside cwd.
        monkeypatch.chdir(temp_dir)
        app = build_app(db_path=str(seeded_db), no_auth=True)
        client = TestClient(app)

        # Push a v2 so there's history to clean up.
        r = client.put(
            "/api/prompts/greet", json={"template": "Hola {{name}}!"}
        )
        assert r.status_code == 200, r.text

        # Pin an alias so the alias-cleanup path is exercised too.
        with SQLiteHelper(seeded_db) as db:
            db.execute(
                "INSERT INTO prompt_aliases (slug, alias, version) VALUES (?, ?, ?)",
                ("greet", "production", "1"),
            )

        # Sanity — both versions visible before delete.
        rv = client.get("/api/prompts/greet/versions").json()
        assert {v["version"] for v in rv["versions"]} == {"1", "2"}

        # Delete.
        rd = client.delete("/api/prompts/greet")
        assert rd.status_code == 200, rd.text
        body = rd.json()
        assert body["slug"] == "greet"
        assert body["versions_deleted"] == 2

        # Now everything is gone — prompt detail 404s, versions list 404s,
        # and the rows in every related table are removed.
        assert client.get("/api/prompts/greet").status_code == 404
        assert client.get("/api/prompts/greet/versions").status_code == 404
        with SQLiteHelper(seeded_db) as db:
            assert (
                db.fetchone(
                    "SELECT COUNT(*) AS n FROM prompts WHERE slug = 'greet'"
                )["n"]
                == 0
            )
            assert (
                db.fetchone(
                    "SELECT COUNT(*) AS n FROM prompt_versions WHERE slug = 'greet'"
                )["n"]
                == 0
            )
            assert (
                db.fetchone(
                    "SELECT COUNT(*) AS n FROM prompt_aliases WHERE slug = 'greet'"
                )["n"]
                == 0
            )

    def test_delete_404_when_unknown(
        self, monkeypatch, seeded_db, temp_dir
    ):
        monkeypatch.chdir(temp_dir)
        app = build_app(db_path=str(seeded_db), no_auth=True)
        client = TestClient(app)
        r = client.delete("/api/prompts/no-such-prompt")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_delete_rejected_when_registry_external(self, client_no_auth):
        """DB outside cwd → 403, mirrors PUT semantics."""
        r = client_no_auth.delete("/api/prompts/greet")
        assert r.status_code == 403
        assert "external" in r.json()["detail"].lower()

    def test_delete_respects_project_scope(
        self, monkeypatch, seeded_db, temp_dir
    ):
        """A prompt tagged with project_id ``other`` must not be deleted
        by a request scoped to project_id ``me`` — even though the slug
        matches. Returns 404 (not found *in this scope*) and leaves the
        row intact.
        """
        # Tag the seeded prompt with a different project than the UI scope.
        with SQLiteHelper(seeded_db) as db:
            db.execute(
                "UPDATE prompts SET project_id = 'other' WHERE slug = 'greet'"
            )
            db.execute(
                "UPDATE prompt_versions SET project_id = 'other' "
                "WHERE slug = 'greet'"
            )

        monkeypatch.chdir(temp_dir)
        app = build_app(
            db_path=str(seeded_db), no_auth=True, project_id="me"
        )
        client = TestClient(app)

        # DELETE under project=me → 404 because the row belongs to
        # project=other.
        r = client.delete("/api/prompts/greet")
        assert r.status_code == 404

        # Row is preserved.
        with SQLiteHelper(seeded_db) as db:
            row = db.fetchone(
                """SELECT template FROM prompt_versions
                   WHERE slug = 'greet' AND project_id = 'other'"""
            )
        assert row is not None
        assert row["template"] == "Hello {{name}}!"

    def test_update_writes_with_app_context_project_id(
        self, monkeypatch, seeded_db, temp_dir
    ):
        """Regression: PUT must stamp the new version with the
        AppContext's project_id, not the cwd-derived fallback.

        Without this, ``safe_get_project_id()`` returns the directory name
        (`temp_dir` here) but the UI is scoped to a different
        ``project_id``, and the new version becomes invisible to the
        editor that just saved it.
        """
        # Pre-stamp v1 with the demo project's id (mirrors what the
        # platform / seed scripts produce in real deployments).
        with SQLiteHelper(seeded_db) as db:
            db.execute(
                "UPDATE prompts SET project_id = ? WHERE slug = 'greet'",
                ("demo-project",),
            )
            db.execute(
                "UPDATE prompt_versions SET project_id = ? WHERE slug = 'greet'",
                ("demo-project",),
            )
        monkeypatch.chdir(temp_dir)
        app = build_app(
            db_path=str(seeded_db), no_auth=True, project_id="demo-project"
        )
        client = TestClient(app)
        r = client.put(
            "/api/prompts/greet", json={"template": "Bonjour {{name}}!"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["version"] == 2

        # Read-side: the project-scoped /versions endpoint should now see v2.
        rv = client.get("/api/prompts/greet/versions")
        assert rv.status_code == 200
        versions = rv.json()["versions"]
        assert {v["version"] for v in versions} == {"1", "2"}

        # And the row in the table carries the right project_id.
        with SQLiteHelper(seeded_db) as db:
            rows = db.fetchall(
                "SELECT version, project_id FROM prompt_versions WHERE slug='greet'"
            )
        for row in rows:
            assert row["project_id"] == "demo-project", row

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

    def test_path_traversal_does_not_serve_files_outside_static(
        self, temp_dir, seeded_db
    ):
        """Path-traversal sequences must not escape the static dir.

        Regression for security_review_1.md C2: ``static / "../foo"`` used
        to resolve outside the bundle and FileResponse would serve any
        file the process could read.
        """
        # Drop a fake static bundle next to a "secret" file we don't want served.
        # The server only mounts /assets and the SPA fallback when a static dir
        # is present, so we build one with a known file and a sibling secret.
        static_dir = temp_dir / "static"
        static_dir.mkdir()
        (static_dir / "index.html").write_text("<html>ok</html>")
        (static_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
        secret = temp_dir / "secret.txt"
        secret.write_text("super-secret-token")

        # Build an app with an explicit static dir by monkey-patching the
        # locator (build_app picks up package-resource static, but in tests
        # we override _static_dir).
        from fastaiagent.ui import server as server_module

        original = server_module._static_dir
        server_module._static_dir = lambda: static_dir
        try:
            app = build_app(db_path=str(seeded_db), no_auth=True)
            client = TestClient(app)

            # Legitimate asset still works.
            r = client.get("/logo.png")
            assert r.status_code == 200
            assert r.content.startswith(b"\x89PNG")

            # Traversal MUST NOT serve the sibling secret. The fallback either
            # 404s or serves the SPA index — never the out-of-bundle file.
            for path in (
                "../secret.txt",
                "..%2Fsecret.txt",
                "subdir/../../secret.txt",
            ):
                r = client.get(f"/{path}")
                # The route is a catch-all, so we just need to confirm we
                # never see the secret content.
                assert b"super-secret-token" not in r.content
        finally:
            server_module._static_dir = original
