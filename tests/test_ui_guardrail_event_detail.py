"""Integration tests for the Guardrail Event Detail surface.

Real FastAPI + real SQLite. Seeds spans + guardrail_events rows directly
into the unified local.db (mirroring how the agent runtime would write
them) and exercises the v5 schema migration, the new detail endpoint,
the false-positive PATCH endpoint, and the new list-page filters.

Sprint 2 / Feature 3.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _insert_span(
    db: SQLiteHelper,
    *,
    span_id: str,
    trace_id: str,
    name: str,
    attributes: dict,
    status: str = "OK",
    ago_minutes: int = 1,
    project_id: str = "",
) -> None:
    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(minutes=ago_minutes)).isoformat()
    end = now.isoformat()
    db.execute(
        """INSERT INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events, project_id)
           VALUES (?, ?, NULL, ?, ?, ?, ?, ?, '[]', ?)""",
        (span_id, trace_id, name, start, end, status, json.dumps(attributes), project_id),
    )


def _insert_event(
    db: SQLiteHelper,
    *,
    event_id: str | None = None,
    trace_id: str = "trace-1",
    span_id: str | None = "span-trigger",
    guardrail_name: str = "no_pii",
    guardrail_type: str = "regex",
    position: str = "output",
    outcome: str = "blocked",
    score: float | None = 0.0,
    message: str | None = "PII detected",
    agent_name: str = "support-bot",
    metadata: dict | None = None,
    project_id: str = "",
    false_positive: int = 0,
    false_positive_at: str | None = None,
) -> str:
    eid = event_id or uuid.uuid4().hex
    db.execute(
        """INSERT INTO guardrail_events
           (event_id, trace_id, span_id, guardrail_name, guardrail_type,
            position, outcome, score, message, agent_name, timestamp, metadata,
            project_id, false_positive, false_positive_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            eid,
            trace_id,
            span_id,
            guardrail_name,
            guardrail_type,
            position,
            outcome,
            score,
            message,
            agent_name,
            _now_iso(),
            json.dumps(metadata or {}),
            project_id,
            false_positive,
            false_positive_at,
        ),
    )
    return eid


@pytest.fixture
def seeded_db(temp_dir: Path) -> Path:
    """A DB with one full triggering trace + three guardrail events
    (blocked / filtered / warned) so every detail-page branch is testable.
    """
    db_path = temp_dir / "local.db"
    init_local_db(db_path).close()

    with SQLiteHelper(db_path) as db:
        # Triggering span — output that PII filter blocked.
        _insert_span(
            db,
            span_id="span-trigger",
            trace_id="trace-1",
            name="agent.support-bot",
            attributes={
                "agent.name": "support-bot",
                "agent.input": "What's my account balance?",
                "agent.output": (
                    "Your balance is $42.10. Email me at "
                    "alice@example.com if you need anything else."
                ),
            },
            ago_minutes=2,
        )
        # Surrounding span (LLM call) so the timeline has more than one row.
        _insert_span(
            db,
            span_id="span-llm",
            trace_id="trace-1",
            name="llm.openai.gpt-4o-mini",
            attributes={
                "gen_ai.request.messages": "[user] What's my balance?",
                "gen_ai.response.content": "Your balance is $42.10",
            },
            ago_minutes=2,
        )

        # Blocked event — PII detected.
        _insert_event(
            db,
            event_id="ev-blocked",
            outcome="blocked",
            metadata={
                "pii_types": ["email"],
                "match": "alice@example.com",
            },
        )
        # Filtered event — same trace, recorded a before/after rewrite.
        _insert_event(
            db,
            event_id="ev-filtered",
            guardrail_name="email_redactor",
            outcome="filtered",
            metadata={
                "before": "alice@example.com",
                "after": "[REDACTED]",
            },
        )
        # Warned event — different trace.
        _insert_span(
            db,
            span_id="span-other",
            trace_id="trace-2",
            name="agent.support-bot",
            attributes={"agent.name": "support-bot"},
            ago_minutes=5,
        )
        _insert_event(
            db,
            event_id="ev-warned",
            trace_id="trace-2",
            span_id="span-other",
            guardrail_name="toxicity_check",
            guardrail_type="classifier",
            position="output",
            outcome="warned",
            metadata={},
        )
    return db_path


@pytest.fixture
def client(seeded_db: Path) -> TestClient:
    app = build_app(db_path=str(seeded_db), no_auth=True)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


class TestSchemaV5:
    """Sprint 2 introduced the false_positive columns at v5; later
    sprints bumped the schema_version while leaving these columns in
    place. The class name is preserved for git-blame continuity; the
    actual assertion compares against ``CURRENT_SCHEMA_VERSION``.
    """

    def test_fresh_db_has_columns(self, temp_dir: Path) -> None:
        db_path = temp_dir / "fresh.db"
        init_local_db(db_path).close()
        with SQLiteHelper(db_path) as db:
            cols = {r["name"] for r in db.fetchall("PRAGMA table_info(guardrail_events)")}
            v = db.fetchone("PRAGMA user_version")
        assert "false_positive" in cols
        assert "false_positive_at" in cols
        assert int(next(iter(v.values()))) == CURRENT_SCHEMA_VERSION

    def test_migration_idempotent(self, temp_dir: Path) -> None:
        db_path = temp_dir / "twice.db"
        init_local_db(db_path).close()
        # Running it again should be a no-op (no errors, version stays).
        init_local_db(db_path).close()
        with SQLiteHelper(db_path) as db:
            v = db.fetchone("PRAGMA user_version")
        assert int(next(iter(v.values()))) == CURRENT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Detail endpoint
# ---------------------------------------------------------------------------


class TestDetailEndpoint:
    def test_blocked_event_returns_three_panels(self, client: TestClient) -> None:
        r = client.get("/api/guardrail-events/ev-blocked")
        assert r.status_code == 200
        body = r.json()
        # Panel 1 — what triggered it.
        assert body["trigger"]["kind"] == "agent_output"
        assert "alice@example.com" in body["trigger"]["text"]
        # Panel 2 — rule details.
        assert body["event"]["guardrail_name"] == "no_pii"
        assert body["event"]["guardrail_type"] == "regex"
        assert body["event"]["metadata"]["pii_types"] == ["email"]
        assert body["event"]["metadata"]["match"] == "alice@example.com"
        # Panel 3 — outcome captured.
        assert body["event"]["outcome"] == "blocked"

    def test_filtered_event_carries_before_after(self, client: TestClient) -> None:
        r = client.get("/api/guardrail-events/ev-filtered")
        assert r.status_code == 200
        body = r.json()
        assert body["event"]["outcome"] == "filtered"
        assert body["event"]["metadata"]["before"] == "alice@example.com"
        assert body["event"]["metadata"]["after"] == "[REDACTED]"

    def test_context_includes_surrounding_spans_and_siblings(
        self, client: TestClient
    ) -> None:
        r = client.get("/api/guardrail-events/ev-blocked")
        body = r.json()
        # Surrounding spans on the same trace.
        span_names = {s["name"] for s in body["context"]["spans"]}
        assert "agent.support-bot" in span_names
        assert "llm.openai.gpt-4o-mini" in span_names
        # The filtered sibling event should appear here too.
        sibling_ids = {e["event_id"] for e in body["context"]["sibling_events"]}
        assert "ev-filtered" in sibling_ids
        # The event itself is excluded from siblings.
        assert "ev-blocked" not in sibling_ids

    def test_unknown_event_returns_404(self, client: TestClient) -> None:
        r = client.get("/api/guardrail-events/does-not-exist")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# False-positive PATCH
# ---------------------------------------------------------------------------


class TestFalsePositivePatch:
    def test_toggle_persists(self, client: TestClient, seeded_db: Path) -> None:
        # Initially false.
        r0 = client.get("/api/guardrail-events/ev-warned")
        assert r0.json()["event"]["false_positive"] is False

        # Toggle on.
        r1 = client.patch(
            "/api/guardrail-events/ev-warned/false-positive",
            json={"false_positive": True, "note": "this is fine"},
        )
        assert r1.status_code == 200
        body = r1.json()
        assert body["false_positive"] is True
        assert body["false_positive_at"]

        # Re-read confirms persistence on the row.
        r2 = client.get("/api/guardrail-events/ev-warned")
        assert r2.json()["event"]["false_positive"] is True

        # Toggle off — also persists.
        r3 = client.patch(
            "/api/guardrail-events/ev-warned/false-positive",
            json={"false_positive": False},
        )
        assert r3.status_code == 200
        assert r3.json()["false_positive"] is False
        assert client.get("/api/guardrail-events/ev-warned").json()["event"][
            "false_positive"
        ] is False

    def test_persists_across_db_reopen(
        self, client: TestClient, seeded_db: Path
    ) -> None:
        """Spec: 'False positive flag persists on page refresh.' Equivalent
        at the test layer is "row survives a new SQLiteHelper connection."
        """
        client.patch(
            "/api/guardrail-events/ev-blocked/false-positive",
            json={"false_positive": True},
        )
        with SQLiteHelper(seeded_db) as db:
            row = db.fetchone(
                "SELECT false_positive FROM guardrail_events WHERE event_id = ?",
                ("ev-blocked",),
            )
        assert int(row["false_positive"]) == 1

    def test_unknown_event_returns_404(self, client: TestClient) -> None:
        r = client.patch(
            "/api/guardrail-events/missing/false-positive",
            json={"false_positive": True},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# List filters: type / position / false_positive
# ---------------------------------------------------------------------------


class TestListFilters:
    def test_filter_by_type(self, client: TestClient) -> None:
        r = client.get("/api/guardrail-events?type=classifier")
        names = [e["guardrail_name"] for e in r.json()["rows"]]
        assert names == ["toxicity_check"]

    def test_filter_by_position(self, client: TestClient) -> None:
        # All seeded events are output-position; filter by input → empty.
        empty = client.get("/api/guardrail-events?position=input")
        assert empty.json()["rows"] == []
        # output → all 3.
        full = client.get("/api/guardrail-events?position=output")
        assert {e["event_id"] for e in full.json()["rows"]} == {
            "ev-blocked",
            "ev-filtered",
            "ev-warned",
        }

    def test_filter_by_false_positive_flag(
        self, client: TestClient
    ) -> None:
        # Mark one as FP, then filter.
        client.patch(
            "/api/guardrail-events/ev-blocked/false-positive",
            json={"false_positive": True},
        )
        fp_only = client.get("/api/guardrail-events?false_positive=true")
        ids = {e["event_id"] for e in fp_only.json()["rows"]}
        assert ids == {"ev-blocked"}
        # And the inverse.
        not_fp = client.get("/api/guardrail-events?false_positive=false")
        ids = {e["event_id"] for e in not_fp.json()["rows"]}
        assert "ev-blocked" not in ids
        assert {"ev-filtered", "ev-warned"}.issubset(ids)


# ---------------------------------------------------------------------------
# Project scoping
# ---------------------------------------------------------------------------


class TestProjectScoping:
    def test_event_in_other_project_404s(self, temp_dir: Path) -> None:
        db_path = temp_dir / "scoped.db"
        init_local_db(db_path).close()
        with SQLiteHelper(db_path) as db:
            _insert_event(
                db,
                event_id="ev-other",
                project_id="other-project",
            )

        # Build app scoped to a *different* project.
        app = build_app(
            db_path=str(db_path), no_auth=True, project_id="my-project"
        )
        client = TestClient(app)

        # Detail 404 — across-project lookup blocked.
        assert client.get("/api/guardrail-events/ev-other").status_code == 404
        # PATCH also 404 — can't flip what we can't see.
        assert (
            client.patch(
                "/api/guardrail-events/ev-other/false-positive",
                json={"false_positive": True},
            ).status_code
            == 404
        )
