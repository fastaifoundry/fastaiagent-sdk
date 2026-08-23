"""security_audit_2 N9 — trace mutations must respect project scoping.

Reads were already project-scoped via ``project_filter``; delete / note /
favorite were not, so a session scoped to project A could mutate project B's
traces by id. These tests use the real FastAPI app + real SQLite (no mocking):
one shared ``local.db`` holding a trace in project "A" and one in project "B",
and two apps each scoped to one project.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _insert_trace(db: SQLiteHelper, trace_id: str, project_id: str) -> None:
    db.execute(
        """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                              start_time, end_time, status, attributes, events, project_id)
           VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]', ?)""",
        (
            f"span_{trace_id}",
            trace_id,
            "agent.bot",
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:01Z",
            json.dumps({"agent.name": "bot", "agent.input": "hi", "agent.output": "yo"}),
            project_id,
        ),
    )


@pytest.fixture
def two_projects(tmp_path: Path):
    """One DB, a trace in project A and a trace in project B, plus a client
    scoped to project A."""
    db_path = tmp_path / ".fastaiagent" / "local.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(db_path).close()
    with SQLiteHelper(db_path) as db:
        _insert_trace(db, "trace_A", "proj-A")
        _insert_trace(db, "trace_B", "proj-B")

    client_a = TestClient(build_app(db_path=str(db_path), no_auth=True, project_id="proj-A"))
    return client_a, db_path


class TestCrossProjectMutationBlocked:
    def test_delete_other_project_trace_404s_and_leaves_it(self, two_projects) -> None:
        client_a, db_path = two_projects
        r = client_a.delete("/api/traces/trace_B")
        assert r.status_code == 404, r.text
        with SQLiteHelper(db_path) as db:
            row = db.fetchone("SELECT COUNT(*) AS n FROM spans WHERE trace_id = ?", ("trace_B",))
        assert row["n"] == 1  # untouched

    def test_bulk_delete_only_affects_own_project(self, two_projects) -> None:
        client_a, db_path = two_projects
        r = client_a.post(
            "/api/traces/bulk-delete", json={"trace_ids": ["trace_A", "trace_B"]}
        )
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] == 1  # only trace_A
        with SQLiteHelper(db_path) as db:
            b = db.fetchone("SELECT COUNT(*) AS n FROM spans WHERE trace_id = ?", ("trace_B",))
            a = db.fetchone("SELECT COUNT(*) AS n FROM spans WHERE trace_id = ?", ("trace_A",))
        assert b["n"] == 1 and a["n"] == 0

    def test_note_on_other_project_404s(self, two_projects) -> None:
        client_a, db_path = two_projects
        r = client_a.post("/api/traces/trace_B/notes", json={"note": "pwned"})
        assert r.status_code == 404, r.text
        with SQLiteHelper(db_path) as db:
            row = db.fetchone("SELECT COUNT(*) AS n FROM trace_notes WHERE trace_id = ?", ("trace_B",))
        assert row["n"] == 0

    def test_favorite_on_other_project_404s(self, two_projects) -> None:
        client_a, _ = two_projects
        r = client_a.post("/api/traces/trace_B/favorite")
        assert r.status_code == 404, r.text


class TestSameProjectMutationWorks:
    def test_delete_own_trace_ok(self, two_projects) -> None:
        client_a, _ = two_projects
        assert client_a.delete("/api/traces/trace_A").status_code == 200

    def test_note_and_favorite_own_trace_ok(self, two_projects) -> None:
        client_a, _ = two_projects
        assert client_a.post("/api/traces/trace_A/notes", json={"note": "ok"}).status_code == 200
        assert client_a.post("/api/traces/trace_A/favorite").json()["favorited"] is True


def test_unscoped_app_can_mutate_any_trace(tmp_path: Path) -> None:
    """Backward-compat: a single-project / legacy app (project_id='') keeps the
    previous 'act on any trace' behavior — this is NOT a breaking change."""
    db_path = tmp_path / ".fastaiagent" / "local.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(db_path).close()
    with SQLiteHelper(db_path) as db:
        _insert_trace(db, "trace_X", "")
    client = TestClient(build_app(db_path=str(db_path), no_auth=True))  # project_id=""
    assert client.post("/api/traces/trace_X/notes", json={"note": "ok"}).status_code == 200
    assert client.delete("/api/traces/trace_X").status_code == 200
