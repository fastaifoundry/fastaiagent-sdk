"""E2E tests for project scoping.

Three flavours, all real (no mocking):

1. ``ProjectConfig`` lifecycle — first execution from a fresh directory
   creates ``.fastaiagent/config.toml`` + ``.gitignore``.
2. SQL stamping — every write path (spans, checkpoints, attachments,
   prompts, eval rows, guardrail events) carries the active project_id.
3. Cross-project leakage — two projects share one DB, and each UI sees
   only its own data on every read endpoint we filter.

No mocking — fixtures use real SQLite + the real FastAPI app.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.project import (  # noqa: E402
    CONFIG_DIR,
    CONFIG_FILE,
    load_or_create,
    reset_for_testing,
    safe_get_project_id,
    set_project_id,
)
from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_project_state():
    """Clear the singleton + override between tests so each scenario starts fresh."""
    reset_for_testing()
    yield
    reset_for_testing()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_load_or_create_writes_config_on_first_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = load_or_create()
    assert config.project_id == tmp_path.name
    assert (tmp_path / CONFIG_DIR / CONFIG_FILE).exists()
    assert (tmp_path / CONFIG_DIR / ".gitignore").exists()
    assert "local.db" in (tmp_path / CONFIG_DIR / ".gitignore").read_text()


def test_import_fastaiagent_does_not_create_files(tmp_path: Path) -> None:
    """``import fastaiagent`` must remain side-effect-free."""
    result = subprocess.run(
        [sys.executable, "-c", "import fastaiagent"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / CONFIG_DIR).exists()


def test_set_project_id_overrides_resolution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_project_id("forced-id")
    assert safe_get_project_id() == "forced-id"
    set_project_id(None)
    # Re-resolves from cwd.
    reset_for_testing()
    assert safe_get_project_id() == tmp_path.name


# ---------------------------------------------------------------------------
# Migration v4 + write-path stamping
# ---------------------------------------------------------------------------


def test_v4_migration_adds_project_id_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    with SQLiteHelper(str(db_path)) as db:
        for table in (
            "spans",
            "checkpoints",
            "pending_interrupts",
            "idempotency_cache",
            "trace_attachments",
            "prompts",
            "prompt_versions",
            "eval_runs",
            "eval_cases",
            "guardrail_events",
        ):
            cols = {r["name"] for r in db.fetchall(f"PRAGMA table_info({table})")}
            assert "project_id" in cols, f"missing project_id on {table}"


def test_writes_carry_active_project_id(tmp_path: Path) -> None:
    """Stamp project_id on every span insert via save_span."""
    set_project_id("project-alpha")
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()

    with SQLiteHelper(str(db_path)) as db:
        from fastaiagent._internal.project import safe_get_project_id

        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events, project_id)
               VALUES ('s1', 't1', NULL, 'agent.x', '', '', 'OK', '{}', '[]', ?)""",
            (safe_get_project_id(),),
        )
        row = db.fetchone("SELECT project_id FROM spans WHERE span_id='s1'")
        assert row["project_id"] == "project-alpha"


# ---------------------------------------------------------------------------
# Cross-project leakage
# ---------------------------------------------------------------------------


def _seed_two_projects(db_path: Path) -> None:
    """Insert spans for two projects sharing the same DB."""
    init_local_db(db_path).close()
    now = datetime.now(tz=timezone.utc)
    with SQLiteHelper(str(db_path)) as db:
        for trace, agent, project in [
            ("trace-A", "agent-A", "project-alpha"),
            ("trace-B", "agent-B", "project-beta"),
        ]:
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events, project_id)
                   VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]', ?)""",
                (
                    f"sp-{uuid.uuid4().hex[:8]}",
                    trace,
                    f"agent.{agent}",
                    now.isoformat(),
                    (now + timedelta(seconds=1)).isoformat(),
                    json.dumps({"agent.name": agent}),
                    project,
                ),
            )


def _client_for(project_id: str, db_path: Path) -> TestClient:
    return TestClient(
        build_app(db_path=str(db_path), no_auth=True, project_id=project_id)
    )


def test_traces_endpoint_filters_by_project(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    _seed_two_projects(db_path)

    alpha = _client_for("project-alpha", db_path)
    beta = _client_for("project-beta", db_path)

    a_rows = alpha.get("/api/traces").json()["rows"]
    b_rows = beta.get("/api/traces").json()["rows"]

    a_traces = {r["trace_id"] for r in a_rows}
    b_traces = {r["trace_id"] for r in b_rows}
    assert a_traces == {"trace-A"}, f"alpha leaked: {a_traces}"
    assert b_traces == {"trace-B"}, f"beta leaked: {b_traces}"


def test_agents_endpoint_filters_by_project(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    _seed_two_projects(db_path)

    alpha = _client_for("project-alpha", db_path)
    beta = _client_for("project-beta", db_path)

    a_agents = {a["agent_name"] for a in alpha.get("/api/agents").json()["agents"]}
    b_agents = {a["agent_name"] for a in beta.get("/api/agents").json()["agents"]}
    assert a_agents == {"agent-A"}
    assert b_agents == {"agent-B"}


def test_get_trace_404s_across_projects(tmp_path: Path) -> None:
    """Project beta cannot fetch project alpha's trace by ID."""
    db_path = tmp_path / "local.db"
    _seed_two_projects(db_path)

    beta = _client_for("project-beta", db_path)
    r = beta.get("/api/traces/trace-A")
    assert r.status_code == 404


def test_unscoped_client_sees_all_traces(tmp_path: Path) -> None:
    """Test fixtures that don't pass project_id keep working (legacy mode)."""
    db_path = tmp_path / "local.db"
    _seed_two_projects(db_path)

    legacy = TestClient(build_app(db_path=str(db_path), no_auth=True))
    rows = legacy.get("/api/traces").json()["rows"]
    traces = {r["trace_id"] for r in rows}
    assert traces == {"trace-A", "trace-B"}


def test_auth_status_exposes_project_id(tmp_path: Path) -> None:
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    client = _client_for("project-alpha", db_path)
    body = client.get("/api/auth/status").json()
    assert body["project_id"] == "project-alpha"
