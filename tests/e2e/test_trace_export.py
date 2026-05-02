"""E2E tests for the trace JSON export endpoint and CLI.

Hand-seeds a complete trace (root agent + LLM child + a checkpoint + an
attachment), then exercises both the HTTP endpoint and the CLI command
that share ``build_export_payload``. No mocking — real SQLite, real
TestClient, real Typer CliRunner.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.trace.trace_export import (  # noqa: E402
    EXPORT_VERSION,
    build_export_payload,
    export_trace_to_file,
)
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


TRACE_ID = "exp00000000000000000000000000ex01"
SPAN_ROOT = "exp-root-0000000000000000000ex01"
SPAN_LLM = "exp-llm-00000000000000000000ex01"
EXEC_ID = "exec-export-001"


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def export_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    now = datetime.now(tz=timezone.utc)
    with SQLiteHelper(db_path) as db:
        # Root agent span carries the execution_id so the export marks it
        # durable and pulls the checkpoints.
        root_attrs = {
            "agent.name": "weather-bot",
            "agent.input": "What's the weather in Berlin?",
            "agent.output": "It's sunny in Berlin.",
            "agent.tokens_used": 312,
            "fastaiagent.runner.type": "agent",
            "fastaiagent.agent.execution_id": EXEC_ID,
        }
        llm_attrs = {
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 180,
            "gen_ai.usage.output_tokens": 132,
            "gen_ai.request.messages": json.dumps(
                [{"role": "user", "content": "What's the weather in Berlin?"}]
            ),
            "gen_ai.response.content": "It's sunny in Berlin.",
        }
        for sid, parent, name, attrs in [
            (SPAN_ROOT, None, "agent.weather-bot", root_attrs),
            (SPAN_LLM, SPAN_ROOT, "llm.openai.gpt-4o-mini", llm_attrs),
        ]:
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
                (
                    sid,
                    TRACE_ID,
                    parent,
                    name,
                    _iso(now - timedelta(seconds=3)),
                    _iso(now),
                    json.dumps(attrs),
                ),
            )
        # Checkpoint linked to the same execution.
        db.execute(
            """INSERT INTO checkpoints
               (checkpoint_id, chain_name, execution_id, node_id, node_index,
                status, state_snapshot, node_input, node_output,
                interrupt_reason, interrupt_context, agent_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                "weather-bot",
                EXEC_ID,
                "research",
                0,
                "completed",
                json.dumps({"city": "Berlin"}),
                json.dumps({"city": "Berlin"}),
                json.dumps({"weather": "sunny"}),
                "",
                "",
                f"agent:{SPAN_ROOT}",
                _iso(now - timedelta(seconds=2)),
            ),
        )
        # One attachment so the multimodal_attachments list is non-empty.
        db.execute(
            """INSERT INTO trace_attachments
               (attachment_id, trace_id, span_id, media_type, size_bytes,
                thumbnail, full_data, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                TRACE_ID,
                SPAN_LLM,
                "image/png",
                7,
                b"thumb..",  # dummy bytes
                b"thumb..",
                json.dumps({"width": 1, "height": 1}),
                _iso(now),
            ),
        )
    return db_path


@pytest.fixture
def client(export_db: Path) -> TestClient:
    return TestClient(build_app(db_path=str(export_db), no_auth=True))


# ---------------------------------------------------------------------------
# Pure builder
# ---------------------------------------------------------------------------


def test_build_export_payload_shape(export_db: Path) -> None:
    db = SQLiteHelper(str(export_db))
    try:
        payload = build_export_payload(db, TRACE_ID)
    finally:
        db.close()

    assert payload["export_version"] == EXPORT_VERSION
    assert payload["trace"]["trace_id"] == TRACE_ID
    assert payload["trace"]["durable"] is True
    assert payload["trace"]["execution_id"] == EXEC_ID
    assert len(payload["spans"]) == 2

    llm = next(s for s in payload["spans"] if s["span_id"] == SPAN_LLM)
    assert llm["model"] == "gpt-4o-mini"
    assert llm["tokens"] == {"input": 180, "output": 132}
    assert llm["cost"] is not None and llm["cost"] > 0

    # Attachments default to metadata-only.
    att = payload["multimodal_attachments"][0]
    assert att["included"] is False
    assert "attachment_data" not in att

    # Checkpoints default to metadata-only.
    cp = payload["checkpoints"][0]
    assert cp["status"] == "completed"
    assert "state_snapshot" not in cp


def test_build_export_payload_with_attachments_and_state(export_db: Path) -> None:
    db = SQLiteHelper(str(export_db))
    try:
        payload = build_export_payload(
            db,
            TRACE_ID,
            include_attachments=True,
            include_checkpoint_state=True,
        )
    finally:
        db.close()

    att = payload["multimodal_attachments"][0]
    assert att["included"] is True
    decoded = base64.b64decode(att["attachment_data"])
    assert decoded == b"thumb.."

    cp = payload["checkpoints"][0]
    assert cp["state_snapshot"] == {"city": "Berlin"}
    assert cp["node_output"] == {"weather": "sunny"}


def test_build_export_payload_unknown_trace_raises(export_db: Path) -> None:
    db = SQLiteHelper(str(export_db))
    try:
        with pytest.raises(KeyError):
            build_export_payload(db, "does-not-exist")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# HTTP endpoint
# ---------------------------------------------------------------------------


def test_export_endpoint_returns_self_contained_json(client: TestClient) -> None:
    r = client.get(f"/api/traces/{TRACE_ID}/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert (
        f'attachment; filename="trace-{TRACE_ID}.json"'
        in r.headers["content-disposition"]
    )
    body = r.json()
    assert body["trace"]["trace_id"] == TRACE_ID
    assert body["multimodal_attachments"][0]["included"] is False


def test_export_endpoint_with_attachments_flag(client: TestClient) -> None:
    r = client.get(f"/api/traces/{TRACE_ID}/export?include_attachments=true")
    body = r.json()
    att = body["multimodal_attachments"][0]
    assert att["included"] is True
    assert base64.b64decode(att["attachment_data"]) == b"thumb.."


def test_export_endpoint_404_for_unknown_trace(client: TestClient) -> None:
    r = client.get("/api/traces/missing-trace-id/export")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_export_trace_cli_writes_valid_json(export_db: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from fastaiagent.cli.main import app

    out = tmp_path / "trace.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "export-trace",
            "--trace-id",
            TRACE_ID,
            "--output",
            str(out),
            "--db",
            str(export_db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["trace"]["trace_id"] == TRACE_ID
    assert payload["sdk_version"]


def test_export_trace_cli_include_flags(export_db: Path, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from fastaiagent.cli.main import app

    out = tmp_path / "trace-full.json"
    result = CliRunner().invoke(
        app,
        [
            "export-trace",
            "--trace-id",
            TRACE_ID,
            "--output",
            str(out),
            "--db",
            str(export_db),
            "--include-attachments",
            "--include-checkpoint-state",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["multimodal_attachments"][0]["included"] is True
    assert payload["checkpoints"][0]["state_snapshot"] == {"city": "Berlin"}


def test_export_helper_writes_to_arbitrary_path(export_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "subdir" / "deep.json"
    written = export_trace_to_file(export_db, TRACE_ID, out)
    assert written.exists()
    assert json.loads(written.read_text())["trace"]["trace_id"] == TRACE_ID
