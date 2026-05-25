"""HTTP-level tests for ``?redact=true`` on UI trace endpoints.

Uses the real FastAPI app + real SQLite. The ``no_auth=True`` fixture
keeps the test focused on the redaction behavior — auth flow is
exhaustively covered by ``test_ui_server.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.trace.redaction import (  # noqa: E402
    RedactionPolicy,
    get_redaction_policy,
    set_redaction_policy,
)
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_policy():
    saved = get_redaction_policy()
    set_redaction_policy(None)
    yield
    set_redaction_policy(saved)


@pytest.fixture
def seeded_trace_db(tmp_path: Path) -> Path:
    """Single trace with a known secret embedded in ``gen_ai.response.content``."""
    db_path = tmp_path / "local.db"
    db = init_local_db(db_path)
    try:
        now = datetime.now(tz=timezone.utc).isoformat()
        attrs = {
            "agent.name": "leaky-agent",
            "fastaiagent.runner.type": "agent",
            "gen_ai.response.content": "Here is your key: sk-DEADBEEFCAFE12345678901234567890",
            "fastaiagent.cost.total_usd": 0.001,
        }
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "span-root",
                "trace-redact",
                None,
                "agent.leaky",
                now,
                now,
                "OK",
                json.dumps(attrs),
                "[]",
            ),
        )
    finally:
        db.close()
    return db_path


@pytest.fixture
def client(seeded_trace_db: Path) -> TestClient:
    app = build_app(db_path=str(seeded_trace_db), no_auth=True)
    return TestClient(app)


class TestUIRedactQueryParam:
    def test_no_redact_returns_raw_content(self, client: TestClient):
        # Without ?redact, even with a read-mode policy installed,
        # callers see the raw stored content. The flag is the gate.
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="read"))
        r = client.get("/api/traces/trace-redact")
        assert r.status_code == 200
        body = r.json()
        content = body["spans"][0]["attributes"]["gen_ai.response.content"]
        assert "sk-DEADBEEF" in content

    def test_redact_true_masks_when_read_mode_policy_installed(self, client: TestClient):
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="read"))
        r = client.get("/api/traces/trace-redact?redact=true")
        assert r.status_code == 200
        body = r.json()
        content = body["spans"][0]["attributes"]["gen_ai.response.content"]
        assert "sk-DEADBEEF" not in content
        assert "[REDACTED]" in content

    def test_redact_true_noop_when_no_policy(self, client: TestClient):
        # Read-mode flag passed, but no policy installed → no redaction.
        # The endpoint never invents patterns on its own.
        r = client.get("/api/traces/trace-redact?redact=true")
        assert r.status_code == 200
        body = r.json()
        content = body["spans"][0]["attributes"]["gen_ai.response.content"]
        assert "sk-DEADBEEF" in content

    def test_redact_true_noop_when_capture_only_policy(self, client: TestClient):
        # Capture-only policy means "redact at write time"; data already
        # written stays untouched on read. The flag still flows but the
        # ``_read_redact`` helper short-circuits.
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="capture"))
        r = client.get("/api/traces/trace-redact?redact=true")
        assert r.status_code == 200
        body = r.json()
        content = body["spans"][0]["attributes"]["gen_ai.response.content"]
        assert "sk-DEADBEEF" in content

    def test_redact_true_on_spans_endpoint(self, client: TestClient):
        # ``/spans`` returns a tree; redaction has to thread through
        # the tree builder, not just the flat-list endpoint.
        set_redaction_policy(RedactionPolicy(patterns=(r"sk-\w+",), mode="both"))
        r = client.get("/api/traces/trace-redact/spans?redact=true")
        assert r.status_code == 200
        body = r.json()
        # ``SpanTreeNode`` is ``{span: SpanRow, children: [...]}``;
        # the root span carries the redacted content.
        root_span = body["tree"]["span"]
        content = root_span["attributes"].get("gen_ai.response.content", "")
        assert "sk-DEADBEEF" not in content
        assert "[REDACTED]" in content
