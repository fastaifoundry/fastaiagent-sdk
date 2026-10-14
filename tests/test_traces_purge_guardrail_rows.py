"""``traces purge`` removes the guardrail rows of what it purges (1.89.0).

It deleted the spans first and then selected guardrail rows through a subquery
over the spans it had just removed, so with ``--older-than-days`` it removed no
guardrail rows at all. Those rows hold the text a rule judged (``before``, and
since 1.89.0 the judged payload), which is often the PII the rule was there
for. Rows with no trace were never purged either.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

OLD = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
NEW = datetime.now(timezone.utc).isoformat()


def _seed(path: Path) -> dict[str, Any]:
    from fastaiagent.ui.db import init_local_db

    db = init_local_db(path)
    ids: dict[str, Any] = {"old_trace": uuid.uuid4().hex, "new_trace": uuid.uuid4().hex}
    for key, ts in (("old_trace", OLD), ("new_trace", NEW)):
        db.execute(
            "INSERT INTO spans (span_id, trace_id, parent_span_id, name, start_time, end_time, "
            "status, attributes, events) VALUES (?, ?, NULL, 'agent.a', ?, ?, 'OK', '{}', '[]')",
            (uuid.uuid4().hex, ids[key], ts, ts),
        )
    rows = [
        ("e-old", ids["old_trace"], OLD, "p-old"),
        ("e-new", ids["new_trace"], NEW, "p-new"),
        ("e-orphan-old", None, OLD, "p-orphan"),
        ("e-orphan-new", None, NEW, "p-new"),
    ]
    for event_id, trace_id, ts, payload in rows:
        db.execute(
            "INSERT OR IGNORE INTO guardrail_payloads (payload_id, text, created_at) "
            "VALUES (?, ?, ?)",
            (payload, f"judged text for {payload}: SSN 111-22-3333", ts),
        )
        db.execute(
            "INSERT INTO guardrail_events (event_id, trace_id, span_id, guardrail_name, "
            "guardrail_type, position, outcome, metadata, timestamp, judged_ref) "
            "VALUES (?, ?, NULL, 'no_pii', 'code', 'output', 'blocked', '{}', ?, ?)",
            (event_id, trace_id, ts, payload),
        )
    db.close()
    return ids


def _state(path: Path) -> tuple[set[str], set[str]]:
    with sqlite3.connect(path) as conn:
        events = {r[0] for r in conn.execute("SELECT event_id FROM guardrail_events")}
        payloads = {r[0] for r in conn.execute("SELECT payload_id FROM guardrail_payloads")}
    return events, payloads


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config

    path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(path))
    monkeypatch.delenv("FASTAIAGENT_TRACE_DB_PATH", raising=False)
    reset_config()
    yield path
    reset_config()


def _purge(*args: str) -> None:
    from typer.testing import CliRunner

    from fastaiagent.cli.main import app

    res = CliRunner().invoke(app, ["traces", "purge", *args, "-y"])
    assert res.exit_code == 0, res.output


def test_older_than_days_removes_the_guardrail_rows_and_payloads(db: Path) -> None:
    _seed(db)
    _purge("--older-than-days", "30")
    events, payloads = _state(db)
    assert events == {"e-new", "e-orphan-new"}, "old rows (and old rows with no trace) must go"
    assert payloads == {"p-new"}, "a payload no remaining row refers to must go"


def test_a_full_purge_empties_both_tables(db: Path) -> None:
    _seed(db)
    _purge()
    assert _state(db) == (set(), set())


def test_deleting_a_trace_in_the_ui_collects_its_payload(db: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    ids = _seed(db)
    client = TestClient(build_app(db_path=str(db), no_auth=True))
    r = client.delete(f"/api/traces/{ids['old_trace']}")
    assert r.status_code == 200, r.text
    events, payloads = _state(db)
    assert "e-old" not in events
    assert "p-old" not in payloads and "p-new" in payloads
