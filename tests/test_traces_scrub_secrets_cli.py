"""``fastaiagent traces scrub-secrets`` cleans rows an older SDK stored (1.89.0).

The rows are seeded the way SDK ≤ 1.88.0 wrote them, because 1.89.0 can no
longer produce them: a guardrail ``llm`` config holding a key in
``agent.guardrails``, and a provider error that echoed a key in an exception
event. One is marked as already sent to the plane, which the command must
report: it cannot recall that copy.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from typer.testing import CliRunner

from fastaiagent.cli.main import app

STORED_KEY = "sk-old-0123456789abcdefOLDkeyDoNotKeep"
ENV_KEY = "gw-env-0123456789ABCDEFenvTokenValue"


def _seed(db_path: Any) -> None:
    from fastaiagent.ui.db import init_local_db

    init_local_db(db_path).close()
    guardrails = [
        {
            "name": "no-weapons",
            "guardrail_type": "topic",
            "position": "input",
            "config": {
                "topics": [{"name": "weapons", "description": ""}],
                "mode": "deny",
                "llm": {"model": "gpt-4o-mini", "api_key": STORED_KEY},
            },
        }
    ]
    echo = {
        "name": "exception",
        "timestamp": "1",
        "attributes": {
            "exception.type": "LLMProviderError",
            "exception.message": f"OpenAI API error 401: Incorrect API key provided: {ENV_KEY}.",
        },
    }
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO spans (span_id, trace_id, parent_span_id, name, attributes, events, "
            "synced) VALUES (?, ?, NULL, ?, ?, ?, 1)",
            (
                "root",
                "t1",
                "agent.old",
                json.dumps({"agent.guardrails": json.dumps(guardrails)}),
                "[]",
            ),
        )
        conn.execute(
            "INSERT INTO spans (span_id, trace_id, parent_span_id, name, attributes, events, "
            "synced) VALUES (?, ?, ?, ?, ?, ?, 0)",
            ("llm", "t1", "root", "llm.openai", "{}", json.dumps([echo])),
        )
        conn.execute(
            "INSERT INTO spans (span_id, trace_id, parent_span_id, name, attributes, events) "
            "VALUES (?, ?, NULL, ?, ?, ?)",
            ("clean", "t2", "agent.new", json.dumps({"agent.input": "hello"}), "[]"),
        )
        conn.commit()
    finally:
        conn.close()


def _dump(db_path: Any) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        return "\n".join(repr(r) for r in conn.execute("SELECT * FROM spans"))
    finally:
        conn.close()


@pytest.fixture
def old_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    for name in ("OPENAI_API_KEY", "FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", ENV_KEY)
    reset_config()
    _seed(db_path)
    yield db_path
    reset_config()


def test_dry_run_reports_and_writes_nothing(old_db: Any) -> None:
    before = _dump(old_db)

    result = CliRunner().invoke(app, ["traces", "scrub-secrets", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "2 span(s) in 1 trace(s)" in result.output
    assert "1 of them were already sent" in result.output
    assert "rotate" in result.output
    assert STORED_KEY not in result.output and ENV_KEY not in result.output
    assert _dump(old_db) == before


def test_scrub_removes_both_leaks_and_keeps_the_rest(old_db: Any) -> None:
    result = CliRunner().invoke(app, ["traces", "scrub-secrets", "-y"])

    assert result.exit_code == 0, result.output
    rows = _dump(old_db)
    assert STORED_KEY not in rows
    assert ENV_KEY not in rows
    assert "Incorrect API key provided" in rows, "the provider's reason must survive"
    assert "gpt-4o-mini" in rows, "the shareable part of the config must survive"
    assert "hello" in rows, "an unrelated payload must be untouched"

    again = CliRunner().invoke(app, ["traces", "scrub-secrets", "--dry-run"])
    assert "No credentials found in 3 span(s)" in again.output


def test_the_rebuilt_rule_from_a_scrubbed_row_still_loads(old_db: Any) -> None:
    """Scrubbing must leave a row Replay can still rebuild a rule from."""
    from fastaiagent.guardrail import Guardrail

    CliRunner().invoke(app, ["traces", "scrub-secrets", "-y"])
    conn = sqlite3.connect(str(old_db))
    try:
        (attrs,) = conn.execute("SELECT attributes FROM spans WHERE span_id='root'").fetchone()
    finally:
        conn.close()
    rules = json.loads(json.loads(attrs)["agent.guardrails"])
    rule = Guardrail.from_dict(rules[0])
    assert rule.config["llm"] == {"model": "gpt-4o-mini"}
