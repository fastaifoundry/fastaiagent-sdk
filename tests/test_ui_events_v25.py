"""local.db v25: the guardrail event record (1.89.0).

A v24 database migrates; an older SDK's INSERT still works against v25 (its
columns are a subset, the new ones nullable); a row from before 1.89.0 reads
back with the new fields empty; and ten rules that judged the same text store
that text once.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fastaiagent._internal.storage import SQLiteHelper

#: The INSERT ``fastaiagent.ui.events.log_guardrail_event`` ran in v1.88.0,
#: verbatim. An older SDK sharing a v25 database runs exactly this.
V188_INSERT = """INSERT INTO guardrail_events
               (event_id, trace_id, span_id, guardrail_name, guardrail_type,
                position, outcome, score, message, agent_name, timestamp, metadata,
                project_id, action, action_taken, severity, floor)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def _v24_db(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastaiagent.ui import db as ui_db

    upto24 = {v: steps for v, steps in ui_db._MIGRATIONS.items() if v <= 24}
    monkeypatch.setattr(ui_db, "_MIGRATIONS", upto24)
    helper = SQLiteHelper(str(path))
    ui_db._run_migrations(helper)
    helper.close()
    monkeypatch.undo()


def _columns(path: Path, table: str) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_a_v24_database_migrates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastaiagent.ui.db import CURRENT_SCHEMA_VERSION, init_local_db

    path = tmp_path / "local.db"
    _v24_db(path, monkeypatch)
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 24
    assert "effect" not in _columns(path, "guardrail_events")

    init_local_db(path).close()

    assert CURRENT_SCHEMA_VERSION == 25
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 25
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    cols = _columns(path, "guardrail_events")
    for c in (
        "effect",
        "rule",
        "latency_ms",
        "judged_ref",
        "guardrail_span_id",
        "batch_id",
        "attempt",
        "final_reason",
    ):
        assert c in cols, c
    assert _columns(path, "guardrail_payloads") == {"payload_id", "text", "created_at"}
    assert {
        "idx_guardrail_events_batch",
        "idx_guardrail_events_span",
        "idx_guardrail_events_ts",
    } <= indexes


def test_an_older_sdk_insert_still_works_and_reads_back_as_legacy(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastaiagent.ui.db import init_local_db
    from fastaiagent.ui.routes.guardrails import _row_to_event

    path = tmp_path / "local.db"
    helper = init_local_db(path)
    helper.execute(
        V188_INSERT,
        (
            "old-1",
            "t1",
            "s1",
            "no_pii",
            "code",
            "output",
            "blocked",
            None,
            "PII",
            "support",
            "2026-10-01T00:00:00+00:00",
            "{}",
            "",
            "reask",
            "reask",
            None,
            0,
        ),
    )
    row = helper.fetchone("SELECT * FROM guardrail_events WHERE event_id = 'old-1'")
    helper.close()
    event = _row_to_event(row)
    assert event.outcome == "blocked"
    assert event.effect is None and event.rule is None and event.blocking is None
    assert event.judged_ref is None and event.batch_id is None


def test_ten_rules_that_judged_one_text_store_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.guardrail.executor import execute_guardrails
    from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
    from fastaiagent.trace import otel
    from fastaiagent.ui import events

    path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(path))
    reset_config()
    otel.reset()
    events._reset_writers()
    try:
        rules = [
            Guardrail(
                name=f"r{i}",
                guardrail_type=GuardrailType.regex,
                position=GuardrailPosition.output,
                config={"pattern": f"nothing{i}"},
            )
            for i in range(10)
        ]
        asyncio.run(execute_guardrails(rules, "a reply", GuardrailPosition.output))
    finally:
        otel.reset()
        reset_config()

    with sqlite3.connect(path) as conn:
        refs = {r[0] for r in conn.execute("SELECT judged_ref FROM guardrail_events")}
        batches = {r[0] for r in conn.execute("SELECT batch_id FROM guardrail_events")}
        payloads = conn.execute("SELECT text FROM guardrail_payloads").fetchall()
        n = conn.execute("SELECT COUNT(*) FROM guardrail_events").fetchone()[0]
    assert n == 10
    assert len(refs) == 1 and len(batches) == 1
    assert payloads == [("a reply",)]


def test_rows_do_not_hold_a_judge_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.guardrail.guardrail import Guardrail, GuardrailResult, GuardrailType
    from fastaiagent.ui import events

    path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(path))
    reset_config()
    events._reset_writers()
    g = Guardrail(name="judge", guardrail_type=GuardrailType.llm_judge, config={"prompt": "ok?"})
    g.config["llm"] = {"provider": "openai", "model": "m", "api_key": "sk-live-999"}
    events.log_guardrail_event(g, GuardrailResult(passed=True), data="x")
    reset_config()
    with sqlite3.connect(path) as conn:
        (rule,) = conn.execute("SELECT rule FROM guardrail_events").fetchone()
    assert '"model": "m"' in rule, "precondition: the rule snapshot was written"
    # Every file of the database, the WAL included, not just the main file.
    raw: Any = b"".join(p.read_bytes() for p in tmp_path.glob("local.db*"))
    assert b"sk-live-999" not in raw
