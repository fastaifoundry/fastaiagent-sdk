"""Tests for fastaiagent.ui.events — guardrail event persistence to local.db.

Real SQLite, no mocks. The UI is refresh-based: events persist to the DB and
are read on user refresh via /api/guardrail-events.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.ui.events import log_guardrail_event


class TestGuardrailEventLogging:
    @pytest.fixture(autouse=True)
    def _ui_enabled(self, monkeypatch, temp_dir):
        monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", "true")
        monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(temp_dir / "local.db"))
        reset_config()
        yield
        reset_config()

    def test_logs_passed_guardrail(self, temp_dir):
        from fastaiagent._internal.config import get_config

        guard = Guardrail(
            name="no_pii",
            guardrail_type=GuardrailType.regex,
            position=GuardrailPosition.output,
            blocking=True,
        )
        result = GuardrailResult(passed=True, score=1.0, message="clean")

        log_guardrail_event(guard, result)

        db_path = get_config().local_db_path
        with SQLiteHelper(db_path) as db:
            rows = db.fetchall("SELECT * FROM guardrail_events")
        assert len(rows) == 1
        assert rows[0]["guardrail_name"] == "no_pii"
        assert rows[0]["guardrail_type"] == "regex"
        assert rows[0]["position"] == "output"
        assert rows[0]["outcome"] == "passed"
        assert rows[0]["score"] == 1.0

    def test_blocking_vs_non_blocking_outcome(self, temp_dir):
        from fastaiagent._internal.config import get_config

        blocker = Guardrail(name="blocker", blocking=True)
        warner = Guardrail(name="warner", blocking=False)
        failed = GuardrailResult(passed=False, message="bad")

        log_guardrail_event(blocker, failed)
        log_guardrail_event(warner, failed)

        db_path = get_config().local_db_path
        with SQLiteHelper(db_path) as db:
            rows = {
                r["guardrail_name"]: r["outcome"]
                for r in db.fetchall(
                    "SELECT guardrail_name, outcome FROM guardrail_events"
                )
            }
        assert rows["blocker"] == "blocked"
        assert rows["warner"] == "warned"

    def test_respects_ui_enabled_flag(self, monkeypatch, temp_dir):
        from fastaiagent._internal.config import get_config

        monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", "false")
        reset_config()
        guard = Guardrail(name="no_pii")
        log_guardrail_event(guard, GuardrailResult(passed=True))

        db_path = get_config().local_db_path
        from pathlib import Path

        if Path(db_path).exists():
            with SQLiteHelper(db_path) as db:
                rows = db.fetchall(
                    "SELECT COUNT(*) AS n FROM sqlite_master "
                    "WHERE name='guardrail_events'"
                )
                if rows[0]["n"] == 1:
                    count = db.fetchall(
                        "SELECT COUNT(*) AS n FROM guardrail_events"
                    )
                    assert count[0]["n"] == 0


class TestGuardrailExecuteIntegration:
    """Guardrail.aexecute writes a row when UI is enabled — end-to-end."""

    def test_aexecute_writes_row(self, monkeypatch, temp_dir):
        from fastaiagent._internal.config import get_config
        from fastaiagent.guardrail.builtins import no_pii

        monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", "true")
        monkeypatch.setenv(
            "FASTAIAGENT_LOCAL_DB", str(temp_dir / "local.db")
        )
        reset_config()

        guard = no_pii()
        asyncio.run(guard.aexecute("my SSN is 111-22-3333"))

        db_path = get_config().local_db_path
        with SQLiteHelper(db_path) as db:
            rows = db.fetchall(
                "SELECT guardrail_name, outcome FROM guardrail_events"
            )
        assert len(rows) == 1
        assert rows[0]["guardrail_name"] == guard.name
        assert rows[0]["outcome"] == "blocked"

    def test_aexecute_skips_when_ui_disabled(self, monkeypatch, temp_dir):
        from fastaiagent.guardrail.builtins import no_pii

        monkeypatch.delenv("FASTAIAGENT_UI_ENABLED", raising=False)
        monkeypatch.setenv(
            "FASTAIAGENT_LOCAL_DB", str(temp_dir / "local.db")
        )
        reset_config()

        guard = no_pii()
        result = asyncio.run(guard.aexecute("hello world"))
        assert result.passed is True
        assert not (temp_dir / "local.db").exists()
