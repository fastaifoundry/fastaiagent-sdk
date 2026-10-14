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
    def _local_db(self, monkeypatch, temp_dir):
        # 1.89.0: rows follow tracing, not FASTAIAGENT_UI_ENABLED.
        monkeypatch.delenv("FASTAIAGENT_UI_ENABLED", raising=False)
        monkeypatch.delenv("FASTAIAGENT_TRACE_ENABLED", raising=False)
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

    @pytest.mark.parametrize("ui_enabled", ["true", "false", None])
    def test_tracing_off_writes_nothing_whatever_ui_enabled_says(
        self, monkeypatch, temp_dir, ui_enabled
    ):
        """``TRACE_ENABLED=0`` is the off switch; ``UI_ENABLED`` no longer matters (1.89.0)."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "0")
        if ui_enabled is None:
            monkeypatch.delenv("FASTAIAGENT_UI_ENABLED", raising=False)
        else:
            monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", ui_enabled)
        reset_config()
        assert log_guardrail_event(Guardrail(name="no_pii"), GuardrailResult(passed=True)) is None
        assert not (temp_dir / "local.db").exists()

    def test_ui_enabled_false_still_records(self, monkeypatch, temp_dir):
        """The event log is local capture, not a UI feature (1.89.0)."""
        from fastaiagent._internal.config import get_config

        monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", "false")
        reset_config()
        log_guardrail_event(Guardrail(name="no_pii"), GuardrailResult(passed=True))
        with SQLiteHelper(get_config().local_db_path) as db:
            assert db.fetchone("SELECT COUNT(*) AS n FROM guardrail_events")["n"] == 1


class TestGuardrailExecuteIntegration:
    """A bare ``Guardrail.aexecute`` records only inside a recording span (1.89.0)."""

    @pytest.fixture(autouse=True)
    def _tracing(self, monkeypatch, temp_dir):
        from fastaiagent.trace import otel

        monkeypatch.delenv("FASTAIAGENT_UI_ENABLED", raising=False)
        monkeypatch.delenv("FASTAIAGENT_TRACE_ENABLED", raising=False)
        monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(temp_dir / "local.db"))
        reset_config()
        otel.reset()
        yield
        otel.reset()
        reset_config()

    def test_aexecute_inside_a_span_writes_one_row(self, temp_dir):
        from fastaiagent._internal.config import get_config
        from fastaiagent.guardrail.builtins import no_pii
        from fastaiagent.trace.otel import get_tracer

        guard = no_pii()
        with get_tracer("test").start_as_current_span("app.handler"):
            asyncio.run(guard.aexecute("my SSN is 111-22-3333"))

        with SQLiteHelper(get_config().local_db_path) as db:
            rows = db.fetchall("SELECT guardrail_name, outcome, effect FROM guardrail_events")
        assert len(rows) == 1
        assert rows[0]["guardrail_name"] == guard.name
        assert rows[0]["outcome"] == "blocked"
        assert rows[0]["effect"] == "blocked"

    def test_a_bare_call_outside_any_span_creates_no_db(self, temp_dir):
        from fastaiagent.guardrail.builtins import no_pii

        result = asyncio.run(no_pii().aexecute("hello world"))
        assert result.passed is True
        assert not (temp_dir / "local.db").exists()
