"""What a cancelled run reports (1.93.0), found against a live model:

* a cancelled ``arun`` / ``run`` returned ``trace_id=None`` though its trace
  exists (``AgentResult.trace_id`` is documented as set on every path);
* a cancelled ``stream()`` returned ``tokens_used=0`` with a non-zero cost;
* the run's span ended UNSET with nothing marking the cancel;
* ``run_error`` named a private class, ``_RunCancelled``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, SQLiteCheckpointer
from fastaiagent.testing import TestModel
from tests._cancel_helpers import cancel_when


@pytest.fixture
def traced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    reset_config()
    otel.reset()
    yield db
    otel.reset()
    reset_config()


def _agent(store: SQLiteCheckpointer, started: threading.Event) -> Agent:
    async def aslow() -> str:
        """Take a while."""
        started.set()
        await asyncio.sleep(5)
        return "late"

    return Agent(
        name="stopped",
        llm=TestModel(
            response=["", "done"], tool_calls=[{"name": "aslow", "arguments": {}}], usage=(10, 5)
        ),
        tools=[FunctionTool(name="aslow", fn=aslow)],
        checkpointer=store,
    )


def _root_status(db: Path) -> Any:
    from fastaiagent.trace import otel

    otel.get_tracer_provider().force_flush()
    with sqlite3.connect(db) as conn:
        (attrs,) = conn.execute(
            "SELECT attributes FROM spans WHERE name = 'agent.stopped'"
        ).fetchone()
    return json.loads(attrs).get("agent.status")


@pytest.mark.parametrize("method", ["run", "stream"])
def test_a_cancelled_run_reports_what_happened(traced: Path, tmp_path: Path, method: str) -> None:
    store = SQLiteCheckpointer(db_path=str(tmp_path / "cp.db"))
    started = threading.Event()
    agent = _agent(store, started)
    cancel_when(started, agent.cancel, f"cf-{method}")
    result = getattr(agent, method)("go", execution_id=f"cf-{method}")
    assert result.status == "cancelled"
    assert result.trace_id, "the cancelled run's trace exists; name it"
    assert result.tokens_used == 15, "one model call was made before the stop"
    assert _root_status(traced) == "cancelled"
    end = store.get_last(f"cf-{method}")
    assert end is not None and end.state_snapshot["run_error"].startswith("RunCancelled:")
