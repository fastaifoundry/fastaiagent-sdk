"""``position="input"`` — the form the framework docs use — behaves like the enum (1.93.0).

A string was stored raw. It matched every ``==`` (``GuardrailPosition`` is a
``str`` enum), so the check ran and blocked, but each ``.value`` raised: the
``guardrail_events`` row was dropped (the error swallowed at DEBUG) and
``to_dict()`` failed.
"""

from __future__ import annotations

import asyncio
import importlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.guardrail.builtins import no_pii
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType


@pytest.fixture
def traced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.ui import events

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.delenv("FASTAIAGENT_UI_ENABLED", raising=False)
    reset_config()
    otel.reset()
    events._reset_writers()
    yield db
    otel.reset()
    reset_config()


def _rows(db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM guardrail_events")]


def test_strings_are_stored_as_the_enums() -> None:
    g = Guardrail(name="g", guardrail_type="regex", position="input", config={"pattern": "x"})
    assert g.position is GuardrailPosition.input
    assert g.guardrail_type is GuardrailType.regex
    assert no_pii(position="tool_result").position is GuardrailPosition.tool_result


def test_to_dict_round_trips() -> None:
    g = Guardrail(name="g", guardrail_type="regex", position="input", config={"pattern": "x"})
    data = g.to_dict()
    assert data["position"] == "input" and data["guardrail_type"] == "regex"
    assert Guardrail.from_dict(data).position is GuardrailPosition.input


def test_a_misspelt_position_raises_instead_of_never_running() -> None:
    with pytest.raises(ValueError, match="inpt"):
        no_pii(position="inpt")  # type: ignore[arg-type]


@pytest.mark.parametrize("framework", ["langchain", "crewai", "pydanticai"])
def test_each_proxy_writes_a_row(traced: Path, framework: str) -> None:
    from fastaiagent.integrations._registry import GuardrailBlocked

    module = importlib.import_module(f"fastaiagent.integrations.{framework}")
    module._run_guardrails("hello", [no_pii(position="input")], side="input", agent_name="bot")
    with pytest.raises(GuardrailBlocked):
        module._run_guardrails(
            "mail me at jo@example.com", [no_pii(position="input")], side="input", agent_name="bot"
        )
    outcomes = [(r["position"], r["outcome"]) for r in _rows(traced)]
    assert outcomes == [("input", "passed"), ("input", "blocked")]


def test_the_agent_path_writes_a_row(traced: Path) -> None:
    from fastaiagent.guardrail.executor import execute_guardrails

    asyncio.run(execute_guardrails([no_pii(position="input")], "hello", GuardrailPosition.input))
    (row,) = _rows(traced)
    assert (row["position"], row["outcome"]) == ("input", "passed")
