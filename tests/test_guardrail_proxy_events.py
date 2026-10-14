"""A framework proxy records each check once, with what it really did (1.89.0).

The LangChain, CrewAI and PydanticAI proxies called ``g.execute()``, which could
write a row, then wrote a second one on failure. And a ``mask`` the proxy could
not apply was recorded as ``filtered`` though the call was blocked.
"""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType

FRAMEWORKS = ["langchain", "crewai", "pydanticai"]


def _boom(_text: str) -> bool:
    raise RuntimeError("detector down")


def _rule(action: str, *, blocking: bool = True, errored: str | None = None) -> Guardrail:
    if errored:
        return Guardrail(
            name=f"{action}-{errored}",
            guardrail_type=GuardrailType.code,
            position=GuardrailPosition.output,
            fn=_boom,
            action=action,
            blocking=blocking,
            on_error=errored,  # type: ignore[arg-type]
        )
    return Guardrail(
        name=f"{action}-{'b' if blocking else 'o'}",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": "SECRET"},
        action=action,
        blocking=blocking,
    )


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
    getattr(events, "_reset_writers", lambda: None)()  # absent before 1.89.0
    yield db
    otel.reset()
    reset_config()


def _rows(db: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM guardrail_events")]


def _run(framework: str, text: str, rules: list[Guardrail]) -> Exception | None:
    from fastaiagent.trace.otel import get_tracer

    module = importlib.import_module(f"fastaiagent.integrations.{framework}")
    with get_tracer("test").start_as_current_span("app.request"):
        try:
            module._run_guardrails(text, rules, side="output", agent_name="bot")
        except Exception as e:  # GuardrailBlocked
            return e
    return None


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize(
    ("action", "effect", "outcome", "final_reason", "raised"),
    [
        ("block", "blocked", "blocked", False, True),
        ("warn", "warned", "warned", False, False),
        ("mask", "blocked", "blocked", True, True),
        ("override", "blocked", "blocked", True, True),
        ("reask", "blocked", "blocked", True, True),
    ],
)
def test_one_row_with_the_proxys_real_outcome(
    traced: Path,
    framework: str,
    action: str,
    effect: str,
    outcome: str,
    final_reason: bool,
    raised: bool,
) -> None:
    err = _run(framework, "the code is SECRET", [_rule(action)])
    assert (err is not None) == raised
    rows = _rows(traced)
    assert len(rows) == 1, f"{framework}/{action}: {len(rows)} rows for one check"
    row = rows[0]
    assert (row["effect"], row["outcome"]) == (effect, outcome)
    assert bool(row["final_reason"]) is final_reason
    assert row["agent_name"] == "bot"
    assert f'"framework": "{framework}"' in row["metadata"]
    assert row["judged_ref"], "the judged text is stored"


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_observe_only_errored_and_clean_checks(traced: Path, framework: str) -> None:
    rules = [
        _rule("mask", blocking=False),
        _rule("block", errored="allow"),
        _rule("block", blocking=False, errored="block"),
    ]
    assert _run(framework, "the code is SECRET", rules) is None
    by_name = {r["guardrail_name"]: r for r in _rows(traced)}
    assert len(by_name) == 3
    assert (by_name["mask-o"]["effect"], by_name["mask-o"]["outcome"]) == ("observed", "warned")
    assert by_name["block-allow"]["effect"] == "unchecked"
    assert by_name["block-allow"]["outcome"] == "errored"
    assert by_name["block-block"]["effect"] == "observed"


def test_outside_any_trace_the_row_still_has_its_agent(traced: Path) -> None:
    from fastaiagent.integrations.langchain import _run_guardrails

    _run_guardrails("all clear", [_rule("block")], side="input", agent_name="bot")
    (row,) = _rows(traced)
    assert row["outcome"] == "passed" and row["agent_name"] == "bot"
    assert row["trace_id"] is None
