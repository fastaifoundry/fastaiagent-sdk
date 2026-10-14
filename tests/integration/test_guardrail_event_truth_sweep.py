"""The guardrail event record tells the truth, across every axis (1.89.0).

The class-closer for the Guardrail Events page: the audit found rows that were
missing by default, rows that described a different outcome from what the run
did (a re-ask as a block, an observe-only mask as a rewrite, a refused rewrite
with no record of the block), the wrong judged text, tool rows with no agent,
and proxies that logged twice. Each was fixed on one path; this sweep holds the
contract on all of them.

Axes: position × action × scheduling (blocking / observe-only) × check state
(clean / fail / errored with on_error allow / block) × path (run / astream) ×
tracing (on / off), with ``FASTAIAGENT_UI_ENABLED`` swept separately.

Contracts:

* **C1.** Tracing on: rows == guardrail spans, and == ``len(result.guardrails)``
  for a completed ``run``. Tracing off: no rows, and ``result.guardrails`` still
  populated.
* **C2.** A row's effect agrees with the run: the final row's effect is the
  truth table's; a rewrite delivered the row's ``after``; an observed or warned
  output is unchanged; a raised run ends on a ``blocked`` row; a ``reasked``
  row has a later attempt.
* **C3.** The judged payload is exactly the text that rule saw.
* **C4.** Every row names the agent, tool positions included.
* **C6.** Every row points at its own guardrail span.

No mocks: ``TestModel`` / ``FunctionModel`` agents, real tools, real SQLite.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType

POSITIONS = ["input", "output", "tool_call", "tool_result"]
ACTIONS = ["block", "warn", "mask", "override", "reask"]
STATES = ["clean", "fail", "errored_allow", "errored_block"]
AGENT = "sweep"
SECRET_REPLY = "the code is SECRET"


def _boom(_text: str) -> bool:
    raise RuntimeError("detector down")


def _rule(position: str, action: str, blocking: bool, state: str) -> Guardrail:
    pos = GuardrailPosition(position)
    if state.startswith("errored"):
        return Guardrail(
            name="rule",
            guardrail_type=GuardrailType.code,
            position=pos,
            fn=_boom,
            action=action,
            blocking=blocking,
            on_error="allow" if state == "errored_allow" else "block",
        )
    return Guardrail(
        name="rule",
        guardrail_type=GuardrailType.regex,
        position=pos,
        config={"pattern": "SECRET", "tripwire_message": "I can't share that."},
        action=action,
        blocking=blocking,
    )


class _Scenario:
    """One agent, and the text each position's rule will judge."""

    def __init__(self, position: str, action: str, blocking: bool, state: str) -> None:
        import fastaiagent as fa
        from fastaiagent.testing import FunctionModel, TestModel
        from fastaiagent.tool.function import FunctionTool

        fail = state == "fail"
        self.input = SECRET_REPLY if (fail and position == "input") else "hello"
        self.reply = SECRET_REPLY if (fail and position == "output") else "all clear"
        self.args = {"q": "SECRET" if (fail and position == "tool_call") else "plain"}
        self.tool_text = "found SECRET" if (fail and position == "tool_result") else "found ok"
        tool_text = self.tool_text

        if position in ("tool_call", "tool_result"):

            def lookup(q: str) -> str:
                return tool_text

            args = self.args

            def responder(messages: Any) -> Any:
                if messages[-1].role.value != "tool":
                    return "", [{"name": "lookup", "arguments": args}]
                return "all clear"

            self.reply = "all clear"
            llm: Any = FunctionModel(responder)
            tools = [FunctionTool(name="lookup", fn=lookup, register=False)]
        else:
            llm = TestModel(response=self.reply)
            tools = []
        self.agent = fa.Agent(
            name=AGENT,
            system_prompt="s",
            llm=llm,
            tools=tools,
            guardrails=[_rule(position, action, blocking, state)],
        )

    def judged(self, position: str) -> str:
        return {
            "input": self.input,
            "output": self.reply,
            "tool_call": json.dumps({"tool": "lookup", "arguments": self.args}, default=str),
            "tool_result": self.tool_text,
        }[position]


def _final_effect(position: str, action: str, blocking: bool, state: str, path: str) -> str:
    """The truth table (plan §2), for the last check of the run."""
    if state == "clean":
        return "none"
    if state == "errored_allow":
        return "unchecked" if blocking else "observed"
    if state == "errored_block":
        return "blocked" if blocking else "observed"
    if not blocking:
        return "observed"
    streamed_output = position == "output" and path == "astream"
    if action == "block" or action == "reask":
        return "blocked"
    if action == "warn":
        return "warned"
    if action == "mask":
        return "blocked" if streamed_output else "rewritten"
    # override: a replaced tool call is not an arguments object any more
    return "blocked" if (streamed_output or position == "tool_call") else "rewritten"


def _drive(scenario: _Scenario, path: str) -> tuple[Any, Exception | None]:
    if path == "run":
        try:
            return scenario.agent.run(scenario.input), None
        except Exception as e:
            return None, e

    async def drain() -> None:
        async for _ in scenario.agent.astream(scenario.input):
            pass

    try:
        asyncio.run(drain())
        return None, None
    except Exception as e:
        return None, e


def _flush(otel: Any) -> None:
    flush = getattr(otel.get_tracer_provider(), "force_flush", None)
    if flush is not None:  # a no-op provider (tracing off) has nothing to flush
        flush()


def _rows(db: Path) -> list[dict[str, Any]]:
    if not db.exists():
        return []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT e.*, p.text AS judged_text FROM guardrail_events e "
                    "LEFT JOIN guardrail_payloads p ON p.payload_id = e.judged_ref "
                    "ORDER BY COALESCE(e.attempt, 0), e.timestamp"
                )
            ]
        except sqlite3.OperationalError:
            return []


def _guardrail_spans(db: Path) -> dict[str, str]:
    if not db.exists():
        return {}
    with sqlite3.connect(db) as conn:
        try:
            return {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT span_id, name FROM spans WHERE name LIKE 'guardrail.%'"
                )
            }
        except sqlite3.OperationalError:
            return {}


@pytest.fixture(params=["tracing_on", "tracing_off"])
def tracing(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, Path]]:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.ui import events

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "1" if request.param == "tracing_on" else "0")
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET", "FASTAIAGENT_UI_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    otel.reset()
    getattr(events, "_reset_writers", lambda: None)()  # absent before 1.89.0
    yield request.param, db
    otel.reset()
    reset_config()


@pytest.mark.parametrize("path", ["run", "astream"])
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("blocking", [True, False], ids=["blocking", "observer"])
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("position", POSITIONS)
def test_the_record_matches_the_run(
    tracing: tuple[str, Path],
    position: str,
    action: str,
    blocking: bool,
    state: str,
    path: str,
) -> None:
    from fastaiagent.trace import otel

    mode, db = tracing
    case = f"{position}/{action}/{'blocking' if blocking else 'observer'}/{state}/{path}/{mode}"
    scenario = _Scenario(position, action, blocking, state)
    result, error = _drive(scenario, path)
    _flush(otel)
    expected = _final_effect(position, action, blocking, state, path)
    assert (error is not None) == (expected == "blocked"), f"{case}: raised={error!r}"

    rows = _rows(db)
    if mode == "tracing_off":
        # C1: tracing is the off switch for the record, not for the caller.
        assert rows == [], f"{case}: rows written with tracing off"
        if result is not None:
            assert result.guardrails, f"{case}: result.guardrails empty with tracing off"
        return

    spans = _guardrail_spans(db)
    # C1
    assert rows, f"{case}: no event row"
    assert len(rows) == len(spans), f"{case}: {len(rows)} rows for {len(spans)} guardrail spans"
    if result is not None:
        assert len(result.guardrails) == len(rows), f"{case}: firings != rows"
        assert [f.effect for f in result.guardrails] == [r["effect"] for r in rows], case

    # C2
    assert rows[-1]["effect"] == expected, f"{case}: final effect {rows[-1]['effect']}"
    for i, row in enumerate(rows):
        if row["effect"] == "reasked":
            later = [r for r in rows[i + 1 :] if (r["attempt"] or 0) > (row["attempt"] or 0)]
            assert later, f"{case}: a reasked row with no later attempt"
        if row["effect"] == "blocked" and row["outcome"] != "errored":
            assert row["outcome"] == "blocked", case
    if result is not None and position == "output":
        last = rows[-1]
        if last["effect"] == "rewritten":
            assert result.output == json.loads(last["metadata"])["after"], case
        elif last["effect"] in ("observed", "warned", "none", "unchecked"):
            assert result.output == scenario.reply, f"{case}: output changed"

    for row in rows:
        # C3
        assert row["judged_text"] == scenario.judged(position), f"{case}: wrong judged text"
        # C4
        assert row["agent_name"] == AGENT, f"{case}: agent_name={row['agent_name']!r}"
        # C6
        assert spans.get(row["guardrail_span_id"]) == "guardrail.rule", f"{case}: span link"


@pytest.mark.parametrize("ui_enabled", [None, "0", "1"], ids=["unset", "0", "1"])
@pytest.mark.parametrize("action", ACTIONS)
def test_ui_enabled_no_longer_matters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ui_enabled: str | None, action: str
) -> None:
    """C1 for every ``FASTAIAGENT_UI_ENABLED`` value: rows follow tracing."""
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel
    from fastaiagent.ui import events

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.delenv("FASTAIAGENT_TRACE_ENABLED", raising=False)
    if ui_enabled is None:
        monkeypatch.delenv("FASTAIAGENT_UI_ENABLED", raising=False)
    else:
        monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", ui_enabled)
    reset_config()
    otel.reset()
    getattr(events, "_reset_writers", lambda: None)()  # absent before 1.89.0
    try:
        scenario = _Scenario("output", action, True, "fail")
        _drive(scenario, "run")
        _flush(otel)
        rows = _rows(db)
        assert rows and len(rows) == len(_guardrail_spans(db))
    finally:
        otel.reset()
        reset_config()


def test_a_supervisor_result_includes_its_workers_firings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C5: ``AgentResult.guardrails`` is every guardrail that executed."""
    import fastaiagent as fa
    from fastaiagent._internal.config import reset_config
    from fastaiagent.agent.team import Supervisor, Worker
    from fastaiagent.testing import FunctionModel, TestModel
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    try:
        worker = fa.Agent(
            name="researcher",
            system_prompt="s",
            llm=TestModel(response=SECRET_REPLY),
            guardrails=[_rule("output", "warn", True, "fail")],
        )

        def manager(messages: Any) -> Any:
            if messages[-1].role.value != "tool":
                return "", [{"name": "delegate_to_researcher", "arguments": {"task": "find it"}}]
            return "done"

        sup = Supervisor(
            name="lead",
            llm=FunctionModel(manager),
            workers=[Worker(agent=worker, role="researcher", description="finds things")],
        )
        result = sup.run("go")
        names = [(f.name, f.effect) for f in result.guardrails]
        assert ("rule", "warned") in names, names
    finally:
        otel.reset()
        reset_config()
