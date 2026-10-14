"""``eval curate --filter guardrail`` selects exactly the runs where a rule fired (G8, 1.89.0).

The filter matched ``guardrail_events.outcome = 'fail'``. The event writer has
never written ``'fail'`` (it writes passed / blocked / warned / filtered /
errored), so from 1.22.0 to 1.88.0 the filter returned nothing. And event rows
exist only with ``FASTAIAGENT_UI_ENABLED=1``, so even a correct value would
have missed every run with the UI off.

The sweep drives the real writer: a real agent (``TestModel``) with a real
guardrail, across action × verdict × blocking × run/astream × UI on/off, then
asks the public ``curate_from_traces`` which traces it selects. A trace is
selected exactly when the rule judged the payload and fired; a pass, or a check
that errored (an infrastructure failure, not a quality signal), is not.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any

import pytest

from fastaiagent.guardrail import Guardrail, GuardrailType
from fastaiagent.guardrail.guardrail import GuardrailPosition

#: Inlined rather than imported from ``fastaiagent.ui.events`` so this file
#: fails on 1.88.0 by assertion, not ImportError. ``test_the_constant_matches``
#: pins the two together.
WRITTEN_OUTCOMES = {"passed", "blocked", "warned", "filtered", "errored"}

ACTIONS = ["block", "warn", "mask", "override", "reask"]
VERDICTS = ["pass", "fail", "errored"]


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _ui(monkeypatch: pytest.MonkeyPatch, on: bool) -> None:
    from fastaiagent._internal.config import reset_config

    monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", "1" if on else "0")
    reset_config()


def _rule(action: str, verdict: str, blocking: bool) -> Guardrail:
    if verdict == "errored":

        def detector_down(_data: Any) -> bool:
            raise RuntimeError("detector down")

        return Guardrail(
            name="rule",
            guardrail_type=GuardrailType.code,
            position=GuardrailPosition.output,
            fn=detector_down,
            action=action,
            blocking=blocking,
            on_error="allow",
        )
    config: dict[str, Any] = {"pattern": "SECRET"}
    if action == "override":
        config["tripwire_message"] = "I can't share that."
    return Guardrail(
        name="rule",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config=config,
        action=action,
        blocking=blocking,
    )


def _run(path: str, action: str, verdict: str, blocking: bool) -> None:
    import fastaiagent as fa
    from fastaiagent.testing import TestModel

    reply = "the code is SECRET" if verdict == "fail" else "all clear"
    agent = fa.Agent(
        name="support",
        system_prompt="s",
        llm=TestModel(response=reply),
        guardrails=[_rule(action, verdict, blocking)],
    )

    async def drain() -> None:
        async for _ in agent.astream("tell me the code"):
            pass

    try:
        if path == "run":
            agent.run("tell me the code")
        else:
            asyncio.run(drain())
    except Exception:
        pass  # a blocking rule ends the run; the trace is what matters here


def _trace_ids(db: Any) -> set[str]:
    with sqlite3.connect(db) as conn:
        return {r[0] for r in conn.execute("SELECT DISTINCT trace_id FROM spans")}


@pytest.mark.parametrize("ui_on", [True, False], ids=["ui_on", "ui_off"])
@pytest.mark.parametrize("path", ["run", "astream"])
@pytest.mark.parametrize("blocking", [True, False], ids=["blocking", "non_blocking"])
@pytest.mark.parametrize("verdict", VERDICTS)
@pytest.mark.parametrize("action", ACTIONS)
def test_selected_exactly_when_a_rule_fired(
    traced_db: Any,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    verdict: str,
    blocking: bool,
    path: str,
    ui_on: bool,
) -> None:
    from fastaiagent.eval import curate_from_traces

    _ui(monkeypatch, ui_on)
    _run(path, action, verdict, blocking)
    traces = _trace_ids(traced_db)
    assert len(traces) == 1, f"expected one traced run, got {traces}"
    (tid,) = traces

    if ui_on:
        with sqlite3.connect(traced_db) as conn:
            written = {r[0] for r in conn.execute("SELECT outcome FROM guardrail_events")}
        assert written, "the UI was on but no event row was written"
        assert written <= WRITTEN_OUTCOMES, f"unknown outcome(s) written: {written}"

    items = curate_from_traces(filter="guardrail", limit=None)
    selected = {it["trace_id"] for it in items}
    should = verdict == "fail"
    assert (tid in selected) == should, (
        f"{action}/{verdict}/blocking={blocking}/{path}/ui={ui_on}: "
        f"{'missed a run where the rule fired' if should else 'selected a run where nothing fired'}"
    )
    if should:
        reason = next(it["reason"] for it in items if it["trace_id"] == tid)
        assert "guardrail 'rule' fired" in reason


def test_a_false_positive_is_not_selected(traced_db: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.eval import curate_from_traces
    from fastaiagent.ui.server import build_app

    _ui(monkeypatch, True)
    _run("run", "block", "fail", True)
    assert curate_from_traces(filter="guardrail", limit=None), "precondition: it fired"

    with sqlite3.connect(traced_db) as conn:
        (event_id,) = conn.execute("SELECT event_id FROM guardrail_events").fetchone()
    client = TestClient(build_app(db_path=str(traced_db), no_auth=True))
    r = client.patch(
        f"/api/guardrail-events/{event_id}/false-positive", json={"false_positive": True}
    )
    assert r.status_code == 200, r.text

    assert curate_from_traces(filter="guardrail", limit=None) == []


def test_the_constant_matches_what_this_sweep_expects() -> None:
    from fastaiagent.ui.events import FIRED_OUTCOMES, GUARDRAIL_OUTCOMES

    assert set(GUARDRAIL_OUTCOMES) == WRITTEN_OUTCOMES
    assert set(FIRED_OUTCOMES) == {"blocked", "warned", "filtered"}
