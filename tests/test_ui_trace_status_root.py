"""A trace fails when its root span fails (1.89.0).

A guardrail span is ERROR for any verdict that is not a pass, a ``warn`` that
let the run complete included. The Traces status filter, Home's failing count
and ``TraceStore.list_traces`` all read any span (or ``MIN(status)``), so a
warned run was "failing". The plane reads the root; so do these now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    otel.reset()

    import fastaiagent as fa
    from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
    from fastaiagent.testing import TestModel

    def agent(name: str, action: str) -> Any:
        return fa.Agent(
            name=name,
            system_prompt="s",
            llm=TestModel(response="the code is SECRET"),
            guardrails=[
                Guardrail(
                    name=f"{name}-rule",
                    guardrail_type=GuardrailType.regex,
                    position=GuardrailPosition.output,
                    config={"pattern": "SECRET"},
                    action=action,
                )
            ],
        )

    warned = agent("warned", "warn").run("hi")
    blocked_id = None
    try:
        agent("blocked", "block").run("hi")
    except Exception:
        pass
    from fastaiagent.trace import TraceStore

    for t in TraceStore().list_traces(last_hours=None):
        if t.trace_id != warned.trace_id:
            blocked_id = t.trace_id
    otel.get_tracer_provider().force_flush()
    yield db, warned.trace_id, blocked_id
    otel.reset()
    reset_config()


def test_trace_store_reports_the_root_status(runs: Any) -> None:
    from fastaiagent.trace import TraceStore

    _, warned, blocked = runs
    status = {t.trace_id: t.status for t in TraceStore().list_traces(last_hours=None)}
    assert status[warned] in ("OK", "UNSET"), "a run that completed is not failing"
    assert status[blocked] == "ERROR"
    found = {t.trace_id: t.status for t in TraceStore().search("rule")}
    assert found[warned] in ("OK", "UNSET") and found[blocked] == "ERROR"


def test_the_ui_status_filter_and_home_count_read_the_root(runs: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    db, warned, blocked = runs
    client = TestClient(build_app(db_path=str(db), no_auth=True))

    failing = client.get("/api/traces", params={"status": "ERROR"}).json()
    assert [r["trace_id"] for r in failing["rows"]] == [blocked]
    assert failing["total"] == 1
    ok = client.get("/api/traces", params={"status": "OK"}).json()
    assert [r["trace_id"] for r in ok["rows"]] == [warned]

    overview = client.get("/api/overview").json()
    assert overview["failing_traces_last_24h"] == 1
    recent = {r["trace_id"]: r["status"] for r in overview["recent_traces"]}
    assert recent[warned] in ("OK", "UNSET") and recent[blocked] == "ERROR"


def test_the_span_tree_says_what_each_guardrail_did(runs: Any) -> None:
    """Span status stays as it was (ERROR for the warn); the local verdict rides beside it."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.server import build_app

    db, warned, _ = runs
    client = TestClient(build_app(db_path=str(db), no_auth=True))
    spans = client.get(f"/api/traces/{warned}").json()["spans"]
    guard = next(s for s in spans if s["name"] == "guardrail.warned-rule")
    assert guard["status"] == "ERROR", "the wire status is unchanged"
    assert guard["guardrail"]["effect"] == "warned"
    assert guard["guardrail"]["outcome"] == "warned"
    assert guard["guardrail"]["blocking"] is True

    def walk(node: dict[str, Any]) -> list[dict[str, Any]]:
        return [node["span"], *[s for c in node["children"] for s in walk(c)]]

    tree = client.get(f"/api/traces/{warned}/spans").json()["tree"]
    in_tree = next(s for s in walk(tree) if s["name"] == "guardrail.warned-rule")
    assert in_tree["guardrail"]["event_id"] == guard["guardrail"]["event_id"]

    scores = client.get(f"/api/traces/{warned}/scores").json()
    (row,) = [g for g in scores["guardrail_events"] if g["guardrail_name"] == "warned-rule"]
    assert row["effect"] == "warned" and row["final_reason"] is None
