"""A replay rerun changes nothing but its own, local-only trace (1.89.0, WS2).

The audit found every rerun path leaking into the real world:

* **R1.** A rerun rebuilt the agent's FunctionTools from the global registry,
  so in the defining process (Python, or the UI started with ``--agent``) it
  **ran the real tools**: the card was charged again, in every mode,
  "recorded" included (it only replays the model).
* **R2.** REST and MCP tools were rebuilt from the trace and **re-fired from
  any process**, a fresh one included.
* **R5/R6/X1.** The rerun's trace was an ordinary trace: exported to the plane
  as a real run, counted in Analytics/Overview/Agents, curated as gold data.

This sweep reruns one agent (``tests/_replay_isolation_harness.py``) across
surface × mode and asserts, through the 1.88 API surface only, that nothing
escaped: no side effect, no new rows outside the rerun's spans, the spans
marked ``fastaiagent.source=replay`` and written already-sent, and no
production view or curated dataset changed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._replay_isolation_harness import (
    AGENT_NAME,
    RERUN_TABLES,
    hits,
    new_rows,
    prepare_world,
    script_model,
    side_lines,
    snapshot,
    spans_of,
    teardown_world,
)

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

MODES = ["recorded", "live", "deterministic"]


@pytest.fixture
def world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_server: Any,
    decisions_stub: Any,
    official_mcp_server: Any,
) -> Any:
    try:
        yield prepare_world(
            tmp_path, monkeypatch, capture_server, decisions_stub, official_mcp_server
        )
    finally:
        teardown_world()


# ---------------------------------------------------------------------------
# The surfaces
# ---------------------------------------------------------------------------


def _same_process(w: Any, mode: str) -> str:
    from fastaiagent.trace.replay import Replay

    script_model(w.stub)
    result = Replay.load(w.original_trace_id).fork_at(0).with_determinism(mode).rerun()
    return str(result.trace_id)


def _ui(w: Any, mode: str, *, with_agent: bool) -> str:
    from fastaiagent.ui.server import build_app

    script_model(w.stub)
    runners = [w.agent] if with_agent else None
    client = TestClient(build_app(db_path=str(w.db), no_auth=True, runners=runners))
    fork = client.post(f"/api/replay/{w.original_trace_id}/fork", json={"step": 0})
    assert fork.status_code == 200, fork.text
    fork_id = fork.json()["fork_id"]
    # ``determinism`` is new in 1.89.0; 1.88.0 ignores it and reruns live.
    client.patch(f"/api/replay/forks/{fork_id}", json={"determinism": mode})
    r = client.post(f"/api/replay/forks/{fork_id}/rerun")
    assert r.status_code == 200, r.text
    return str(r.json()["new_trace_id"])


_CHILD = """
import json, sys
from fastaiagent.trace.replay import Replay
r = Replay.load(sys.argv[1]).fork_at(0).with_determinism(sys.argv[2]).rerun()
print(json.dumps({"trace_id": r.trace_id}))
"""


def _child_env(w: Any) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("FASTAIAGENT_")}
    env.update(
        FASTAIAGENT_LOCAL_DB=str(w.db),
        FASTAIAGENT_ALLOW_PRIVATE_NETWORKS="1",
        OPENAI_API_KEY="stub-key",
    )
    return env


def _fresh_process(w: Any, mode: str) -> str:
    script_model(w.stub)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD, w.original_trace_id, mode],
        capture_output=True,
        text=True,
        env=_child_env(w),
        cwd=str(w.db.parent.parent),
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return str(json.loads(out.stdout.strip().splitlines()[-1])["trace_id"])


def _cli(w: Any, mode: str) -> str:
    script_model(w.stub)
    out_file = w.db.parent / f"cli-{mode}.json"
    args = ["replay", "fork", w.original_trace_id, "--output", str(out_file)]
    if mode != "live":
        args += ["--mode", mode]  # new in 1.89.0
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from fastaiagent.cli.main import app; "
            "sys.argv = ['fastaiagent'] + sys.argv[1:]; app()",
            *args,
        ],
        capture_output=True,
        text=True,
        env=_child_env(w),
        cwd=str(w.db.parent.parent),
        timeout=120,
    )
    assert out.returncode == 0, (out.stdout + out.stderr)[-2000:]
    return str(json.loads(out_file.read_text())["trace_id"])


SURFACES = {
    "same_process": _same_process,
    "ui_noagent": lambda w, m: _ui(w, m, with_agent=False),
    "ui_agent": lambda w, m: _ui(w, m, with_agent=True),
    "fresh_process": _fresh_process,
    "cli": _cli,
}


def _views(db: Path, *, include: bool = False) -> dict[str, Any]:
    from fastaiagent.ui.server import build_app

    client = TestClient(build_app(db_path=str(db), no_auth=True))
    q = "?include_experiments=true" if include else ""
    analytics = client.get(f"/api/analytics{q}").json()["summary"]["trace_count"]
    overview = client.get(f"/api/overview{q}").json()["traces_last_24h"]
    agents = {
        a["agent_name"]: a.get("run_count") for a in client.get(f"/api/agents{q}").json()["agents"]
    }
    return {"analytics": analytics, "overview": overview, "agent_runs": agents.get(AGENT_NAME)}


# ---------------------------------------------------------------------------
# The default cells
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("surface", list(SURFACES))
def test_a_default_rerun_escapes_nowhere(world: Any, surface: str, mode: str) -> None:
    from fastaiagent.eval.curate import curate_from_traces

    views_before = _views(world.db)
    before = snapshot(world)

    rerun_trace = SURFACES[surface](world, mode)

    after = snapshot(world)
    where = f"{surface}/{mode}"
    # 1, 2: no side effect from any tool type.
    assert side_lines(world) == before["side"], f"{where}: the rerun charged the card"
    assert hits(world, "/refunds") == before["hits"].count("/refunds"), (
        f"{where}: the rerun re-POSTed the refund"
    )
    assert after["mcp_calls"] == before["mcp_calls"], f"{where}: the rerun called the MCP server"
    # 5: nothing but the rerun's own trace was written.
    added = new_rows(before, after)
    stray = {t: n for t, n in added.items() if not t.startswith(RERUN_TABLES)}
    assert not stray, f"{where}: the rerun wrote rows outside its trace: {stray}"
    # 3, 4: the rerun is marked as a replay, and written as already sent.
    spans = spans_of(world.db, rerun_trace)
    assert spans, f"{where}: the rerun left no trace"
    assert {s["attributes"].get("fastaiagent.source") for s in spans} == {"replay"}, (
        f"{where}: rerun spans are not marked as a replay"
    )
    root = next(s for s in spans if not s["parent_span_id"])
    assert root["attributes"].get("fastaiagent.replay.source_trace_id") == world.original_trace_id
    assert all(s["synced"] == 2 for s in spans), f"{where}: the rerun would be exported"
    # 6: production views did not move; asking for experiments shows it.
    assert _views(world.db) == views_before, f"{where}: the rerun counted as production"
    assert _views(world.db, include=True)["analytics"] == views_before["analytics"] + 1
    # 7: curation never takes the rerun as gold.
    curated = {it["trace_id"] for it in curate_from_traces(agent=AGENT_NAME, limit=None)}
    assert curated == {world.original_trace_id}, f"{where}: the rerun was curated"


def test_a_rerun_is_neither_exported_nor_registered(world: Any, capture_server: Any) -> None:
    """Connected: no rerun span reaches the plane, and no agent definition is pushed."""
    from fastaiagent._platform import push
    from fastaiagent.client import _connection
    from fastaiagent.trace.platform_export import PlatformSpanExporter

    original_ids = {world.original_trace_id}
    saved = (_connection.api_key, _connection.target, _connection.project_id)
    _connection.api_key, _connection.target, _connection.project_id = (
        "test-key",
        capture_server.url,
        "test-proj",
    )
    push.reset_registration_state()
    try:
        PlatformSpanExporter().export([])  # drain the original run
        capture_server.requests.clear()
        rerun_trace = _same_process(world, "recorded")
        PlatformSpanExporter().export([])
    finally:
        _connection.api_key, _connection.target, _connection.project_id = saved
        push.reset_registration_state()

    ingested = {
        span.get("trace_id")
        for r in capture_server.ingest_requests
        for span in (r["body"] or {}).get("spans", [])
    }
    assert rerun_trace not in ingested, "the rerun was exported as a real run"
    assert ingested <= original_ids, f"unexpected traces were exported: {ingested - original_ids}"
    pushes = [r for r in capture_server.requests if r["path"].endswith("/sdk/agents")]
    assert pushes == [], "the rerun pushed its rebuilt agent definition to the plane"


def test_a_connected_rerun_keeps_domain_guardrails_but_is_not_governed(
    isolated_local_db: Path,
) -> None:
    """Docs truth (D10): a rerun still enforces the plane's domain-wide guardrails
    (edge enforcement runs locally), but no managed approval policy gates it: it
    has no ``agent_id``, and it never registers to get one."""
    import fastaiagent as fa
    from fastaiagent._platform import push
    from fastaiagent.testing import FunctionModel
    from fastaiagent.trace import Replay, TraceStore, otel
    from tests._governance_plane import AGENT_ID, GATED_TOOL, reset_connection, serve

    moved: list[int] = []

    def transfer_funds(amount: int, to: str) -> str:
        moved.append(amount)
        return f"sent {amount} to {to}"

    def responder(messages: Any) -> Any:
        if messages[-1].role.value != "tool":
            return "", [{"name": GATED_TOOL, "arguments": {"amount": 5, "to": "Bob"}}]
        return "Done."

    with serve() as (plane, url):
        plane.gated_decision = "allow"
        plane.accept_agent_push = True
        plane.guardrail_rules = [
            {
                "name": "domain-no-ssn",
                "guardrail_type": "output",
                "validation_mode": "blocking",
                "implementation_type": "regex",
                "config": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b"},
                "tripwire_message": "SSN blocked",
                "on_error": "block",
                "agent_ids": [],  # domain-wide
            }
        ]
        fa.connect(api_key="fa_k_gov_test", target=url)
        try:
            agent = fa.Agent(
                name="banker-replay",
                agent_id=AGENT_ID,
                llm=FunctionModel(responder),
                tools=[fa.FunctionTool(name=GATED_TOOL, fn=transfer_funds, register=False)],
            )
            original = agent.run("Send Bob $5")
            assert moved == [5]
            decided = [c for c in plane.decide_calls if c.get("tool_name") == GATED_TOOL]
            assert len(decided) == 1, "the original run should be governed (non-vacuous)"

            push.reset_registration_state()  # a rerun that registered would push now
            before = len(plane.paths)
            rerun = (
                Replay.load(original.trace_id)
                .fork_at(0)
                .with_determinism("recorded", on_miss="error")
                .with_tool_override(GATED_TOOL, transfer_funds)
                .rerun()
            )
            otel.get_tracer_provider().force_flush()
            during = plane.paths[before:]

            assert moved == [5, 5], "the override should have run"
            assert not [p for p in during if p.endswith("/policy/decide")], (
                "a managed approval policy gated the rerun"
            )
            assert not [p for p in during if p.endswith("/sdk/agents")], (
                "the rerun pushed its rebuilt agent"
            )
            names = [s.name for s in TraceStore().get_trace(str(rerun.trace_id)).spans]
            assert "guardrail.domain-no-ssn" in names, (
                f"the domain-wide guardrail did not run on the rerun: {names}"
            )
        finally:
            push.reset_registration_state()
            reset_connection()
