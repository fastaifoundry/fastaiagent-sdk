"""Route + SDK tests for the Agent Tools directory.

Real SQLite, real OTel spans, real Tool subclasses — no mocks of the
subject under test. Exercises:

  - The new ``Tool.origin`` taxonomy survives ``to_dict()`` and reaches
    the span store.
  - ``LocalKB.as_tool()`` overrides origin to "kb".
  - The ``GET /api/agents/<name>/tools`` route aggregates both
    registered (from the agent root span's ``agent.tools`` JSON) and
    used (from descendant ``tool.*`` spans) correctly.
  - Cross-referencing flags registered-but-never-used tools.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402
from fastaiagent.tool.base import Tool  # noqa: E402
from fastaiagent.tool.function import FunctionTool  # noqa: E402
from fastaiagent.tool.mcp import MCPTool  # noqa: E402
from fastaiagent.tool.rest import RESTTool  # noqa: E402


# ─── SDK-level: origin attribute survives to_dict() ────────────────────────


class _MyCustomTool(Tool):
    """User-defined tool subclass — should fall back to 'custom' origin."""


def test_origin_defaults_cover_every_shipped_subclass():
    assert Tool.origin == "custom"
    assert FunctionTool.origin == "function"
    assert MCPTool.origin == "mcp"
    assert RESTTool.origin == "rest"


def test_origin_round_trips_through_to_dict():
    fn = FunctionTool(name="greet", fn=lambda name: f"Hello, {name}!")
    custom = _MyCustomTool(name="bespoke", description="home-rolled")
    rest = RESTTool(name="weather", url="https://api.example.com/weather")
    mcp = MCPTool(name="search", server_url="http://localhost:3000")
    assert fn.to_dict()["origin"] == "function"
    assert custom.to_dict()["origin"] == "custom"
    assert rest.to_dict()["origin"] == "rest"
    assert mcp.to_dict()["origin"] == "mcp"


def test_kb_as_tool_overrides_origin_to_kb():
    pytest.importorskip("faiss")
    from fastaiagent.kb.local import LocalKB

    kb = LocalKB(name="docs", path="/tmp/kb-test-origin", chunk_size=120)
    try:
        tool = kb.as_tool()
        assert tool.origin == "kb"
        assert tool.to_dict()["origin"] == "kb"
    finally:
        kb.close()


# ─── Route-level: /api/agents/<name>/tools ────────────────────────────────


def _insert_agent_root(
    db: SQLiteHelper,
    *,
    trace_id: str,
    agent_name: str,
    registered_tools: list[dict[str, object]],
    ago_minutes: int = 5,
) -> str:
    """Insert one agent.<name> root span carrying agent.tools JSON."""
    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(minutes=ago_minutes)).isoformat()
    end = now.isoformat()
    span_id = f"s-root-{trace_id}"
    attrs = {
        "agent.name": agent_name,
        "agent.tools": json.dumps(registered_tools),
    }
    db.execute(
        """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                               start_time, end_time, status, attributes, events)
           VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]')""",
        (span_id, trace_id, f"agent.{agent_name}", start, end, json.dumps(attrs)),
    )
    return span_id


_TOOL_SPAN_COUNTER: dict[str, int] = {}


def _insert_tool_span(
    db: SQLiteHelper,
    *,
    trace_id: str,
    parent_span_id: str,
    tool_name: str,
    origin: str,
    status: str = "OK",
    tool_status: str = "ok",
    latency_ms: int = 80,
) -> None:
    now = datetime.now(tz=timezone.utc)
    start = (now - timedelta(milliseconds=latency_ms)).isoformat()
    end = now.isoformat()
    # Unique span_id per call so one tool can be invoked multiple times
    # within a single trace (the two lookup_order calls do exactly that).
    key = f"{trace_id}:{tool_name}"
    _TOOL_SPAN_COUNTER[key] = _TOOL_SPAN_COUNTER.get(key, -1) + 1
    span_id = f"s-tool-{tool_name}-{trace_id}-{_TOOL_SPAN_COUNTER[key]}"
    attrs = {"tool.name": tool_name, "tool.origin": origin, "tool.status": tool_status}
    db.execute(
        """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                               start_time, end_time, status, attributes, events)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]')""",
        (
            span_id,
            trace_id,
            parent_span_id,
            f"tool.{tool_name}",
            start,
            end,
            status,
            json.dumps(attrs),
        ),
    )


@pytest.fixture
def app_env(tmp_path: Path):
    db_path = tmp_path / "local.db"
    init_local_db(db_path).close()
    with SQLiteHelper(db_path) as db:
        # Agent root: registered with 3 tools — one decorator, one MCP, one KB.
        # Registered list is intentionally different from what actually gets
        # called so we can test the used/registered cross-reference.
        registered = [
            {
                "name": "lookup_order",
                "description": "Look up an order by id",
                "origin": "function",
            },
            {
                "name": "search_support-docs",
                "description": "Search the 'support-docs' knowledge base",
                "origin": "kb",
            },
            {
                "name": "file_search",
                "description": "MCP: searches local files",
                "origin": "mcp",
            },
        ]
        root = _insert_agent_root(
            db,
            trace_id="t-1",
            agent_name="support-bot",
            registered_tools=registered,
        )
        # Used: only 2 of the 3 registered ever got called, plus an
        # unregistered hallucinated name. One call fails.
        _insert_tool_span(
            db,
            trace_id="t-1",
            parent_span_id=root,
            tool_name="lookup_order",
            origin="function",
        )
        _insert_tool_span(
            db,
            trace_id="t-1",
            parent_span_id=root,
            tool_name="lookup_order",
            origin="function",
            tool_status="error",
            status="ERROR",
        )
        _insert_tool_span(
            db,
            trace_id="t-1",
            parent_span_id=root,
            tool_name="search_support-docs",
            origin="kb",
        )
        _insert_tool_span(
            db,
            trace_id="t-1",
            parent_span_id=root,
            tool_name="phantom_tool",
            origin="unknown",
            tool_status="unknown",
        )
    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
    )
    return app


def test_returns_registered_tools_with_origin(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/agents/support-bot/tools")
    assert r.status_code == 200
    body = r.json()
    reg = {t["name"]: t for t in body["registered"]}
    assert reg["lookup_order"]["origin"] == "function"
    assert reg["search_support-docs"]["origin"] == "kb"
    assert reg["file_search"]["origin"] == "mcp"


def test_returns_used_tools_with_call_counts(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/agents/support-bot/tools")
    assert r.status_code == 200
    used = {t["name"]: t for t in r.json()["used"]}
    assert used["lookup_order"]["call_count"] == 2
    assert used["lookup_order"]["error_count"] == 1
    # Success rate is 1/2.
    assert used["lookup_order"]["success_rate"] == pytest.approx(0.5)
    assert used["search_support-docs"]["call_count"] == 1
    assert used["phantom_tool"]["call_count"] == 1


def test_used_and_registered_are_cross_referenced(app_env):
    with TestClient(app_env) as c:
        body = c.get("/api/agents/support-bot/tools").json()
    reg = {t["name"]: t for t in body["registered"]}
    used = {t["name"]: t for t in body["used"]}
    # file_search was registered but never called.
    assert reg["file_search"]["used"] is False
    # lookup_order was both registered and used.
    assert reg["lookup_order"]["used"] is True
    assert used["lookup_order"]["registered"] is True
    # phantom_tool was called but never registered (hallucinated).
    assert used["phantom_tool"]["registered"] is False


def test_route_404_for_unknown_agent(app_env):
    with TestClient(app_env) as c:
        r = c.get("/api/agents/ghost/tools")
    # Unknown agent returns 200 with empty lists — /tools is forgiving,
    # since "no traces yet" should render fine (empty state) rather than
    # error.
    assert r.status_code == 200
    assert r.json() == {
        "agent_name": "ghost",
        "registered": [],
        "used": [],
        # No live object and no spans, so the (empty) registered list is
        # still nominally trace-derived.
        "source": "trace",
    }


# ─── Registered runners: `fastaiagent ui --agent` / build_app(runners=...) ──
#
# Before the --agent flag, ctx.runners was always empty for CLI-launched UIs,
# so none of the code paths below were reachable in practice.


def _live_agent_fixture(name: str = "live-bot"):
    """A real Agent with real tools, deterministic LLM, never run."""
    from fastaiagent.agent.agent import Agent
    from fastaiagent.testing import TestModel

    lookup = FunctionTool(
        name="lookup_order",
        fn=lambda order_id: f"order {order_id}",
        description="Look up an order by id",
        replay_class="read_only",
    )
    refund = FunctionTool(
        name="issue_refund",
        fn=lambda order_id: f"refunded {order_id}",
        description="Refund an order",
        replay_class="side_effecting",
    )
    return Agent(
        name=name,
        system_prompt="You help customers.",
        llm=TestModel(response="ok"),
        tools=[lookup, refund],
    )


@pytest.fixture
def app_with_registered_agent(tmp_path: Path):
    """A UI bound to a registered agent that has never produced a span."""
    db_path = tmp_path / "registered.db"
    init_local_db(db_path).close()
    agent = _live_agent_fixture()
    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
        runners=[agent],
    )
    return app, agent


def test_registered_agent_is_listed_before_it_has_run(app_with_registered_agent):
    app, _agent = app_with_registered_agent
    with TestClient(app) as c:
        body = c.get("/api/agents").json()
    names = {a["agent_name"] for a in body["agents"]}
    assert "live-bot" in names
    row = next(a for a in body["agents"] if a["agent_name"] == "live-bot")
    assert row["run_count"] == 0


def test_registered_agent_detail_is_200_not_404(app_with_registered_agent):
    """Regression guard: ``list_agents`` back-fills registered names, so
    ``get_agent`` must too or the directory page links to a 404."""
    app, _agent = app_with_registered_agent
    with TestClient(app) as c:
        r = c.get("/api/agents/live-bot")
    assert r.status_code == 200, r.text
    assert r.json()["agent_name"] == "live-bot"
    assert r.json()["run_count"] == 0


def test_unregistered_unknown_agent_still_404s(app_with_registered_agent):
    app, _agent = app_with_registered_agent
    with TestClient(app) as c:
        assert c.get("/api/agents/ghost").status_code == 404


def test_registered_tools_come_from_the_live_object(app_with_registered_agent):
    app, _agent = app_with_registered_agent
    with TestClient(app) as c:
        body = c.get("/api/agents/live-bot/tools").json()
    assert body["source"] == "runtime"
    reg = {t["name"]: t for t in body["registered"]}
    assert set(reg) == {"lookup_order", "issue_refund"}
    assert reg["lookup_order"]["replay_class"] == "read_only"
    assert reg["issue_refund"]["replay_class"] == "side_effecting"
    assert reg["lookup_order"]["description"] == "Look up an order by id"
    assert reg["lookup_order"]["origin"] == "function"
    # Never run, so nothing is used and nothing cross-references as used.
    assert body["used"] == []
    assert all(t["used"] is False for t in body["registered"])


def test_live_object_wins_over_stale_span_but_used_still_from_spans(tmp_path: Path):
    """The registered half comes from the live object; the used half is
    still real runtime history the object can't know about."""
    db_path = tmp_path / "mixed.db"
    init_local_db(db_path).close()
    with SQLiteHelper(db_path) as db:
        # A stale span claiming a tool the agent no longer has.
        root = _insert_agent_root(
            db,
            trace_id="t-stale",
            agent_name="live-bot",
            registered_tools=[
                {"name": "removed_tool", "description": "gone", "origin": "function"}
            ],
        )
        _insert_tool_span(
            db,
            trace_id="t-stale",
            parent_span_id=root,
            tool_name="lookup_order",
            origin="function",
        )
    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
        runners=[_live_agent_fixture()],
    )
    with TestClient(app) as c:
        body = c.get("/api/agents/live-bot/tools").json()

    assert body["source"] == "runtime"
    reg_names = {t["name"] for t in body["registered"]}
    # The stale span's tool is gone; the live tools are there.
    assert "removed_tool" not in reg_names
    assert reg_names == {"lookup_order", "issue_refund"}
    # Used still reflects the real call, and cross-referencing works.
    used = {t["name"]: t for t in body["used"]}
    assert used["lookup_order"]["call_count"] == 1
    assert used["lookup_order"]["registered"] is True
    reg = {t["name"]: t for t in body["registered"]}
    assert reg["lookup_order"]["used"] is True
    assert reg["issue_refund"]["used"] is False


def test_span_derived_path_still_used_when_nothing_registered(app_env):
    """No runners → unchanged behaviour, now labelled."""
    with TestClient(app_env) as c:
        body = c.get("/api/agents/support-bot/tools").json()
    assert body["source"] == "trace"
    assert {t["name"] for t in body["registered"]} == {
        "lookup_order",
        "search_support-docs",
        "file_search",
    }
