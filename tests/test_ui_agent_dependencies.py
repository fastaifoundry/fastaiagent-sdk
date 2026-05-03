"""Integration tests for the Agent Dependency Graph endpoint.

Constructs real Agent / Supervisor / Swarm objects (no mocks beyond a
no-op LLMClient — we never make a network call) and asserts that
``GET /api/agents/{name}/dependencies`` reconstructs the structural view
the UI renders.

Sprint 2 / Feature 2.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent import Agent, LLMClient, tool  # noqa: E402
from fastaiagent.agent.swarm import Swarm  # noqa: E402
from fastaiagent.agent.team import Supervisor, Worker  # noqa: E402
from fastaiagent.guardrail.builtins import json_valid, no_pii  # noqa: E402
from fastaiagent.tool.base import Tool, ToolResult  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

# --- Fixtures: a real toolkit, real guardrails, real LLMClient instances. ---


@tool(description="Look up a customer record by id.")
def lookup_record(record_id: str) -> str:
    return f"record-{record_id}"


@tool(description="Return the user's display name.")
def get_user_info(user_id: str) -> str:
    return f"user-{user_id}"


class CustomTool(Tool):
    """User-defined Tool subclass — origin should default to ``custom``."""

    def __init__(self) -> None:
        super().__init__(
            name="custom_thing",
            description="A bespoke tool",
            parameters={"type": "object", "properties": {}},
        )

    async def aexecute(self, arguments: dict, context=None) -> ToolResult:
        return ToolResult(output="ok")


def _llm() -> LLMClient:
    """Return an LLMClient that is *configured* but never called.

    Tests don't invoke ``agent.run()`` — they only introspect the agent's
    static structure, so no API key is needed.
    """
    return LLMClient(provider="openai", model="gpt-4o-mini")


def _build_support_agent() -> Agent:
    return Agent(
        name="support-bot",
        system_prompt=(
            "You are a friendly support agent for {{company}}. "
            "Use {{customer_name}} for the user."
        ),
        llm=_llm(),
        tools=[lookup_record, get_user_info, CustomTool()],
        guardrails=[no_pii(), json_valid()],
    )


def _build_research_agent() -> Agent:
    return Agent(
        name="researcher",
        system_prompt="You research topics by searching the web.",
        llm=_llm(),
        tools=[lookup_record],
    )


def _build_writer_agent() -> Agent:
    return Agent(
        name="writer",
        system_prompt="You write a one-paragraph summary using {{style}}.",
        llm=_llm(),
        tools=[],
    )


@pytest.fixture
def empty_db(temp_dir: Path) -> Path:
    db_path = temp_dir / ".fastaiagent" / "local.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(db_path).close()
    return db_path


# ---------------------------------------------------------------------------
# Single agent
# ---------------------------------------------------------------------------


class TestSingleAgentDependencies:
    def test_renders_tools_guardrails_prompt_model(self, empty_db: Path) -> None:
        agent = _build_support_agent()
        app = build_app(db_path=str(empty_db), no_auth=True, runners=[agent])
        client = TestClient(app)

        r = client.get("/api/agents/support-bot/dependencies")
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["agent"]["name"] == "support-bot"
        assert body["agent"]["type"] == "agent"
        assert body["model"]["provider"] == "openai"
        assert body["model"]["model"] == "gpt-4o-mini"

        tool_names = {t["name"] for t in body["tools"]}
        assert tool_names == {"lookup_record", "get_user_info", "custom_thing"}
        # All three are registered (declared on the Agent).
        assert all(t["registered"] for t in body["tools"])

        guardrail_names = {g["name"] for g in body["guardrails"]}
        assert "no_pii" in guardrail_names
        assert "json_valid" in guardrail_names

        prompts = body["prompts"]
        assert len(prompts) == 1
        assert prompts[0]["name"] == "system_prompt"
        assert set(prompts[0]["variables"]) == {"company", "customer_name"}

        # No KBs registered — list should be empty (not missing).
        assert body["knowledge_bases"] == []
        assert body["sub_agents"] == []

    def test_minimal_agent_renders_just_model(self, empty_db: Path) -> None:
        agent = Agent(name="minimal", llm=_llm())
        app = build_app(db_path=str(empty_db), no_auth=True, runners=[agent])
        client = TestClient(app)

        r = client.get("/api/agents/minimal/dependencies")
        assert r.status_code == 200
        body = r.json()
        assert body["tools"] == []
        assert body["guardrails"] == []
        assert body["prompts"] == []
        assert body["knowledge_bases"] == []
        assert body["sub_agents"] == []
        assert body["model"]["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Supervisor + workers
# ---------------------------------------------------------------------------


class TestSupervisorDependencies:
    def test_supervisor_exposes_workers_as_sub_agents(
        self, empty_db: Path
    ) -> None:
        supervisor = Supervisor(
            name="planner",
            llm=_llm(),
            workers=[
                Worker(
                    agent=_build_research_agent(),
                    role="researcher",
                    description="Searches for info",
                ),
                Worker(
                    agent=_build_writer_agent(),
                    role="writer",
                    description="Writes content",
                ),
            ],
        )
        app = build_app(db_path=str(empty_db), no_auth=True, runners=[supervisor])
        client = TestClient(app)

        r = client.get("/api/agents/planner/dependencies")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["agent"]["type"] == "supervisor"
        assert len(body["sub_agents"]) == 2
        sub_names = [s["agent"]["name"] for s in body["sub_agents"]]
        assert sorted(sub_names) == ["researcher", "writer"]

        # Worker subtree carries its own tools/prompts.
        researcher = next(
            s for s in body["sub_agents"] if s["agent"]["name"] == "researcher"
        )
        assert researcher["agent"]["type"] == "worker"
        assert researcher["role"] == "researcher"
        assert {t["name"] for t in researcher["tools"]} == {"lookup_record"}

        writer = next(
            s for s in body["sub_agents"] if s["agent"]["name"] == "writer"
        )
        # writer has no tools and a single-variable prompt.
        assert writer["tools"] == []
        assert writer["prompts"][0]["variables"] == ["style"]

    def test_query_individual_worker(self, empty_db: Path) -> None:
        """Hitting /agents/<worker>/dependencies returns just that worker."""
        supervisor = Supervisor(
            name="planner",
            llm=_llm(),
            workers=[
                Worker(agent=_build_research_agent(), role="researcher"),
            ],
        )
        app = build_app(db_path=str(empty_db), no_auth=True, runners=[supervisor])
        client = TestClient(app)

        r = client.get("/api/agents/researcher/dependencies")
        assert r.status_code == 200
        body = r.json()
        assert body["agent"]["name"] == "researcher"
        assert body["parent"]["name"] == "planner"
        assert body["parent"]["type"] == "supervisor"


# ---------------------------------------------------------------------------
# Swarm peers + handoffs
# ---------------------------------------------------------------------------


class TestSwarmDependencies:
    def test_swarm_peer_exposes_other_peers_and_handoffs(
        self, empty_db: Path
    ) -> None:
        triage = Agent(name="triage", llm=_llm(), tools=[lookup_record])
        billing = Agent(name="billing", llm=_llm(), tools=[lookup_record])
        support = Agent(name="support", llm=_llm())
        swarm = Swarm(
            name="customer-router",
            agents=[triage, billing, support],
            entrypoint="triage",
        )
        app = build_app(db_path=str(empty_db), no_auth=True, runners=[swarm])
        client = TestClient(app)

        r = client.get("/api/agents/triage/dependencies")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["parent"]["type"] == "swarm"
        assert body["parent"]["name"] == "customer-router"
        peer_names = {p["name"] for p in body["peers"]}
        assert peer_names == {"billing", "support"}
        # Default handoffs allow every peer pair.
        assert any(
            h["from"] == "triage" and h["to"] == "billing"
            for h in body["handoffs"]
        )


# ---------------------------------------------------------------------------
# Degraded / unresolved fallback
# ---------------------------------------------------------------------------


class TestAgentDirectoryListsRegistered:
    """Registered runners should appear in /api/agents even before spans."""

    def test_supervisor_and_workers_listed_without_runs(
        self, empty_db: Path
    ) -> None:
        supervisor = Supervisor(
            name="planner",
            llm=_llm(),
            workers=[
                Worker(agent=_build_research_agent(), role="researcher"),
                Worker(agent=_build_writer_agent(), role="writer"),
            ],
        )
        app = build_app(db_path=str(empty_db), no_auth=True, runners=[supervisor])
        client = TestClient(app)

        body = client.get("/api/agents").json()
        names = {a["agent_name"] for a in body["agents"]}
        assert {"planner", "researcher", "writer"}.issubset(names)
        # Stub stats are zero — they haven't actually run.
        for a in body["agents"]:
            assert a["run_count"] == 0
            assert a["last_run"] == ""

    def test_swarm_peers_listed_without_runs(self, empty_db: Path) -> None:
        triage = Agent(name="triage", llm=_llm())
        billing = Agent(name="billing", llm=_llm())
        swarm = Swarm(
            name="customer-router",
            agents=[triage, billing],
            entrypoint="triage",
        )
        app = build_app(db_path=str(empty_db), no_auth=True, runners=[swarm])
        client = TestClient(app)

        names = {
            a["agent_name"] for a in client.get("/api/agents").json()["agents"]
        }
        assert {"triage", "billing"}.issubset(names)


class TestDegraded:
    def test_unregistered_agent_falls_back_to_spans(
        self, empty_db: Path
    ) -> None:
        """Agent visible in the spans table but not registered in build_app
        returns a degraded payload reconstructed from span attributes.
        """
        # Seed a span that mentions this agent.
        import json

        from fastaiagent._internal.storage import SQLiteHelper

        db = SQLiteHelper(str(empty_db))
        try:
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time,
                    end_time, status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "span-1",
                    "trace-1",
                    None,
                    "agent.legacy-bot",
                    "2026-01-01T00:00:00",
                    "2026-01-01T00:00:01",
                    "OK",
                    json.dumps(
                        {
                            "agent.name": "legacy-bot",
                            "agent.tools": json.dumps(
                                [{"name": "old_tool", "origin": "function"}]
                            ),
                            "agent.llm.provider": "openai",
                            "agent.llm.model": "gpt-3.5-turbo",
                        }
                    ),
                    "[]",
                ),
            )
        finally:
            db.close()

        app = build_app(db_path=str(empty_db), no_auth=True)
        client = TestClient(app)

        r = client.get("/api/agents/legacy-bot/dependencies")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("unresolved") is True
        assert body["model"]["provider"] == "openai"
        assert body["model"]["model"] == "gpt-3.5-turbo"
        assert {t["name"] for t in body["tools"]} == {"old_tool"}

    def test_unknown_agent_404(self, empty_db: Path) -> None:
        # Project-scoped 404 — no spans, no runner, project_id set.
        app = build_app(
            db_path=str(empty_db),
            no_auth=True,
            project_id="my-project",
        )
        client = TestClient(app)
        r = client.get("/api/agents/no-such-agent/dependencies")
        assert r.status_code == 404
