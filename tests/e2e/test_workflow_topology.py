"""E2E tests for the workflow topology endpoint.

Hits the real FastAPI app with real registered runners (Chain / Swarm /
Supervisor) and asserts the JSON shape the frontend depends on. No
mocking — uses the in-memory ``MockLLMClient`` from ``tests.conftest``
because the endpoint never calls the LLM, it only inspects structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.agent import Agent, Supervisor, Swarm, Worker  # noqa: E402
from fastaiagent.chain import Chain  # noqa: E402
from fastaiagent.chain.node import NodeType  # noqa: E402
from fastaiagent.tool import tool as tool_decorator  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

from tests.conftest import MockLLMClient  # noqa: E402


@tool_decorator(description="Search docs")
def search_docs(query: str) -> str:
    return f"results for {query}"


@tool_decorator(description="Lookup a customer")
def lookup_customer(id: str) -> str:
    return f"customer {id}"


def _empty_db(tmp_path: Path) -> Path:
    db = init_local_db(tmp_path / "local.db")
    db.close()
    return tmp_path / "local.db"


def _client(db_path: Path, *runners) -> TestClient:
    app = build_app(db_path=str(db_path), no_auth=True, runners=list(runners))
    return TestClient(app)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


def test_chain_topology_returns_full_graph(tmp_path: Path) -> None:
    """Real Chain: 5 nodes including HITL + conditional edge."""
    llm = MockLLMClient()
    researcher = Agent(name="researcher", llm=llm, tools=[search_docs, lookup_customer])
    writer = Agent(name="writer", llm=llm)
    critic = Agent(name="critic", llm=llm)

    chain = Chain("refund-flow")
    chain.add_node("research", agent=researcher)
    chain.add_node("approval", type=NodeType.hitl, name="Manager approval")
    chain.add_node("draft", agent=writer)
    chain.add_node("review", agent=critic)
    chain.add_node("notify_rejection", type=NodeType.end, name="Notify rejection")

    chain.connect("research", "approval")
    chain.connect("approval", "draft", condition="approved == True")
    chain.connect("approval", "notify_rejection", condition="approved == False")
    chain.connect("draft", "review")

    db = _empty_db(tmp_path)
    client = _client(db, chain)

    r = client.get("/api/workflows/chain/refund-flow/topology")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["name"] == "refund-flow"
    assert body["type"] == "chain"
    assert body["entrypoint"] == "research"

    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {"research", "approval", "draft", "review", "notify_rejection"}

    types_by_id = {n["id"]: n["type"] for n in body["nodes"]}
    assert types_by_id["research"] == "agent"
    assert types_by_id["approval"] == "hitl"
    assert types_by_id["notify_rejection"] == "end"

    # Node-level metadata: agent nodes carry model + tool_count.
    research_node = next(n for n in body["nodes"] if n["id"] == "research")
    assert research_node["agent_name"] == "researcher"
    assert research_node["tool_count"] == 2
    assert research_node["model"] == "mock-model"

    # Edges: 4 total, with one conditional pair carrying explicit conditions.
    edges = body["edges"]
    assert len(edges) == 4
    by_pair = {(e["from"], e["to"]): e for e in edges}
    assert by_pair[("research", "approval")]["type"] == "sequential"
    assert by_pair[("approval", "draft")]["type"] == "conditional"
    assert by_pair[("approval", "draft")]["condition"] == "approved == True"
    assert by_pair[("approval", "notify_rejection")]["condition"] == "approved == False"

    # Tools fan out by owner.
    tool_pairs = {(t["owner"], t["name"]) for t in body["tools"]}
    assert ("research", "search_docs") in tool_pairs
    assert ("research", "lookup_customer") in tool_pairs

    # KB list is always present (empty when none registered).
    assert body["knowledge_bases"] == []


# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------


def test_swarm_topology_emits_handoff_edges(tmp_path: Path) -> None:
    llm = MockLLMClient()
    a = Agent(name="researcher", llm=llm)
    b = Agent(name="writer", llm=llm)
    c = Agent(name="critic", llm=llm)
    swarm = Swarm(
        name="content_team",
        agents=[a, b, c],
        entrypoint="researcher",
        handoffs={
            "researcher": ["writer"],
            "writer": ["critic"],
            "critic": ["writer"],
        },
        max_handoffs=4,
    )

    db = _empty_db(tmp_path)
    client = _client(db, swarm)

    r = client.get("/api/workflows/swarm/content_team/topology")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["type"] == "swarm"
    assert body["entrypoint"] == "researcher"
    assert body["max_handoffs"] == 4

    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {"researcher", "writer", "critic"}
    assert all(n["type"] == "agent" for n in body["nodes"])

    # All handoff edges present, all tagged "handoff".
    pairs = {(e["from"], e["to"]) for e in body["edges"]}
    assert pairs == {
        ("researcher", "writer"),
        ("writer", "critic"),
        ("critic", "writer"),
    }
    assert all(e["type"] == "handoff" for e in body["edges"])


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def test_supervisor_topology_emits_delegation_edges(tmp_path: Path) -> None:
    llm = MockLLMClient()
    researcher = Agent(name="researcher", llm=llm, tools=[search_docs])
    writer = Agent(name="writer", llm=llm)
    sup = Supervisor(
        name="team-lead",
        llm=llm,
        workers=[
            Worker(agent=researcher, role="researcher", description="Searches"),
            Worker(agent=writer, role="writer", description="Writes"),
        ],
    )

    db = _empty_db(tmp_path)
    client = _client(db, sup)

    r = client.get("/api/workflows/supervisor/team-lead/topology")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["type"] == "supervisor"
    sup_id = "supervisor:team-lead"
    assert body["entrypoint"] == sup_id

    node_ids = {n["id"] for n in body["nodes"]}
    assert node_ids == {sup_id, "worker:researcher", "worker:writer"}

    delegation_targets = {e["to"] for e in body["edges"] if e["from"] == sup_id}
    assert delegation_targets == {"worker:researcher", "worker:writer"}
    assert all(e["type"] == "delegation" for e in body["edges"])

    # Tool fan-out via worker:role owner ids.
    tool_owners = {t["owner"] for t in body["tools"]}
    assert "worker:researcher" in tool_owners


# ---------------------------------------------------------------------------
# Empty / unregistered
# ---------------------------------------------------------------------------


def test_topology_404_when_runner_not_registered(tmp_path: Path) -> None:
    db = _empty_db(tmp_path)
    client = _client(db)  # no runners
    r = client.get("/api/workflows/chain/missing/topology")
    assert r.status_code == 404
    assert "build_app(runners=" in r.text


def test_topology_400_for_invalid_runner_type(tmp_path: Path) -> None:
    db = _empty_db(tmp_path)
    client = _client(db)
    r = client.get("/api/workflows/notachain/foo/topology")
    assert r.status_code == 400


def test_topology_404_when_type_mismatches_registered_runner(tmp_path: Path) -> None:
    """If a Chain named 'foo' is registered, /workflows/swarm/foo/topology 404s."""
    llm = MockLLMClient()
    chain = Chain("foo")
    chain.add_node("only", agent=Agent(name="x", llm=llm))
    db = _empty_db(tmp_path)
    client = _client(db, chain)
    r = client.get("/api/workflows/swarm/foo/topology")
    assert r.status_code == 404


def test_workflows_list_includes_registered_flag(tmp_path: Path) -> None:
    db = _empty_db(tmp_path)

    # Empty case: no runners
    client_empty = _client(db)
    r = client_empty.get("/api/workflows")
    assert r.status_code == 200
    body = r.json()
    assert body["registered"] is False

    # Registered case
    llm = MockLLMClient()
    chain = Chain("c")
    chain.add_node("only", agent=Agent(name="x", llm=llm))
    client_with = _client(db, chain)
    r2 = client_with.get("/api/workflows")
    assert r2.status_code == 200
    assert r2.json()["registered"] is True
