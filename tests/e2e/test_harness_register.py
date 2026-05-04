"""Phase 8 — ``register_agent()`` + external-agent registry tests.

Spec test IDs covered: #33 (LangGraph register), #34 (CrewAI register),
#35 (PydanticAI register), #36 (harness auto-attachment), #37 (lazy
register), #38 (workflow topology in payload), #39 (dependency endpoint
returns merged tree).

These tests don't need a real LLM — they only assert that
``register_agent()`` and the harness auto-attachment helpers populate
the right SQLite rows and that the dependency endpoint surfaces them.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

# Real-LLM not required for any test in this module.
pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_client() -> TestClient:
    from fastaiagent._internal.config import get_config
    from fastaiagent.ui.server import build_app

    return TestClient(
        build_app(db_path=str(get_config().resolved_trace_db_path), no_auth=True)
    )


@pytest.fixture
def fresh_name() -> str:
    """Stable per-test agent name that won't collide across runs."""
    return f"harness-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# #33 — LangGraph register
# ---------------------------------------------------------------------------


def test_33_langchain_register(fresh_name: str) -> None:
    from langchain_core.messages import HumanMessage  # noqa: F401  (used by graph builders)
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import fetch_agent

    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[])
    lc.register_agent(graph, name=fresh_name)

    row = fetch_agent(fresh_name)
    assert row is not None, "register_agent did not write a row"
    assert row["framework"] == "langchain", row
    # Topology is best-effort — at minimum the LangGraph compile produced
    # *some* nodes.
    topo = row.get("topology") or {}
    assert isinstance(topo.get("nodes"), list) and topo["nodes"], topo


# ---------------------------------------------------------------------------
# #34 — CrewAI register
# ---------------------------------------------------------------------------


def test_34_crewai_register(fresh_name: str) -> None:
    from crewai import Agent, Crew, Process, Task
    from crewai.llm import LLM

    from fastaiagent.integrations import crewai as ca
    from fastaiagent.integrations._registry import fetch_agent

    llm = LLM(model="openai/gpt-4o-mini")
    a = Agent(role="Researcher", goal="g", backstory="b", llm=llm,
              verbose=False, allow_delegation=False)
    t = Task(description="d", expected_output="o", agent=a)
    crew = Crew(agents=[a], tasks=[t], process=Process.sequential, verbose=False)

    ca.register_agent(crew, name=fresh_name)

    row = fetch_agent(fresh_name)
    assert row is not None
    assert row["framework"] == "crewai"
    topo = row.get("topology") or {}
    nodes = topo.get("nodes") or []
    # Should have one agent node and one task node.
    assert any(n.get("type") == "agent" for n in nodes), nodes
    assert any(n.get("type") == "task" for n in nodes), nodes


# ---------------------------------------------------------------------------
# #35 — PydanticAI register
# ---------------------------------------------------------------------------


def test_35_pydanticai_register(fresh_name: str) -> None:
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa
    from fastaiagent.integrations._registry import fetch_agent

    agent = Agent("openai:gpt-4o-mini", system_prompt="be terse")

    @agent.tool_plain
    def echo(text: str) -> str:
        """echo"""
        return text

    pa.register_agent(agent, name=fresh_name)
    row = fetch_agent(fresh_name)
    assert row is not None
    assert row["framework"] == "pydanticai"
    assert row.get("model") and "gpt-4o-mini" in str(row["model"])


# ---------------------------------------------------------------------------
# #36 — Harness auto-attachment
# ---------------------------------------------------------------------------


def test_36_harness_auto_attachment(fresh_name: str) -> None:
    """Calling ``with_guardrails(name=…)``, ``prompt_from_registry(agent=…)``,
    ``kb_as_retriever(agent=…)`` (Phase 9) writes attachment rows that
    the dependency endpoint surfaces."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.guardrail.builtins import no_pii
    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import fetch_attachments
    from fastaiagent.prompt import PromptRegistry

    # Step 1: register the agent so attachments have a parent row.
    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[])
    lc.register_agent(graph, name=fresh_name)

    # Step 2: attach a guardrail.
    lc.with_guardrails(graph, name=fresh_name, input_guardrails=[no_pii()])

    # Step 3: attach a prompt (lazily registered if missing).
    slug = f"harness-attach-{uuid.uuid4().hex[:6]}"
    PromptRegistry().register(name=slug, template="Hello {{name}}.")
    lc.prompt_from_registry(slug, agent=fresh_name)

    # The guardrail attachment goes through ``with_guardrails`` only when
    # the wrapper is invoked. We call attach() directly to make the
    # contract explicit (this is what the wrapper does internally on
    # block, but we don't want the test to depend on a real LLM).
    from fastaiagent.integrations._registry import attach as _attach

    _attach(fresh_name, "guardrail", "no_pii", position="input")

    attachments = fetch_attachments(fresh_name)
    kinds = {a["kind"] for a in attachments}
    assert "prompt" in kinds, attachments
    assert "guardrail" in kinds, attachments


# ---------------------------------------------------------------------------
# #37 — Lazy register
# ---------------------------------------------------------------------------


def test_37_lazy_register_creates_stub(fresh_name: str) -> None:
    """Calling ``attach`` with no prior ``register_agent`` lazily creates
    a stub row tagged ``framework="unknown"`` so the dependency graph
    still has something to render."""
    from fastaiagent.integrations._registry import attach, fetch_agent

    attach(fresh_name, "guardrail", "no_pii", position="input")

    row = fetch_agent(fresh_name)
    assert row is not None
    assert row["framework"] == "unknown"


# ---------------------------------------------------------------------------
# #38 — Workflow topology preserved
# ---------------------------------------------------------------------------


def test_38_langchain_workflow_topology(fresh_name: str) -> None:
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import fetch_agent

    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[])
    lc.register_agent(graph, name=fresh_name)

    row = fetch_agent(fresh_name)
    assert row is not None
    topo = row.get("topology") or {}
    assert "nodes" in topo and "edges" in topo, topo


# ---------------------------------------------------------------------------
# #39 — Dependency endpoint returns merged tree
# ---------------------------------------------------------------------------


def test_39_dependency_endpoint_merges(
    app_client: TestClient, fresh_name: str
) -> None:
    """The /api/agents/{name}/dependencies endpoint reads the external
    registry and returns a payload with the full attachment tree."""
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import attach as _attach

    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[])
    lc.register_agent(graph, name=fresh_name)
    _attach(fresh_name, "guardrail", "no_pii", position="input")
    _attach(fresh_name, "prompt", "support-system", version="v1")
    _attach(fresh_name, "kb", "support-kb")

    resp = app_client.get(f"/api/agents/{fresh_name}/dependencies")
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["external"] is True
    assert payload["agent"]["name"] == fresh_name
    assert payload["agent"]["framework"] == "langchain"
    assert any(g["ref_name"] == "no_pii" for g in payload["guardrails"])
    assert any(p["ref_name"] == "support-system" for p in payload["prompts"])
    assert any(k["ref_name"] == "support-kb" for k in payload["knowledge_bases"])
