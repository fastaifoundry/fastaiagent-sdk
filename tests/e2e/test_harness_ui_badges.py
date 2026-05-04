"""Phase 4 — Framework badge attribute + filter (real LLM, gated by env).

Spec test IDs covered: #14, #14b, #15.

These tests exercise the REST contract that powers the React badge /
filter. The frontend reads ``TraceRow.framework`` from
``GET /api/traces`` and applies the filter via
``GET /api/traces?framework=langchain``. We verify both directly.
"""

from __future__ import annotations

import os

import pytest

# ``fastapi`` ships in the SDK's ``[ui]`` extra. CI's base test matrix
# doesn't install it, so we ``importorskip`` rather than fail collection.
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))

if not HAS_OPENAI:
    pytest.skip(
        "OPENAI_API_KEY not set — Phase 4 needs at least the OpenAI path",
        allow_module_level=True,
    )

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")
needs_anthropic = pytest.mark.skipif(not HAS_ANTHROPIC, reason="ANTHROPIC_API_KEY not set")

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_client() -> TestClient:
    """FastAPI TestClient pointed at the same local.db the integrations
    write spans into, so the spawned traces are visible to /api/traces.
    """
    from fastaiagent._internal.config import get_config
    from fastaiagent.ui.server import build_app

    db_path = get_config().resolved_trace_db_path
    return TestClient(build_app(db_path=str(db_path), no_auth=True))


def _spawn_langchain_trace() -> None:
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    handler = lc.get_callback_handler()

    @tool
    def echo(text: str) -> str:
        """echo back"""
        return f"echo: {text}"

    graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0), tools=[echo])
    graph.invoke(
        {"messages": [HumanMessage(content="Reply with: ok")]},
        config={"callbacks": [handler]},
    )


def _spawn_crewai_trace() -> None:
    from crewai import Agent, Crew, Process, Task
    from crewai.llm import LLM

    from fastaiagent.integrations import crewai as ca

    ca.enable()
    a = Agent(
        role="R",
        goal="answer",
        backstory="terse",
        llm=LLM(model="openai/gpt-4o-mini", temperature=0),
        verbose=False,
        allow_delegation=False,
    )
    t = Task(description="Reply with: ok", expected_output="ok", agent=a)
    Crew(agents=[a], tasks=[t], process=Process.sequential, verbose=False).kickoff()


def _spawn_pydanticai_trace() -> None:
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    Agent("openai:gpt-4o-mini", system_prompt="Reply with: ok").run_sync("ping")


def _spawn_anthropic_pa_trace() -> None:
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    Agent("anthropic:claude-haiku-4-5", system_prompt="Reply with: ok").run_sync("ping")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@needs_openai
def test_14_framework_attr_lc(app_client: TestClient) -> None:
    """Spec #14 (LC): root span carries fastaiagent.framework=langchain
    and the /api/traces endpoint surfaces it as TraceRow.framework."""
    _spawn_langchain_trace()
    resp = app_client.get("/api/traces?page_size=200")
    assert resp.status_code == 200, resp.text
    rows = resp.json()["rows"]
    assert any(r.get("framework") == "langchain" for r in rows), (
        "no langchain row in /api/traces"
    )


@needs_openai
def test_14_framework_attr_crewai(app_client: TestClient) -> None:
    _spawn_crewai_trace()
    resp = app_client.get("/api/traces?page_size=200")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert any(r.get("framework") == "crewai" for r in rows)


@needs_openai
def test_14_framework_attr_pydanticai(app_client: TestClient) -> None:
    _spawn_pydanticai_trace()
    resp = app_client.get("/api/traces?page_size=200")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert any(r.get("framework") == "pydanticai" for r in rows)


@needs_anthropic
def test_14b_pricing_both_providers(app_client: TestClient) -> None:
    """Spec #14b: cost computed for both OpenAI and Anthropic LLM spans."""
    _spawn_pydanticai_trace()  # OpenAI path
    _spawn_anthropic_pa_trace()  # Anthropic path

    resp = app_client.get("/api/traces?page_size=200&framework=pydanticai")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    # Among the recent pydanticai traces we should see at least two with
    # non-zero cost — the OpenAI one and the Anthropic one.
    costed = [r for r in rows if (r.get("total_cost_usd") or 0) > 0]
    assert len(costed) >= 2, (
        f"expected ≥2 costed pydanticai traces (one per provider), got {costed}"
    )


@needs_openai
def test_15_framework_filter(app_client: TestClient) -> None:
    """Spec #15: ?framework=langchain returns only LangChain traces."""
    _spawn_langchain_trace()
    _spawn_pydanticai_trace()

    resp = app_client.get("/api/traces?framework=langchain&page_size=200")
    assert resp.status_code == 200
    rows = resp.json()["rows"]
    assert rows, "filter returned empty page"
    for r in rows:
        assert r.get("framework") == "langchain", r

    # Free-text framework filter (UI is open-ended so new frameworks
    # work without a code change). Syntactically valid but unknown
    # framework returns 200 with no rows; syntactically invalid (leads
    # with a digit / contains shell metas) is rejected.
    resp_unknown = app_client.get("/api/traces?framework=langsmith-future")
    assert resp_unknown.status_code == 200, resp_unknown.text
    assert resp_unknown.json()["rows"] == []
    resp_bad = app_client.get("/api/traces?framework=1bad")
    assert resp_bad.status_code == 422
