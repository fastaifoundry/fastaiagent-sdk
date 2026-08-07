"""Foreign-framework → control-plane linkage (v1.46.0).

Covers the four defects found by the framework/plane assessment:

* **A1** — foreign root spans carry ``agent.name`` so the plane can resolve a
  trace to an agent instead of prefix-stripping a generic span name
  (``langchain.chain`` → ``"chain"``, colliding across every LangChain user).
* **A2** — ``register_agent()`` also registers with the control plane, so the
  name stamped by A1 has something to match.
* **B**  — an errored span records its exception exactly once (the integrations
  used to call ``record_exception`` explicitly *and* let OTel's
  ``use_span(record_exception=True)`` record it again on re-raise).
* **C**  — CrewAI LLM spans correlate on builds whose ``LLMCallStartedEvent``
  carries no ``call_id``, so tokens/cost land; and a failed provider call closes
  the LLM span as ERROR rather than leaving it UNSET with no usage.

Real LLM calls throughout — no mocking. The failure tests do not mock either:
they point the provider at a closed local port so a genuine connection error
propagates through the real client stack.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

CLOSED_PORT_BASE_URL = "http://127.0.0.1:9/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _attrs(row: sqlite3.Row) -> dict:
    return json.loads(row["attributes"] or "{}")


def _root(rows: list[sqlite3.Row]) -> sqlite3.Row:
    for r in rows:
        if r["parent_span_id"] is None:
            return r
    raise AssertionError("no root span recorded")


def _exception_events(row: sqlite3.Row) -> list[dict]:
    return [e for e in json.loads(row["events"] or "[]") if e.get("name") == "exception"]


def _plane_derived_name(rows: list[sqlite3.Row]) -> str | None:
    """Reimplements the plane's ``_root_agent_name`` resolution order."""
    root = _root(rows)
    attrs = _attrs(root)
    for key in ("agent.name", "chain.name", "swarm.name", "workflow.name"):
        if attrs.get(key):
            return str(attrs[key])
    nm = root["name"] or ""
    return nm.split(".", 1)[1] if "." in nm else (nm or None)


@pytest.fixture(scope="module", autouse=True)
def _langchain_v0_globals() -> None:
    """Reconcile a third-party version mismatch, not a mock.

    ``langchain>=1`` removed the module-level ``verbose`` / ``debug`` /
    ``llm_cache`` globals that ``langchain-core~=0.3`` still probes in
    ``langchain_core.globals``; without them, constructing or invoking any
    LangChain chat model raises ``AttributeError``. No fastaiagent behaviour is
    stubbed — this only restores the attributes langchain-core expects.
    """
    try:
        import langchain
    except ImportError:  # pragma: no cover - langchain not installed
        return
    for attr in ("verbose", "debug", "llm_cache"):
        if not hasattr(langchain, attr):
            setattr(langchain, attr, None if attr == "llm_cache" else False)


@pytest.fixture(scope="module")
def trace_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """One throwaway trace store for the whole module.

    Module-scoped on purpose: the OTel ``TracerProvider`` and its storage
    processor are process singletons wired on first use, so re-pointing
    ``FASTAIAGENT_LOCAL_DB`` per test would silently keep writing to the first
    test's database. Tests therefore share one store and read only the spans
    their own call produced (see :func:`spans_since`).
    """
    db = tmp_path_factory.mktemp("linkage") / "linkage.db"
    os.environ["FASTAIAGENT_LOCAL_DB"] = str(db)
    os.environ["FASTAIAGENT_UI_ENABLED"] = "1"
    from fastaiagent._internal.config import reset_config

    reset_config()
    return str(db)


def _max_rowid(db_path: str) -> int:
    if not Path(db_path).exists():
        return 0
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM spans").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:  # table not created yet
        return 0
    finally:
        conn.close()


@pytest.fixture
def spans_since(trace_db: str):
    """Return only the spans written after this fixture was set up."""
    cursor = _max_rowid(trace_db)

    def _read() -> list[sqlite3.Row]:
        conn = sqlite3.connect(trace_db)
        conn.row_factory = sqlite3.Row
        try:
            return list(
                conn.execute(
                    "SELECT name, parent_span_id, status, attributes, events "
                    "FROM spans WHERE rowid > ? ORDER BY start_time",
                    (cursor,),
                )
            )
        finally:
            conn.close()

    return _read


def _require_openai() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


# ---------------------------------------------------------------------------
# A1 — agent.name reaches the root span
# ---------------------------------------------------------------------------


def test_langgraph_root_span_carries_agent_name(spans_since) -> None:
    _require_openai()
    pytest.importorskip("langgraph")
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    name = f"e2e-linkage-{uuid.uuid4().hex[:8]}"
    graph = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=False), tools=[]
    )
    guarded = lc.with_guardrails(graph, name=name)
    guarded.invoke({"messages": [("user", "Say OK and nothing else.")]})

    rows = spans_since()
    assert _attrs(_root(rows)).get("agent.name") == name
    # The span NAME stays generic — the fix is the attribute, which the plane
    # consults first. Guarding this keeps the regression honest.
    assert _root(rows)["name"].startswith(("langchain.", "langgraph."))
    assert _plane_derived_name(rows) == name


def test_pydanticai_root_span_carries_agent_name(spans_since) -> None:
    _require_openai()
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    name = f"e2e-linkage-{uuid.uuid4().hex[:8]}"
    guarded = pa.with_guardrails(
        Agent("openai:gpt-4o-mini", system_prompt="Be terse."), name=name
    )
    guarded.run_sync("Say OK and nothing else.")

    assert _plane_derived_name(spans_since()) == name


def test_unnamed_run_emits_no_agent_name(spans_since) -> None:
    """Unnamed runs must not invent an identity — documented behaviour."""
    _require_openai()
    pytest.importorskip("langgraph")
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=False), tools=[]
    )
    graph.invoke({"messages": [("user", "Say OK and nothing else.")]})

    assert "agent.name" not in _attrs(_root(spans_since()))


def test_agent_name_context_manager(spans_since) -> None:
    """The public escape hatch for code that doesn't use with_guardrails."""
    _require_openai()
    pytest.importorskip("langgraph")
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._identity import agent_name

    lc.enable()
    name = f"e2e-ctx-{uuid.uuid4().hex[:8]}"
    graph = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=False), tools=[]
    )
    with agent_name(name):
        graph.invoke({"messages": [("user", "Say OK and nothing else.")]})

    assert _plane_derived_name(spans_since()) == name


# ---------------------------------------------------------------------------
# A2 — register_agent() reaches the plane
# ---------------------------------------------------------------------------


def test_push_external_agent_is_noop_when_disconnected() -> None:
    """Must never raise or push when there is no plane connection."""
    from fastaiagent.integrations._registry import (
        push_external_agent,
        reset_plane_pushed_for_tests,
    )

    reset_plane_pushed_for_tests()
    assert push_external_agent(f"offline-{uuid.uuid4().hex[:6]}", "langchain") is None


# ---------------------------------------------------------------------------
# B + C — error paths (real connection failure, not a mock)
# ---------------------------------------------------------------------------


def test_crewai_failure_records_exception_once_and_marks_llm_span(
    spans_since, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B: exactly one exception event per errored span.
    C: the LLM span closes as ERROR instead of dangling UNSET.
    """
    _require_openai()
    pytest.importorskip("crewai")
    from crewai import LLM, Agent, Crew, Process, Task

    from fastaiagent.integrations import crewai as ca

    # Real client, real socket, closed port -> genuine ConnectionError.
    monkeypatch.setenv("OPENAI_BASE_URL", CLOSED_PORT_BASE_URL)
    ca.enable()

    agent = Agent(
        role="R",
        goal="g",
        backstory="b",
        llm=LLM(model="openai/gpt-4o-mini", temperature=0),
        allow_delegation=False,
        verbose=False,
    )
    task = Task(description="say hi", expected_output="hi", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)

    with pytest.raises(Exception):
        crew.kickoff()

    rows = spans_since()
    errored = [r for r in rows if _exception_events(r)]
    assert errored, "expected at least one span carrying an exception event"
    for row in errored:
        assert len(_exception_events(row)) == 1, (
            f"{row['name']!r} recorded {len(_exception_events(row))} exception events; "
            "the explicit record_exception + OTel auto-record duplication is back"
        )

    llm_spans = [r for r in rows if r["name"].startswith("llm.")]
    assert llm_spans, "expected an llm.* span even though the call failed"
    assert all(r["status"] == "ERROR" for r in llm_spans), (
        "failed LLM span must close as ERROR, not dangle UNSET — otherwise it is "
        "indistinguishable from a call still in flight"
    )


def test_crewai_llm_span_carries_usage_on_success(spans_since) -> None:
    """C: correlation works even when LLMCallStartedEvent has no call_id,
    so tokens/cost actually land on the span."""
    _require_openai()
    pytest.importorskip("crewai")
    from crewai import LLM, Agent, Crew, Process, Task

    from fastaiagent.integrations import crewai as ca

    ca.enable()
    agent = Agent(
        role="Support",
        goal="Answer briefly.",
        backstory="Terse assistant.",
        llm=LLM(model="openai/gpt-4o-mini", temperature=0),
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="Say OK and nothing else.", expected_output="OK", agent=agent
    )
    Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False).kickoff()

    llm_spans = [r for r in spans_since() if r["name"].startswith("llm.")]
    assert llm_spans, "expected an llm.* span"
    assert any("gen_ai.usage.input_tokens" in _attrs(r) for r in llm_spans), (
        "no CrewAI LLM span carried gen_ai.usage.* — correlation regressed, so the "
        "plane receives zero tokens/cost for CrewAI runs"
    )
