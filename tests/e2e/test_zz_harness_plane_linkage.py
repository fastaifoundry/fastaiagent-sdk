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

Real LLM calls throughout — no mocking. The failure test does not mock either:
it points the provider at a closed local port so a genuine connection error
propagates through the real client stack.

Reads the ambient trace store via ``TraceStore.default()`` and polls for the
span it produced, matching the other harness modules. Do **not** repoint
``FASTAIAGENT_LOCAL_DB`` here: the OTel provider and its storage processor are
process singletons wired on first use, so a later override neither takes effect
nor stays contained — it leaks into every module that runs afterwards.

The ``zz_`` prefix is deliberate. Collection is alphabetical and the store is
shared across modules, so this module must run *after* the existing harness
suites. ``test_harness_pydanticai.test_11_token_capture`` selects **any** trace
tagged ``fastaiagent.framework == "pydanticai"`` rather than a sentinel of its
own, so extra pydanticai traffic recorded before it can be picked up instead of
its own run. Sorting last keeps this module additive: existing order-sensitive
tests see exactly the trace population they saw before it existed.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

pytestmark = pytest.mark.e2e

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))
needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")

CLOSED_PORT_BASE_URL = "http://127.0.0.1:9/v1"


# ---------------------------------------------------------------------------
# Helpers — mirror tests/e2e/test_harness_pydanticai.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _langchain_v0_globals() -> None:
    """Reconcile a third-party version clash, not a mock.

    ``langchain>=1`` removed the module-level ``verbose`` / ``debug`` /
    ``llm_cache`` globals that ``langchain-core~=0.3`` still probes in
    ``langchain_core.globals``; without them, constructing or invoking any
    LangChain chat model raises ``AttributeError``. No fastaiagent behaviour is
    stubbed — this only restores what langchain-core expects.
    """
    try:
        import langchain
    except ImportError:  # pragma: no cover - langchain not installed
        return
    for attr in ("verbose", "debug", "llm_cache"):
        if not hasattr(langchain, attr):
            setattr(langchain, attr, None if attr == "llm_cache" else False)


def _trace_store():
    from fastaiagent.trace.storage import TraceStore

    return TraceStore.default()


def _wait_for_trace(predicate, timeout: float = 20.0):
    """Return the first trace containing a span matching ``predicate``."""
    store = _trace_store()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for summary in store.list_traces():
            try:
                trace = store.get_trace(summary.trace_id)
            except Exception:
                continue
            if any(predicate(span) for span in trace.spans):
                return trace
        time.sleep(0.25)
    return None


def _root(trace):
    for span in trace.spans:
        if span.parent_span_id is None:
            return span
    return trace.spans[0]


def _exception_events(span) -> list:
    return [e for e in (span.events or []) if (e or {}).get("name") == "exception"]


def _plane_derived_name(trace) -> str | None:
    """Reimplements the plane's ``_root_agent_name`` resolution order."""
    root = _root(trace)
    attrs = root.attributes or {}
    for key in ("agent.name", "chain.name", "swarm.name", "workflow.name"):
        if attrs.get(key):
            return str(attrs[key])
    nm = root.name or ""
    return nm.split(".", 1)[1] if "." in nm else (nm or None)


def _by_agent_name(name: str):
    return lambda span: (span.attributes or {}).get("agent.name") == name


def _by_crew_role(role: str):
    def _pred(span) -> bool:
        attrs = span.attributes or {}
        return role in str(attrs.get("crewai.agent.role", "")) or role in (span.name or "")

    return _pred


def _marker_in_payload(marker: str):
    """Match a span whose recorded input/output carries our unique marker —
    the only way to find an *unnamed* run's trace."""

    def _pred(span) -> bool:
        attrs = span.attributes or {}
        for key in ("input", "output", "gen_ai.request.messages"):
            if marker in str(attrs.get(key, "")):
                return True
        return False

    return _pred


# ---------------------------------------------------------------------------
# A1 — agent.name reaches the root span
# ---------------------------------------------------------------------------


@needs_openai
def test_langgraph_root_span_carries_agent_name() -> None:
    pytest.importorskip("langgraph")
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    name = f"e2e-linkage-{uuid.uuid4().hex[:8]}"
    graph = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=False), tools=[]
    )
    lc.with_guardrails(graph, name=name).invoke(
        {"messages": [("user", "Say OK and nothing else.")]}
    )

    trace = _wait_for_trace(_by_agent_name(name))
    assert trace is not None, f"no trace carrying agent.name={name!r}"
    root = _root(trace)
    assert (root.attributes or {}).get("agent.name") == name
    # The span NAME stays generic — the fix is the attribute, which the plane
    # consults first. Asserting both keeps the regression honest.
    assert root.name.startswith(("langchain.", "langgraph."))
    assert _plane_derived_name(trace) == name


@needs_openai
def test_pydanticai_root_span_carries_agent_name() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa

    pa.enable()
    name = f"e2e-linkage-{uuid.uuid4().hex[:8]}"
    pa.with_guardrails(
        Agent("openai:gpt-4o-mini", system_prompt="Be terse."), name=name
    ).run_sync("Say OK and nothing else.")

    trace = _wait_for_trace(_by_agent_name(name))
    assert trace is not None, f"no trace carrying agent.name={name!r}"
    assert _plane_derived_name(trace) == name


@needs_openai
def test_unnamed_run_emits_no_agent_name() -> None:
    """Unnamed runs must not invent an identity — documented behaviour."""
    pytest.importorskip("langgraph")
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    marker = f"unnamed-{uuid.uuid4().hex[:8]}"
    graph = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=False), tools=[]
    )
    graph.invoke({"messages": [("user", f"Reply with exactly: {marker}")]})

    trace = _wait_for_trace(_marker_in_payload(marker))
    assert trace is not None, "could not locate the unnamed run's trace"
    assert "agent.name" not in (_root(trace).attributes or {})


@needs_openai
def test_agent_name_context_manager() -> None:
    """The public escape hatch for code that doesn't use with_guardrails."""
    pytest.importorskip("langgraph")
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import agent_name
    from fastaiagent.integrations import langchain as lc

    lc.enable()
    name = f"e2e-ctx-{uuid.uuid4().hex[:8]}"
    graph = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=False), tools=[]
    )
    with agent_name(name):
        graph.invoke({"messages": [("user", "Say OK and nothing else.")]})

    trace = _wait_for_trace(_by_agent_name(name))
    assert trace is not None, f"no trace carrying agent.name={name!r}"
    assert _plane_derived_name(trace) == name


# ---------------------------------------------------------------------------
# A2 — register_agent() reaches the plane
# ---------------------------------------------------------------------------


def test_push_external_agent_is_noop_when_disconnected() -> None:
    """Must never raise or push when there is no plane connection."""
    from fastaiagent.client import _connection
    from fastaiagent.integrations._registry import (
        push_external_agent,
        reset_plane_pushed_for_tests,
    )

    if _connection.is_connected:
        pytest.skip("a plane connection is active; this asserts the offline path")
    reset_plane_pushed_for_tests()
    assert push_external_agent(f"offline-{uuid.uuid4().hex[:6]}", "langchain") is None


# ---------------------------------------------------------------------------
# B + C — error paths (real connection failure, not a mock)
# ---------------------------------------------------------------------------


@needs_openai
def test_crewai_failure_records_exception_once_and_marks_llm_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B: exactly one exception event per errored span.
    C: the LLM span closes as ERROR instead of dangling UNSET.
    """
    pytest.importorskip("crewai")
    from crewai import LLM, Agent, Crew, Process, Task

    from fastaiagent.integrations import crewai as ca

    # Real client, real socket, closed port -> genuine ConnectionError.
    monkeypatch.setenv("OPENAI_BASE_URL", CLOSED_PORT_BASE_URL)
    ca.enable()

    role = f"E2ERole-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        role=role,
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

    trace = _wait_for_trace(_by_crew_role(role))
    assert trace is not None, "no trace recorded for the failed crew run"

    errored = [s for s in trace.spans if _exception_events(s)]
    assert errored, "expected at least one span carrying an exception event"
    for span in errored:
        assert len(_exception_events(span)) == 1, (
            f"{span.name!r} recorded {len(_exception_events(span))} exception events; "
            "the explicit record_exception + OTel auto-record duplication is back"
        )

    llm_spans = [s for s in trace.spans if (s.name or "").startswith("llm.")]
    assert llm_spans, "expected an llm.* span even though the call failed"
    assert all(s.status == "ERROR" for s in llm_spans), (
        "failed LLM span must close as ERROR, not dangle UNSET — otherwise it is "
        "indistinguishable from a call still in flight"
    )


@needs_openai
def test_crewai_llm_span_carries_usage_on_success() -> None:
    """C: correlation works even when LLMCallStartedEvent has no call_id,
    so tokens/cost actually land on the span."""
    pytest.importorskip("crewai")
    from crewai import LLM, Agent, Crew, Process, Task

    from fastaiagent.integrations import crewai as ca

    ca.enable()
    role = f"E2ESupport-{uuid.uuid4().hex[:8]}"
    agent = Agent(
        role=role,
        goal="Answer briefly.",
        backstory="Terse assistant.",
        llm=LLM(model="openai/gpt-4o-mini", temperature=0),
        allow_delegation=False,
        verbose=False,
    )
    task = Task(
        description="Say OK and nothing else.", expected_output="OK", agent=agent
    )
    Crew(
        agents=[agent], tasks=[task], process=Process.sequential, verbose=False
    ).kickoff()

    trace = _wait_for_trace(_by_crew_role(role))
    assert trace is not None, "no trace recorded for the crew run"

    llm_spans = [s for s in trace.spans if (s.name or "").startswith("llm.")]
    assert llm_spans, "expected an llm.* span"
    assert any(
        "gen_ai.usage.input_tokens" in (s.attributes or {}) for s in llm_spans
    ), (
        "no CrewAI LLM span carried gen_ai.usage.* — correlation regressed, so the "
        "plane receives zero tokens/cost for CrewAI runs"
    )
