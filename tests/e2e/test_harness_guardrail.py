"""Phase 6 — ``with_guardrails()`` wrapper tests (real LLM, gated by env).

Spec test IDs covered: #21 (LC input block), #22 (LC output block + event
row), #23 (event surfaces in /api/guardrails), #24 (CrewAI block),
#25 (PydanticAI block), #26 (LC streaming with input-only guardrails).

Decision A (block-only) means tests assert "block raised + event logged",
not "PII redacted from output".
"""

from __future__ import annotations

import os
import time

import pytest

# ``fastapi`` ships in the SDK's ``[ui]`` extra. CI's base test matrix
# doesn't install it, so we ``importorskip`` rather than fail collection.
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

# Guardrail events are recorded whenever tracing is on (1.89.0); this module
# no longer needs FASTAIAGENT_UI_ENABLED, and deliberately does not set it.

HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))

if not HAS_OPENAI:
    pytest.skip(
        "OPENAI_API_KEY not set — Phase 6 needs at least the OpenAI path",
        allow_module_level=True,
    )

needs_openai = pytest.mark.skipif(not HAS_OPENAI, reason="OPENAI_API_KEY not set")

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_client() -> TestClient:
    from fastaiagent._internal.config import get_config
    from fastaiagent.ui.server import build_app

    return TestClient(
        build_app(db_path=str(get_config().resolved_trace_db_path), no_auth=True)
    )


def _block_word_guardrail(*, side: str):
    """Guardrail that fails when a sentinel word appears in the text.

    Cheap, deterministic, no LLM round-trip, no external services. Used
    to drive ``with_guardrails`` reliably in tests without hoping the
    LLM emits PII or toxic content.
    """
    from fastaiagent.guardrail.guardrail import (
        Guardrail,
        GuardrailPosition,
        GuardrailResult,
        GuardrailType,
    )

    # The output sentinel is a word the model cannot avoid: gpt-5-mini refused to
    # repeat "FORBIDDEN_TOKEN" and did not always copy an invented token exactly,
    # so the output rule sometimes had nothing to block.
    sentinel = "BLOCK_ME_PLZ" if side == "input" else "Paris"

    def _check(text: str) -> GuardrailResult:
        if sentinel in text:
            return GuardrailResult(
                passed=False,
                message=f"sentinel '{sentinel}' detected on {side}",
                metadata={"sentinel": sentinel},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name=f"block_{side}_sentinel",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.input
        if side == "input"
        else GuardrailPosition.output,
        blocking=True,
        description=f"Blocks {side} containing the test sentinel.",
        fn=_check,
    )


# ---------------------------------------------------------------------------
# LangChain / LangGraph
# ---------------------------------------------------------------------------


@needs_openai
def test_21_langchain_input_block() -> None:
    """Spec #21: a blocking input guardrail raises GuardrailBlocked."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import GuardrailBlocked

    lc.enable()
    graph = create_react_agent(
        ChatOpenAI(model="gpt-5-mini", reasoning_effort="minimal"), tools=[]
    )

    guarded = lc.with_guardrails(
        graph,
        name="harness-test-lc",
        input_guardrails=[_block_word_guardrail(side="input")],
    )

    with pytest.raises(GuardrailBlocked):
        guarded.invoke(
            {"messages": [HumanMessage(content="Please say hi BLOCK_ME_PLZ now.")]}
        )


@needs_openai
def test_22_langchain_output_block_logs_event(app_client: TestClient) -> None:
    """Spec #22: an output guardrail that blocks logs a guardrail_events
    row (block-only semantics — no redaction, decision A)."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc
    from fastaiagent.integrations._registry import GuardrailBlocked

    lc.enable()
    # The LLM is asked a question whose answer is the sentinel ("Paris"),
    # the most reliable way to make the OUTPUT guardrail trip on gpt-5-mini.
    graph = create_react_agent(
        ChatOpenAI(model="gpt-5-mini", reasoning_effort="minimal"), tools=[]
    )

    guarded = lc.with_guardrails(
        graph,
        name="harness-test-lc-out",
        output_guardrails=[_block_word_guardrail(side="output")],
    )

    from datetime import datetime, timezone

    # Only this run's rows: the e2e gates can share a local.db across runs.
    started = datetime.now(timezone.utc).isoformat()
    with pytest.raises(GuardrailBlocked):
        guarded.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="What is the capital of France? Answer in one word."
                    )
                ]
            }
        )

    # Event must be logged. Use the API for a stable read path.
    resp = app_client.get("/api/guardrail-events?page_size=50")
    assert resp.status_code == 200, resp.text
    events = resp.json().get("rows") or []
    mine = [
        e
        for e in events
        if e.get("guardrail_name") == "block_output_sentinel"
        and e.get("agent_name") == "harness-test-lc-out"
        and str(e.get("timestamp") or "") >= started
    ]
    # Exactly one row for one check (1.89.0): the proxy used to write two.
    assert len(mine) == 1, mine
    assert mine[0]["effect"] == "blocked" and mine[0]["metadata"]["framework"] == "langchain"


@needs_openai
def test_23_event_in_ui_table(app_client: TestClient) -> None:
    """Spec #23: the /api/guardrails endpoint is the same surface the
    UI's Guardrail Events page reads, so a logged event from a wrapped
    agent shows up there."""
    # Reuse #22's event. Spec only requires "row visible".
    resp = app_client.get("/api/guardrail-events?page_size=50")
    assert resp.status_code == 200
    payload = resp.json()
    events = payload.get("rows") or []
    assert events, "no guardrail events at all in /api/guardrail-events"


# ---------------------------------------------------------------------------
# CrewAI
# ---------------------------------------------------------------------------


@needs_openai
def test_24_crewai_input_block() -> None:
    """Spec #24: input guardrail blocks a CrewAI kickoff."""
    from crewai import Agent, Crew, Process, Task
    from crewai.llm import LLM

    from fastaiagent.integrations import crewai as ca
    from fastaiagent.integrations._registry import GuardrailBlocked

    ca.enable()

    llm = LLM(model="openai/gpt-5-mini")
    a = Agent(
        role="R",
        goal="answer",
        backstory="terse",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    t = Task(description="Say BLOCK_ME_PLZ.", expected_output="ok", agent=a)
    crew = Crew(agents=[a], tasks=[t], process=Process.sequential, verbose=False)

    guarded = ca.with_guardrails(
        crew,
        name="harness-test-ca",
        input_guardrails=[_block_word_guardrail(side="input")],
    )

    with pytest.raises(GuardrailBlocked):
        guarded.kickoff(inputs={"input": "BLOCK_ME_PLZ now"})


# ---------------------------------------------------------------------------
# PydanticAI
# ---------------------------------------------------------------------------


@needs_openai
def test_25_pydanticai_input_block() -> None:
    """Spec #25: input guardrail blocks a PydanticAI run_sync."""
    from pydantic_ai import Agent

    from fastaiagent.integrations import pydanticai as pa
    from fastaiagent.integrations._registry import GuardrailBlocked

    pa.enable()
    agent = Agent("openai:gpt-5-mini", system_prompt="terse")

    guarded = pa.with_guardrails(
        agent,
        name="harness-test-pa",
        input_guardrails=[_block_word_guardrail(side="input")],
    )

    with pytest.raises(GuardrailBlocked):
        guarded.run_sync("Please BLOCK_ME_PLZ now.")


# ---------------------------------------------------------------------------
# Streaming with input-only guardrails
# ---------------------------------------------------------------------------


@needs_openai
def test_26_langchain_streaming_input_only() -> None:
    """Spec #26: streaming with input-only guardrails completes normally
    (the input check fires before the stream opens; output guardrails
    would buffer the full stream and add latency, so they're omitted)."""
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    from fastaiagent.integrations import langchain as lc

    lc.enable()
    graph = create_react_agent(
        ChatOpenAI(model="gpt-5-mini", reasoning_effort="minimal"), tools=[]
    )

    guarded = lc.with_guardrails(
        graph,
        name="harness-test-lc-stream",
        input_guardrails=[_block_word_guardrail(side="input")],
    )

    chunks = list(
        guarded.stream(
            {"messages": [HumanMessage(content="Reply: ok")]},
            stream_mode="updates",
        )
    )
    assert chunks, "streaming returned no chunks"


# Soft cleanup so subsequent test modules see a fresh module-level config
def teardown_module(module: object) -> None:
    # Give the SQLite WAL a moment to flush before later modules reopen.
    time.sleep(0.1)
