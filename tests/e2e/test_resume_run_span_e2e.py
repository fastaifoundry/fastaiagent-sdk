"""Live e2e: a resumed run is one linked agent trace on real providers (1.89.0).

An approval-gated agent (its tool calls ``interrupt()``) pauses, is resumed
with a decision, and finishes. Before 1.89.0 the resume opened no agent span:
the model call it made after the tool was stored as an orphan
``llm.<provider>.<model>`` root and ``Agent.resume`` returned ``trace_id=None``.

Checked against ``local.db``, per target (OpenAI ``gpt-5-mini`` and
``gpt-5.1``, Anthropic ``claude-haiku-4-5``, each skipped when its key is
missing) × approved/rejected:

* exactly one ``agent.<name>`` root in the resume's trace, a new trace (D7);
* linked by ``agent.execution_id`` / ``agent.resumed_execution_id``;
* carrying ``agent.resume.{kind,approved,resolver,reason}``;
* its ``llm.*`` call nested under it — and no ``llm.*`` root anywhere.

No mocks: real model, real tool loop, real ``SQLiteCheckpointer``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

TARGETS = [
    ("openai", "gpt-5-mini"),
    ("openai", "gpt-5.1"),
    ("anthropic", "claude-haiku-4-5"),
]
_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# GPT-5 models reject ``temperature``: they take a reasoning level instead.
_LLM_KWARGS: dict[str, dict[str, Any]] = {
    "gpt-5-mini": {"reasoning": "minimal"},
    "gpt-5.1": {"reasoning": "none"},
    "claude-haiku-4-5": {"temperature": 0},
}


@pytest.fixture(params=TARGETS, ids=lambda t: f"{t[0]}:{t[1]}")
def target(request) -> tuple[str, str]:
    provider, model = request.param
    if not os.environ.get(_KEY_ENV[provider]):
        pytest.skip(f"{_KEY_ENV[provider]} not set")
    return provider, model


@pytest.fixture
def local_db(monkeypatch, tmp_path) -> Iterator[Path]:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.delenv("FASTAIAGENT_API_KEY", raising=False)
    monkeypatch.delenv("FASTAIAGENT_TARGET", raising=False)
    reset_config()
    otel.reset()
    try:
        yield db
    finally:
        otel.reset()
        reset_config()


def _spans(db: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT trace_id, span_id, parent_span_id, name, attributes FROM spans"
        ).fetchall()
    finally:
        conn.close()
    return [{**dict(r), "attrs": json.loads(r["attributes"] or "{}")} for r in rows]


@pytest.mark.parametrize("approved", [True, False], ids=["approved", "rejected"])
def test_a_live_resume_is_one_linked_agent_trace(
    target: tuple[str, str], approved: bool, local_db: Path
) -> None:
    from fastaiagent import Agent, FunctionTool, Resume, SQLiteCheckpointer, interrupt
    from fastaiagent.llm import LLMClient

    ran: list[str] = []

    def issue_refund(order_id: str) -> str:
        """Refund an order. A manager must approve every refund."""
        decision = interrupt(reason="manager_approval", context={"order_id": order_id})
        if not decision.approved:
            return f"Refund for {order_id} was REJECTED by the manager."
        ran.append(order_id)
        return f"Refund for {order_id} issued."

    provider, model = target
    agent = Agent(
        name="refund-desk",
        system_prompt=(
            "You process refunds. Always call issue_refund with the order id, then "
            "report the tool's result to the user in one sentence."
        ),
        llm=LLMClient(provider=provider, model=model, **_LLM_KWARGS[model]),
        tools=[FunctionTool(name="issue_refund", fn=issue_refund)],
        checkpointer=SQLiteCheckpointer(str(local_db)),
    )
    exec_id = f"live-{provider}-{'yes' if approved else 'no'}"

    paused = agent.run("Please refund order A17.", execution_id=exec_id)
    assert paused.status == "paused", paused
    assert paused.pending_interrupt["context"]["order_id"] == "A17"

    done = agent.resume(
        exec_id,
        resume_value=Resume(
            approved=approved, metadata={"resolver": "maria", "reason": "verified receipt"}
        ),
    )

    assert done.status == "completed", done
    assert done.output
    assert ran == (["A17"] if approved else [])
    assert done.trace_id and done.trace_id != paused.trace_id

    spans = _spans(local_db)
    roots = [s for s in spans if not s["parent_span_id"]]
    orphans = [s["name"] for s in roots if s["name"].startswith("llm.")]
    assert orphans == [], f"orphan llm roots: {orphans}"

    resume_spans = [s for s in spans if s["trace_id"] == done.trace_id]
    resume_roots = [s for s in resume_spans if not s["parent_span_id"]]
    assert [s["name"] for s in resume_roots] == ["agent.refund-desk"]
    root = resume_roots[0]["attrs"]
    assert root["agent.execution_id"] == exec_id
    assert root["agent.resumed_execution_id"] == exec_id
    assert root["agent.resume.kind"] == "interrupt"
    assert root["agent.resume.approved"] is approved
    assert root["agent.resume.resolver"] == "maria"
    assert root["agent.resume.reason"] == "verified receipt"
    assert root["agent.tokens_used"] == done.tokens_used > 0
    assert any(s["name"].startswith("llm.") for s in resume_spans), "resume made no model call"

    # The run and its resume: two agent roots, one execution.
    agent_roots = [s for s in roots if s["name"] == "agent.refund-desk"]
    assert sorted(s["trace_id"] for s in agent_roots) == sorted([paused.trace_id, done.trace_id])
    assert {s["attrs"]["agent.execution_id"] for s in agent_roots} == {exec_id}
