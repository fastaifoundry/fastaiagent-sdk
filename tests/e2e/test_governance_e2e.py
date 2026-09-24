"""E2E (Task C) — a connected agent honors a managed approval policy.

No mocks of the model. A **real** ``gpt-4o-mini`` agent runs against a localhost
stand-in for the plane's frozen governance endpoints (``/policy``,
``/policy/decide``, ``/runs/{id}/pending``, ``/hitl/events`` —
``tests/_governance_plane.py``). A high-stakes tool (``transfer_funds``) matches
the cached approval policy → ``/policy/decide`` returns ``require_approval`` →
the SDK posts a pending run and **pauses** (a real checkpoint).

Since 1.74.0 the calling application is the approver (plane decision,
2026-09-23): ``arun()`` returns the pause, and the app resumes with
``Resume(approved=...)``. Pinned here with a real model:

* approve → the tool runs and the model confirms;
* **reject → the tool never runs and the model is told** (audit H1: before
  1.74.0 it ran and the model reported a successful transfer).

The blocking ``wait_for_approval=True`` was removed in 1.76.0, and its tests with it.

The live-plane version of the same reproduction is
``test_connected_approvals_e2e.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests._governance_plane import AGENT_ID, GovPlane, reset_connection, serve
from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e

_API_KEY = "fa_k_gov_e2e"


@pytest.fixture
def gov_plane(isolated_local_db: Path) -> Iterator[tuple[GovPlane, str]]:
    with serve() as (state, url):
        try:
            yield state, url
        finally:
            reset_connection()


def _make_agent(tmp_path: Path) -> tuple[Any, list[dict[str, Any]]]:
    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

    ran: list[dict[str, Any]] = []

    def transfer_funds(amount: int, to: str) -> str:
        ran.append({"amount": amount, "to": to})
        return f"Transferred ${amount} to {to}."

    agent = Agent(
        name="banker",
        agent_id=AGENT_ID,
        system_prompt=(
            "You are a banking assistant. To move money, call transfer_funds(amount, to). "
            "After the tool returns, tell the user in one sentence exactly what it said."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    return agent, ran


def test_governance_pause_approve_resume(gov_plane: tuple[GovPlane, str], tmp_path: Path) -> None:
    require_env()  # OPENAI_API_KEY — real gpt-4o-mini
    state, url = gov_plane

    import fastaiagent
    from fastaiagent.chain.interrupt import Resume
    from fastaiagent.client import _connection

    fastaiagent.connect(api_key=_API_KEY, target=url)
    assert (_connection.policy_cache or {}).get("approval_policies"), "policy should be cached"

    agent, ran = _make_agent(tmp_path)
    res = asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-ok"))

    # The default returns the pause to the app, with what it needs to ask its user.
    assert res.status == "paused", res
    assert res.pending_interrupt["reason"] == "policy_approval_required"
    assert res.pending_interrupt["context"]["tool_input"] == {"amount": 500, "to": "Bob"}
    assert ran == []
    assert state.decide_calls and state.decide_calls[0]["tool_name"] == "transfer_funds"
    assert state.decide_calls[0]["agent_id"] == AGENT_ID  # the platform agent id, not the name
    assert state.pending_posts and state.pending_posts[0]["body"]["kind"] == "approval"
    assert state.pending_polls == 0, "the default must not wait on the plane"

    final = asyncio.run(
        agent.aresume("run-ok", resume_value=Resume(approved=True, metadata={"resolver": "eve"}))
    )
    assert final.status == "completed", final
    assert ran == [{"amount": 500, "to": "Bob"}]
    assert "bob" in final.output.lower()


def test_governance_reject_does_not_run_the_tool(
    gov_plane: tuple[GovPlane, str], tmp_path: Path
) -> None:
    """The handoff's §2 reproduction, with the fix: a "no" never reaches the tool."""
    require_env()
    state, url = gov_plane

    import fastaiagent
    from fastaiagent.chain.interrupt import Resume

    fastaiagent.connect(api_key=_API_KEY, target=url)
    agent, ran = _make_agent(tmp_path)
    res = asyncio.run(agent.arun("Transfer $500 to Bob.", execution_id="run-no"))
    assert res.status == "paused", res

    final = asyncio.run(
        agent.aresume("run-no", resume_value=Resume(approved=False, metadata={"resolver": "carol"}))
    )

    assert final.status == "completed", final
    assert ran == [], f"a rejected approval executed transfer_funds: {ran}"
    reply = final.output.lower()
    assert "transferred $500" not in reply and "successfully" not in reply, final.output
