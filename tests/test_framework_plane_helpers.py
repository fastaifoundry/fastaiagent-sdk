"""What a framework agent (LangGraph, CrewAI, Pydantic AI) needs from the plane (1.98.0).

The plane team ran docs/integrations/frameworks.md against a live plane and found
three gaps in the SDK, each tested here against a real HTTP stand-in for the plane
(``tests/_governance_plane.py``) — no mocks:

* ``plane_guardrails_for_agent("support")`` matched only domain-wide rules: the
  plane scopes a rule by agent **id**, so a rule attached to the agent was
  skipped without a word. A pushed name now resolves to its id.
* A tool approval from a framework tool meant calling the internal, async
  ``governance.decide``. ``fa.check_tool_call`` / ``fa.acheck_tool_call`` are the
  public path, with the agent gate's fail-closed rules.
* The governance enrollment named only agents that existed at ``connect()``, so
  an agent pushed or auto-registered later showed no SDK version or fail mode on
  the plane. A push that adds an agent now re-enrolls with the full list.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.testing.models import TestModel
from tests._governance_plane import GATED_TOOL, GovPlane, reset_connection, serve

SSN = r"\b\d{3}-\d{2}-\d{4}\b"


@pytest.fixture
def plane(isolated_local_db: Path) -> Iterator[GovPlane]:
    """Connected to a stand-in that gives every pushed agent its own UUID."""
    with serve() as (state, url):
        state.accept_agent_push = True
        state.push_ids_by_name = True
        fa.connect(api_key="fa_k_fw_test", target=url)
        try:
            yield state
        finally:
            reset_connection()


def _push(name: str) -> str:
    result = fa.Agent(name=name, system_prompt="You help.", llm=TestModel("ok")).push()
    assert result.agent_id
    return str(result.agent_id)


def _wait_for(predicate: Any, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert predicate()


def _scoped_rule(agent_id: str) -> dict[str, Any]:
    return {
        "name": "block-ssn",
        "guardrail_type": "input",
        "validation_mode": "blocking",
        "implementation_type": "regex",
        "config": {"pattern": SSN},
        "on_error": "block",
        "agent_ids": [agent_id],
    }


# ── B2: plane_guardrails_for_agent takes a pushed name ───────────────────────


def test_a_pushed_name_gets_the_rule_attached_to_its_agent(plane: GovPlane) -> None:
    agent_id = _push("support")
    plane.guardrail_rules = [_scoped_rule(agent_id)]
    fa.refresh_policy()

    by_name = fa.plane_guardrails_for_agent("support")
    assert [g.name for g in by_name] == ["block-ssn"]
    assert [g.name for g in fa.plane_guardrails_for_agent(agent_id)] == ["block-ssn"]
    assert by_name[0].execute("my ssn is 123-45-6789").action_taken == "blocked"


def test_an_unpushed_name_says_why_its_rules_are_skipped(
    plane: GovPlane, caplog: pytest.LogCaptureFixture
) -> None:
    plane.guardrail_rules = [_scoped_rule(_push("support"))]
    fa.refresh_policy()

    with caplog.at_level(logging.WARNING, logger="fastaiagent.guardrail.from_policy"):
        assert fa.plane_guardrails_for_agent("billing") == []
    assert "no agent of that name was pushed" in caplog.text


# ── B4: check_tool_call ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("allow", fa.ToolDecision("allow")),
        ("deny", fa.ToolDecision("deny", "transfers are blocked for this agent")),
        (
            "require_approval",
            fa.ToolDecision(
                "require_approval", "Matched approval policy 'transfer-approval'", "apr-1"
            ),
        ),
        ("error", fa.ToolDecision("deny", "Refused: governance check unavailable")),
    ],
)
def test_a_gated_tool_gets_the_planes_verdict(
    plane: GovPlane, verdict: str, expected: fa.ToolDecision
) -> None:
    agent_id = _push("banker")
    plane.gated_decision = verdict

    assert fa.check_tool_call(GATED_TOOL, {"amount": 250}, agent="banker") == expected
    assert plane.decide_calls[-1] == {
        "tool_name": GATED_TOOL,
        "tool_input": {"amount": 250},
        "agent_id": agent_id,  # the plane UUID, never the name
        "context": None,
    }


def test_an_ungated_tool_is_allowed_without_asking(plane: GovPlane) -> None:
    _push("banker")
    assert fa.check_tool_call("lookup_balance", {}, agent="banker") == fa.ToolDecision("allow")
    assert plane.decide_calls == []


def test_a_gated_tool_for_an_unpushed_name_is_refused_without_asking(plane: GovPlane) -> None:
    decision = fa.check_tool_call(GATED_TOOL, {"amount": 1}, agent="banker")
    assert decision.decision == "deny"
    assert "has not been pushed" in (decision.reason or "")
    assert plane.decide_calls == []


def test_a_plane_agent_id_is_used_as_given(plane: GovPlane) -> None:
    agent_id = _push("banker")
    from fastaiagent._platform.push import reset_registration_state

    reset_registration_state()  # a later process: the id came from config, not a push
    plane.gated_decision = "allow"
    assert fa.check_tool_call(GATED_TOOL, {}, agent=agent_id).decision == "allow"
    assert plane.decide_calls[-1]["agent_id"] == agent_id


def test_not_connected_allows_and_sends_nothing(isolated_local_db: Path) -> None:
    assert fa.check_tool_call(GATED_TOOL, {}, agent="banker") == fa.ToolDecision("allow")


def test_fail_closed_with_no_policy_refuses(plane: GovPlane) -> None:
    from fastaiagent.client import _connection

    _push("banker")
    _connection.policy_cache = None
    _connection.governance_fail_mode = "closed"
    decision = fa.check_tool_call(GATED_TOOL, {}, agent="banker")
    assert decision.decision == "deny"
    assert "fail-closed" in (decision.reason or "")


def test_the_sync_helper_works_inside_a_running_loop(plane: GovPlane) -> None:
    """A sync framework tool called from an async framework: the loop is busy."""
    _push("banker")
    plane.gated_decision = "deny"

    async def framework_calls_a_sync_tool() -> fa.ToolDecision:
        return fa.check_tool_call(GATED_TOOL, {}, agent="banker")

    assert asyncio.run(framework_calls_a_sync_tool()).decision == "deny"
    assert asyncio.run(fa.acheck_tool_call(GATED_TOOL, {}, agent="banker")).decision == "deny"


# ── B5: enrollment names agents pushed after connect() ───────────────────────


def _governed(enroll: dict[str, Any]) -> set[str]:
    return set(enroll.get("governed_agent_ids") or [])


def test_each_new_push_re_enrolls_with_the_full_list(plane: GovPlane) -> None:
    _wait_for(lambda: len(plane.enrolls) >= 1)  # connect()'s own, on a daemon thread
    first = _push("support")
    second = _push("billing")

    assert _governed(plane.enrolls[-1]) >= {first, second}
    # The plane replaces the stored list, so no enrollment drops an earlier agent.
    assert first in _governed(plane.enrolls[-2])
    sent = len(plane.enrolls)
    _push("support")  # nothing new to name
    assert len(plane.enrolls) == sent


def test_an_agent_auto_registered_on_its_first_run_is_enrolled(isolated_local_db: Path) -> None:
    """The native path: connect() first, then build and run an agent."""
    with serve() as (state, url):
        state.accept_agent_push = True
        state.push_ids_by_name = True
        state.extra_scopes = ["agent:write"]
        fa.connect(api_key="fa_k_fw_test", target=url)
        try:
            fa.Agent(name="triage", system_prompt="You help.", llm=TestModel("ok")).run("hi")
            _wait_for(lambda: "triage" in state.pushed_ids)
            agent_id = state.pushed_ids["triage"]
            _wait_for(lambda: any(agent_id in _governed(e) for e in state.enrolls))
        finally:
            reset_connection()
