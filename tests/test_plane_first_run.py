"""A connected agent's first run sees everything the plane meant for it (1.104.0).

Rules attached to an agent and approval policies are matched by the plane's
``agent_id``, which an agent gets when it registers. Auto-registration starts on a
background thread at the start of the first run, so over a slow link that run
used to pick its guardrails, and gate its tools, before the id arrived: a rule
attached to the agent was skipped and a governed tool ran ungated, without a word.
The run now waits (bounded) for its own registration when, and only when, the
policy has something that needs the id.

Also here: a failed policy pull is a WARNING (it was a debug line, so "no rules
apply" looked the same as "no rules authored"), and the public import paths.

All against the real HTTP stand-in for the plane (``tests/_governance_plane.py``),
whose ``push_delay`` makes registration slow. No mocks.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent._platform import push
from fastaiagent.testing.models import TestModel
from fastaiagent.tool import FunctionTool
from tests._governance_plane import AGENT_ID, GATED_TOOL, GovPlane, reset_connection, serve

LEAK = "the code is SECRET-42"


@pytest.fixture
def slow_plane(isolated_local_db: Path) -> Iterator[GovPlane]:
    """Connected, auto-registration on, and each registration takes 1 s."""
    with serve() as (state, url):
        state.accept_agent_push = True  # every push gets AGENT_ID
        state.extra_scopes = ["agent:write"]
        state.push_delay = 1.0
        yield state, url  # type: ignore[misc]
        reset_connection()


def _rule(*, agent_ids: list[str]) -> dict[str, Any]:
    return {
        "name": "no-secret-codes",
        "guardrail_type": "output",
        "validation_mode": "blocking",
        "implementation_type": "regex",
        "config": {"pattern": r"SECRET-\d+", "should_match": False},
        "on_error": "block",
        "agent_ids": agent_ids,
    }


def _connect(state: GovPlane, url: str, rules: list[dict[str, Any]]) -> None:
    state.guardrail_rules = rules
    fa.connect(api_key="fa_k_first_run", target=url)


def test_a_rule_attached_to_the_agent_applies_on_its_first_run(slow_plane: Any) -> None:
    state, url = slow_plane
    _connect(state, url, [_rule(agent_ids=[AGENT_ID])])
    agent = fa.Agent(name="first-run-scoped", llm=TestModel(LEAK))  # built after connect()
    assert agent.agent_id is None

    with pytest.raises(fa.GuardrailBlockedError):
        agent.run("hi")
    assert agent.agent_id == AGENT_ID


def test_the_first_run_of_a_streamed_agent_waits_too(slow_plane: Any) -> None:
    state, url = slow_plane
    _connect(state, url, [_rule(agent_ids=[AGENT_ID])])
    agent = fa.Agent(name="first-run-streamed", llm=TestModel(LEAK))

    async def drain() -> None:
        async for _ in agent.astream("hi"):
            pass

    import asyncio

    with pytest.raises(fa.GuardrailBlockedError):
        asyncio.run(drain())


def test_an_approval_policy_gates_the_tool_on_the_first_run(slow_plane: Any) -> None:
    state, url = slow_plane
    state.gated_decision = "deny"  # refused, so the run completes instead of pausing
    _connect(state, url, [])
    transfer = FunctionTool(
        name=GATED_TOOL, fn=lambda amount: f"sent {amount}", description="Move money."
    )
    agent = fa.Agent(
        name="first-run-governed",
        llm=TestModel(
            response=["", "done"],
            tool_calls=[{"name": GATED_TOOL, "arguments": {"amount": 5}}],
        ),
        tools=[transfer],
    )

    result = agent.run("send 5")
    assert [c["agent_id"] for c in state.decide_calls] == [AGENT_ID]
    assert result.output == "done"


def test_a_domain_wide_policy_does_not_wait(slow_plane: Any) -> None:
    """Only domain-wide rules and no governed tool: nothing needs the id, so the
    first run does not pay for the slow registration."""
    state, url = slow_plane
    state.push_delay = 3.0
    _connect(state, url, [_rule(agent_ids=[])])
    agent = fa.Agent(name="first-run-domain-wide", llm=TestModel(LEAK))

    started = time.monotonic()
    with pytest.raises(fa.GuardrailBlockedError):
        agent.run("hi")
    assert time.monotonic() - started < 2.0


def test_the_wait_is_bounded_and_says_what_it_skipped(
    slow_plane: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    state, url = slow_plane
    state.push_delay = 3.0
    monkeypatch.setattr(push, "REGISTRATION_WAIT_SECONDS", 0.3)
    _connect(state, url, [_rule(agent_ids=[AGENT_ID])])
    agent = fa.Agent(name="first-run-timeout", llm=TestModel(LEAK))

    with caplog.at_level(logging.WARNING, logger="fastaiagent._platform.push"):
        result = agent.run("hi")
    assert result.output == LEAK  # the attached rule could not apply ...
    assert "started its first run before its registration" in caplog.text  # ... and it says so


# ── a failed policy pull is not silent ──────────────────────────────────────


def test_a_failed_policy_pull_at_connect_warns(
    isolated_local_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with serve() as (state, url):
        state.policy_status = 503
        try:
            with caplog.at_level(logging.WARNING, logger="fastaiagent.client"):
                fa.connect(api_key="fa_k_no_policy", target=url)
            assert "Could not pull the plane's policy" in caplog.text
            assert "No plane-authored guardrails or approval policies apply" in caplog.text
        finally:
            reset_connection()


def test_a_failed_refresh_warns_and_keeps_the_last_policy(
    isolated_local_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with serve() as (state, url):
        try:
            fa.connect(api_key="fa_k_refresh", target=url)
            state.policy_status = 403
            with caplog.at_level(logging.WARNING, logger="fastaiagent.client"):
                version = fa.refresh_policy()
            assert version == "gov-test-v1"
            assert "Keeping the last policy pulled" in caplog.text
            assert "policy:read" in caplog.text
        finally:
            reset_connection()


def test_an_unreachable_plane_does_not_add_a_policy_warning(
    isolated_local_db: Path, caplog: pytest.LogCaptureFixture
) -> None:
    try:
        with caplog.at_level(logging.WARNING, logger="fastaiagent.client"):
            fa.connect(api_key="fa_k_nowhere", target="http://127.0.0.1:9")
        assert "Could not reach platform" in caplog.text
        assert "Could not pull the plane's policy" not in caplog.text
    finally:
        reset_connection()


# ── public import paths ──────────────────────────────────────────────────────


def test_the_names_a_connected_app_needs_are_public() -> None:
    import fastaiagent.eval
    import fastaiagent.eval.results
    import fastaiagent.guardrail
    from fastaiagent._internal import errors

    assert fa.GuardrailBlockedError is errors.GuardrailBlockedError
    assert fastaiagent.guardrail.GuardrailBlockedError is errors.GuardrailBlockedError
    assert fastaiagent.eval.record_gate_result is fastaiagent.eval.results.record_gate_result
