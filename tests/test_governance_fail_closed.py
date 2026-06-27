"""Unit tests for the WS4 opt-in fail-closed tool gate (no network).

Drives :func:`fastaiagent.governance.gate_tool_call` directly against the
in-process ``_connection`` singleton — no plane, no LLM. Confirms the new
``governance_fail_mode`` branch is strictly additive: the default ``"open"``
path is byte-identical to today's fail-open behavior, and ``"closed"`` only
refuses when governance genuinely can't be confirmed (connected + enrolled
agent + missing policy cache).
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent import governance
from fastaiagent.client import _connection


@pytest.fixture
def _gov_state():
    """Snapshot + restore the connection fields the gate reads."""
    saved = {
        k: getattr(_connection, k)
        for k in ("api_key", "policy_cache", "governance_fail_mode")
    }
    yield
    for k, v in saved.items():
        setattr(_connection, k, v)


def _gate(tool: str = "transfer_funds", agent_id: str = "agent-1", run_id: str = "run-1"):
    return asyncio.run(governance.gate_tool_call(tool, {"amount": 1}, agent_id, run_id))


def test_fail_open_default_allows_when_no_policy_cache(_gov_state) -> None:
    # Connected + enrolled agent, but no policy cache (plane unreachable at
    # connect). The default "open" mode must allow — unchanged behavior.
    _connection.api_key = "fa_k_test"  # is_connected == True
    _connection.policy_cache = None
    _connection.governance_fail_mode = "open"
    assert _gate() is None


def test_fail_closed_refuses_when_no_policy_cache(_gov_state) -> None:
    _connection.api_key = "fa_k_test"
    _connection.policy_cache = None
    _connection.governance_fail_mode = "closed"  # opt-in
    refusal = _gate()
    assert refusal is not None
    assert "fail-closed" in refusal.lower()
    assert "governance unavailable" in refusal.lower()


def test_fail_closed_noop_without_agent_id(_gov_state) -> None:
    # No agent_id => not enrolled => the gate stays a no-op even when fail-closed.
    _connection.api_key = "fa_k_test"
    _connection.policy_cache = None
    _connection.governance_fail_mode = "closed"
    assert _gate(agent_id="") is None


def test_fail_closed_noop_when_disconnected(_gov_state) -> None:
    _connection.api_key = None  # is_connected == False
    _connection.policy_cache = None
    _connection.governance_fail_mode = "closed"
    assert _gate() is None


def test_fail_closed_allows_when_cache_present_but_unmatched(_gov_state) -> None:
    # The new branch fires ONLY on a MISSING cache. With a populated cache whose
    # patterns don't match the tool, fail-closed must still allow (advisory
    # elsewhere) — documenting the bounded scope of the opt-in mode.
    _connection.api_key = "fa_k_test"
    _connection.policy_cache = {
        "version": "v1",
        "approval_policies": [{"tool_pattern": "wire_transfer", "condition_type": "always"}],
    }
    _connection.governance_fail_mode = "closed"
    assert _gate(tool="send_email") is None
