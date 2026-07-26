"""Plane-authored guardrails: a guardrail created on the Enterprise plane is
distributed via GET /public/v1/policy and enforced at the edge by a connected
SDK agent.

These tests exercise the real builder + real runners + a real Agent (offline
TestModel), with no mocks. The connection's ``policy_cache`` is populated directly
to stand in for what ``connect()`` pulls from the plane.
"""

from __future__ import annotations

import pytest

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.agent.agent import Agent
from fastaiagent.client import _connection
from fastaiagent.guardrail import GuardrailPosition, no_pii
from fastaiagent.guardrail.from_policy import (
    clear_cache,
    guardrail_from_policy_rule,
    plane_guardrails_for_agent,
)
from fastaiagent.testing.models import TestModel

SSN = r"\b\d{3}-\d{2}-\d{4}\b"


def _rule(**overrides):
    rule = {
        "name": "block-ssn-output",
        "guardrail_type": "output",
        "validation_mode": "blocking",
        "implementation_type": "regex",
        "config": {"pattern": SSN},
        "tripwire_message": "SSN blocked",
        "on_error": "block",
        "agent_ids": [],
    }
    rule.update(overrides)
    return rule


def _set_policy(rules, version="v1"):
    _connection.policy_cache = {
        "version": version,
        "guardrail_rules": rules,
        "approval_policies": [],
    }
    clear_cache()


@pytest.fixture(autouse=True)
def _reset_connection():
    saved = _connection.policy_cache
    yield
    _connection.policy_cache = saved
    clear_cache()


# ── builder ──────────────────────────────────────────────────────────────────


def test_builder_regex_blocks_and_passes():
    g = guardrail_from_policy_rule(_rule())
    assert g is not None
    assert g.guardrail_type.value == "regex"
    assert g.position == GuardrailPosition.output
    assert g.blocking is True
    assert g.on_error == "block"
    assert g.execute("my ssn is 123-45-6789").passed is False
    assert g.execute("nothing sensitive here").passed is True


def test_builder_maps_position_blocking_on_error():
    g = guardrail_from_policy_rule(
        _rule(guardrail_type="input", validation_mode="parallel", on_error="allow")
    )
    assert g.position == GuardrailPosition.input
    assert g.blocking is False  # parallel => non-blocking
    assert g.on_error == "allow"


def test_builder_skips_code_rule():
    # A code rule's logic is a server-side callable we don't have.
    assert guardrail_from_policy_rule(_rule(implementation_type="code", config={})) is None


def test_builder_skips_unknown_impl():
    assert guardrail_from_policy_rule(_rule(implementation_type="wat")) is None


def test_builder_classifier_reuses_native_runner():
    g = guardrail_from_policy_rule(
        _rule(
            implementation_type="classifier",
            config={
                "categories": {"competitor": ["langchain", "crewai"]},
                "blocked": ["competitor"],
            },
        )
    )
    assert g.guardrail_type.value == "classifier"
    assert g.execute("we love LangChain").passed is False
    assert g.execute("we use our own stack").passed is True


# ── scoping ──────────────────────────────────────────────────────────────────


def test_unconnected_returns_empty():
    _connection.policy_cache = None
    clear_cache()
    assert plane_guardrails_for_agent("agent-1") == []


def test_domain_wide_applies_to_any_agent():
    _set_policy([_rule(agent_ids=[])])
    assert len(plane_guardrails_for_agent("agent-1")) == 1
    assert len(plane_guardrails_for_agent(None)) == 1


def test_agent_scoped_rule_filters_by_agent_id():
    _set_policy([_rule(agent_ids=["agent-A"])])
    assert len(plane_guardrails_for_agent("agent-A")) == 1
    assert plane_guardrails_for_agent("agent-B") == []


def test_cache_rebuilds_on_version_change():
    _set_policy([_rule()], version="v1")
    assert len(plane_guardrails_for_agent("a")) == 1
    _set_policy([], version="v2")  # policy edited on the plane
    assert plane_guardrails_for_agent("a") == []


# ── enforcement through the Agent ────────────────────────────────────────────


def test_effective_guardrails_unconnected_is_local_only():
    _connection.policy_cache = None
    clear_cache()
    agent = Agent(name="t", llm=TestModel(response="hi"), guardrails=[no_pii()])
    eff = agent._effective_guardrails()
    assert len(eff) == 1  # just the local no_pii


def test_effective_guardrails_merges_local_and_plane():
    _set_policy([_rule()])
    agent = Agent(name="t", llm=TestModel(response="hi"), guardrails=[no_pii()])
    eff = agent._effective_guardrails()
    names = {g.name for g in eff}
    assert "no_pii" in names and "block-ssn-output" in names


async def test_connected_agent_enforces_plane_guardrail_with_no_local_guardrails():
    """The whole point: an agent with NO local guardrails still blocks on a
    guardrail authored on the plane."""
    _set_policy([_rule()])
    agent = Agent(name="support", llm=TestModel(response="Sure — your SSN is 123-45-6789."))
    with pytest.raises(GuardrailBlockedError):
        await agent.arun("what's my ssn?", trace=False)


async def test_same_agent_passes_when_policy_cleared():
    _connection.policy_cache = None
    clear_cache()
    agent = Agent(name="support", llm=TestModel(response="Sure — your SSN is 123-45-6789."))
    result = await agent.arun("what's my ssn?", trace=False)
    assert "123-45-6789" in result.output  # no plane guardrail => not blocked
