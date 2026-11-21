"""``fastaiagent.testing.plane_rules`` — an offline stand-in for platform-authored rules (1.105.0).

Without it, an offline test of "my app handles a platform block" meant writing into
``fastaiagent.client._connection.policy_cache`` by hand, which coding assistants did and then
reported as proof. The helper sends rules through the real policy cache and builder, restores
what was there, and refuses a rule a connected agent would skip.

Also checks the promise that makes it honest: a rule behaves here as it does when the same
rule arrives from a (stand-in) platform over HTTP.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

import fastaiagent as fa
from fastaiagent.client import _connection
from fastaiagent.testing import TestModel, plane_rules
from tests._governance_plane import reset_connection, serve


def _rule(**over: Any) -> dict[str, Any]:
    return {
        "name": "no-codenames",
        "guardrail_type": "input",
        "validation_mode": "blocking",
        "implementation_type": "regex",
        "config": {"pattern": r"(?i)\bbluejay\b", "should_match": False},
        "on_error": "block",
        **over,
    }


def _agent(reply: str = "ok", **kw: Any) -> fa.Agent:
    return fa.Agent(name="support", llm=TestModel(reply), **kw)


def test_an_input_rule_blocks_and_names_itself(isolated_local_db: Path) -> None:
    with plane_rules([_rule()]):
        with pytest.raises(fa.GuardrailBlockedError) as err:
            _agent().run("tell me about bluejay")
    assert err.value.guardrail_name == "no-codenames"


def test_a_benign_input_passes_and_the_check_is_on_the_result(isolated_local_db: Path) -> None:
    with plane_rules([_rule()]):
        result = _agent("fine").run("how do refunds work?")
    assert result.output == "fine"
    assert [f.name for f in result.guardrails] == ["no-codenames"]


def test_an_output_rule_and_a_streamed_run(isolated_local_db: Path) -> None:
    rule = _rule(
        name="no-ssn",
        guardrail_type="output",
        config={"pattern": r"\d{3}-\d{2}-\d{4}", "should_match": False},
    )
    agent = _agent("SSN is 123-45-6789")

    async def drain() -> None:
        async for _ in agent.astream("hi"):
            pass

    with plane_rules([rule]):
        with pytest.raises(fa.GuardrailBlockedError):
            agent.run("hi")
        with pytest.raises(fa.GuardrailBlockedError):
            asyncio.run(drain())


def test_outside_the_block_nothing_is_enforced_and_state_is_restored(
    isolated_local_db: Path,
) -> None:
    before = _connection.policy_cache
    with plane_rules([_rule()]):
        assert fa.plane_guardrails_for_agent(None)
    assert _connection.policy_cache is before
    assert fa.plane_guardrails_for_agent(None) == []
    assert _agent("ok").run("tell me about bluejay").output == "ok"


def test_state_is_restored_when_the_block_raises(isolated_local_db: Path) -> None:
    before = _connection.policy_cache
    with pytest.raises(RuntimeError):
        with plane_rules([_rule()]):
            raise RuntimeError("test body failed")
    assert _connection.policy_cache is before
    assert fa.plane_guardrails_for_agent(None) == []


def test_a_rule_scoped_to_an_agent_applies_only_to_it(isolated_local_db: Path) -> None:
    with plane_rules([_rule(agent_ids=["agent-7"])]):
        assert _agent().run("tell me about bluejay").output == "ok"  # no id: not its rule
        with pytest.raises(fa.GuardrailBlockedError):
            _agent(agent_id="agent-7").run("tell me about bluejay")


def test_the_rules_are_marked_plane_authored(isolated_local_db: Path) -> None:
    with plane_rules([_rule()]):
        (built,) = fa.plane_guardrails_for_agent(None)
    assert built.origin == "plane"


def test_a_rule_a_connected_agent_would_skip_is_refused(isolated_local_db: Path) -> None:
    """A ``code`` rule's logic lives on the platform, so the edge skips it. A helper that
    accepted it would give a test that checks nothing."""
    before = _connection.policy_cache
    with pytest.raises(ValueError, match="cannot be enforced"):
        with plane_rules([_rule(name="server-side", implementation_type="code", config={})]):
            pass
    assert _connection.policy_cache is before


def test_the_callers_rules_are_not_mutated(isolated_local_db: Path) -> None:
    rule = _rule()
    snapshot = dict(rule)
    with plane_rules([rule]):
        pass
    assert rule == snapshot


def test_a_rule_behaves_the_same_from_the_helper_and_from_a_platform(
    isolated_local_db: Path,
) -> None:
    """The point of the helper: same rule, same verdicts, whether it came from plane_rules or
    over HTTP from a platform (the shared real-HTTP stand-in)."""
    rule = _rule()
    probes = ["tell me about bluejay", "how do refunds work?"]

    def verdicts() -> list[str]:
        out = []
        for text in probes:
            try:
                _agent().run(text)
                out.append("passed")
            except fa.GuardrailBlockedError as err:
                out.append(f"blocked:{err.guardrail_name}")
        return out

    with plane_rules([rule]):
        offline = verdicts()
    with serve() as (state, url):
        state.guardrail_rules = [rule]
        try:
            fa.connect(api_key="fa_k_same_rule", target=url)
            connected = verdicts()
        finally:
            reset_connection()
    assert offline == connected == ["blocked:no-codenames", "passed"]
