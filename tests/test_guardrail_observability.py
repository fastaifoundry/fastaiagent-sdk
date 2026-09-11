"""Two silences, closed.

Both findings here are the same shape: the SDK knew something and told nobody.

**d — an unknown position was enforced somewhere else, quietly.**
``from_policy`` maps the plane's ``guardrail_type`` onto an SDK position. An
unknown *implementation type* is logged and skipped; an unknown *position* fell
back to ``output`` and said nothing. The asymmetry was the defect: skipping is
safe, silently relocating is not. A rule authored to gate the user's prompt
would inspect the model's reply instead, and the console would show a healthy
control sitting over an ungated input. The fallback is kept — refusing the rule
outright would be a behaviour change needing sign-off — but it is now audible.

**e — a non-halting guardrail outcome was unobservable.**
``warn``, ``mask`` and ``override`` all let the run finish, which is the point
of them. But ``AgentResult`` carried no guardrail field at all, so a run with
the Local UI off and no plane attached returned the same clean string whether a
rule had fired or not. The one outcome class designed not to stop the run was
the one a caller could not see. ``AgentResult.guardrails`` is the fix.

Every assertion below was negative-controlled: reverted against the 1.63.0
behaviour, watched fail, restored.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from fastaiagent.guardrail.from_policy import guardrail_from_policy_rule
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailType,
    collected_firings,
    start_firing_collection,
    stop_firing_collection,
)

# --------------------------------------------------------------- d


def _rule(**over):
    """A minimally valid plane rule, overridable per test."""
    base = {
        "name": "gate_the_prompt",
        "implementation_type": "regex",
        "guardrail_type": "input",
        "config": {"pattern": "secret", "should_match": False},
    }
    base.update(over)
    return base


class TestAnUnknownPositionIsAudible:
    def test_the_rule_is_still_built_and_still_lands_on_output(self, caplog):
        """The fallback is deliberately unchanged.

        Refusing the rule would be safer, but it changes what an existing
        customer rule does — that is row #7's category, not this one.
        """
        g = guardrail_from_policy_rule(_rule(guardrail_type="inputs"))

        assert g is not None, "an unknown position must not drop the rule"
        assert g.position is GuardrailPosition.output

    def test_it_now_says_so(self, caplog):
        with caplog.at_level(logging.WARNING, logger="fastaiagent.guardrail.from_policy"):
            guardrail_from_policy_rule(_rule(guardrail_type="inputs"))

        assert caplog.records, "an unknown position used to fall back in total silence"
        msg = caplog.records[0].getMessage()
        assert "inputs" in msg, "the operator needs the value they actually wrote"
        assert "gate_the_prompt" in msg, "and which rule it was"

    def test_a_known_position_stays_quiet(self, caplog):
        """The negative control on the log line itself.

        Without this, a build that warned unconditionally would pass the test
        above — and every connected run would carry a spurious warning.
        """
        with caplog.at_level(logging.WARNING, logger="fastaiagent.guardrail.from_policy"):
            for known in ("input", "output", "tool", "tool_call", "tool_result"):
                guardrail_from_policy_rule(_rule(guardrail_type=known))

        assert caplog.records == []

    def test_the_plane_tool_alias_is_not_an_unknown_position(self, caplog):
        """``tool`` is the plane's own vocabulary — it models one tool phase and
        the SDK splits it in two. Warning on it would train operators to ignore
        the warning."""
        with caplog.at_level(logging.WARNING, logger="fastaiagent.guardrail.from_policy"):
            g = guardrail_from_policy_rule(_rule(guardrail_type="tool"))

        assert g is not None
        assert g.position is GuardrailPosition.tool_call
        assert caplog.records == []

    def test_a_missing_position_defaults_without_warning(self, caplog):
        """No ``guardrail_type`` key at all is a pre-v1.9 rule, not a typo."""
        rule = _rule()
        del rule["guardrail_type"]
        with caplog.at_level(logging.WARNING, logger="fastaiagent.guardrail.from_policy"):
            g = guardrail_from_policy_rule(rule)

        assert g is not None
        assert g.position is GuardrailPosition.output
        assert caplog.records == []


# --------------------------------------------------------------- e


def _warning_rule() -> Guardrail:
    """A rule that fails and lets the run continue — the case that was invisible."""
    return Guardrail(
        name="no_secrets_in_prompt",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.input,
        config={"pattern": "secret", "should_match": False},
        action="warn",
    )


def _clean_rule() -> Guardrail:
    return Guardrail(
        name="clean",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.input,
        config={"pattern": "zzzz", "should_match": False},
        action="warn",
    )


class TestFiringsAreCollected:
    def test_a_warn_that_does_not_stop_the_run_is_now_visible(self):
        token = start_firing_collection()
        try:
            result = asyncio.run(_warning_rule().aexecute("my secret is here"))
            firings = collected_firings()
        finally:
            stop_firing_collection(token)

        assert result.passed is False
        assert result.action_taken == "warned"

        assert len(firings) == 1, "the whole point: the run continued and said so"
        f = firings[0]
        assert f.name == "no_secrets_in_prompt"
        assert f.position == "input"
        assert f.action_taken == "warned"
        assert f.passed is False
        assert f.fired() is True

    def test_a_clean_pass_is_recorded_but_does_not_count_as_fired(self):
        token = start_firing_collection()
        try:
            asyncio.run(_clean_rule().aexecute("nothing to see"))
            firings = collected_firings()
        finally:
            stop_firing_collection(token)

        assert len(firings) == 1, "every execution is recorded, not just the failures"
        assert firings[0].fired() is False, "a clean pass is not a firing"
        assert firings[0].action_taken == "none"

    @pytest.mark.parametrize("ui_on", [False, True])
    def test_it_works_whether_or_not_the_local_ui_is_on(self, monkeypatch, request, ui_on):
        """The case the finding is actually about.

        ``log_guardrail_event`` sits behind ``ui_enabled``; the firing append
        must **not**, or it closes nothing for an unconnected run — which is the
        whole finding. The discriminating half is ``ui_on=False``: an append
        hidden inside that gate would collect nothing there.

        Forced explicitly rather than relying on the default. An earlier draft
        asserted ``ui_enabled is False`` on the theory that the default made the
        monkeypatch redundant — it passed alone and **failed in the full suite**,
        because something earlier in the run turns the UI on and the config is an
        ``lru_cache``. The value is not this test's to assume in either
        direction, so it sets both.
        """
        from fastaiagent._internal.config import get_config, reset_config

        monkeypatch.setenv("FASTAIAGENT_UI_ENABLED", "1" if ui_on else "0")
        reset_config()
        request.addfinalizer(reset_config)  # the lru_cache outlives the env patch
        assert get_config().ui_enabled is ui_on, "the precondition must actually hold"

        token = start_firing_collection()
        try:
            asyncio.run(_warning_rule().aexecute("my secret is here"))
            firings = collected_firings()
        finally:
            stop_firing_collection(token)

        assert len(firings) == 1

    def test_nothing_is_recorded_outside_a_run(self):
        """A bare ``guardrail.execute(...)`` has no run to attribute itself to.

        Also the leak check: if the collector were a module-level list rather
        than a ContextVar, this would accumulate across every call in the
        process and grow without bound.
        """
        assert collected_firings() == []
        _warning_rule().execute("my secret is here")
        assert collected_firings() == []

    def test_scopes_do_not_bleed_into_each_other(self):
        first = start_firing_collection()
        try:
            asyncio.run(_warning_rule().aexecute("my secret is here"))
            assert len(collected_firings()) == 1
        finally:
            stop_firing_collection(first)

        second = start_firing_collection()
        try:
            assert collected_firings() == [], "a new run starts empty"
            asyncio.run(_clean_rule().aexecute("fine"))
            assert len(collected_firings()) == 1
        finally:
            stop_firing_collection(second)

        assert collected_firings() == [], "and the scope closes"

    def test_order_is_execution_order(self):
        token = start_firing_collection()
        try:
            asyncio.run(_clean_rule().aexecute("fine"))
            asyncio.run(_warning_rule().aexecute("my secret is here"))
            names = [f.name for f in collected_firings()]
        finally:
            stop_firing_collection(token)

        assert names == ["clean", "no_secrets_in_prompt"]

    def test_an_errored_check_counts_as_fired(self):
        """``errored`` is the other non-obvious case: under ``on_error="allow"``
        it yields ``passed=True``, so a caller branching on ``passed`` alone
        cannot tell a degraded pass from a clean one.

        ``groundedness`` with no context raises inside ``extract_pair``, before
        any model call — so this stays hermetic.
        """
        g = Guardrail(
            name="ungrounded",
            guardrail_type=GuardrailType.groundedness,
            position=GuardrailPosition.output,
            config={},
            on_error="allow",
        )

        token = start_firing_collection()
        try:
            asyncio.run(g.aexecute("an answer with no context to check it against"))
            firings = collected_firings()
        finally:
            stop_firing_collection(token)

        assert len(firings) == 1
        assert firings[0].errored is True
        assert firings[0].passed is True, "on_error='allow' degrades to a pass"
        assert firings[0].fired() is True, "a degraded pass must not read as clean"

    def test_an_unrunnable_regex_is_still_a_verdict_not_an_error(self):
        """Pinned, not endorsed.

        A malformed pattern yields ``passed=False, errored=False`` — a genuine
        block verdict rather than a check that could not run, so ``on_error``
        never gets consulted. The plane calls the same case an error. That is
        the open half of row **#11**, "needs agreement": recorded here so the
        next session sees the current behaviour instead of rediscovering it,
        and so a change to it fails loudly rather than silently.

        This test found its own premise wrong — it was written asserting
        ``errored is True``.
        """
        g = Guardrail(
            name="broken",
            guardrail_type=GuardrailType.regex,
            position=GuardrailPosition.input,
            config={"pattern": "(unclosed", "should_match": True},
            on_error="allow",
        )

        token = start_firing_collection()
        try:
            asyncio.run(g.aexecute("anything"))
            firings = collected_firings()
        finally:
            stop_firing_collection(token)

        assert firings[0].errored is False
        assert firings[0].passed is False
        assert firings[0].action_taken == "blocked"
        assert firings[0].fired() is True, "however it is classified, it must be visible"

    def test_the_record_carries_no_payload(self):
        """``GuardrailFiring`` is deliberately narrower than ``GuardrailResult``.

        A ``pii`` rule's ``metadata["matches"]`` holds the matched value itself,
        and ``modified_data`` is the rewritten payload. Neither belongs on an
        object handed back beside the answer.
        """
        from fastaiagent.guardrail.guardrail import GuardrailFiring

        fields = set(GuardrailFiring.model_fields)
        assert "metadata" not in fields
        assert "modified_data" not in fields
        assert fields == {"name", "position", "action_taken", "passed", "errored", "message"}


class TestAgentResultCarriesThem:
    def test_the_field_exists_and_defaults_empty(self):
        """Additive: every existing caller keeps working, and a run with no
        guardrails is an empty list rather than ``None``."""
        from fastaiagent.agent.agent import AgentResult

        r = AgentResult(output="hi")
        assert r.guardrails == []

    # The agent-level integration — a real model, a real masking rule, and
    # ``AgentResult.guardrails`` reporting it — lives in
    # ``tests/e2e/test_guardrail_actions_e2e.py`` rather than here. It needs
    # an LLM, and a canned reply would skip the middle of the path it claims
    # to cover. Everything above uses real guardrails and no stubs.

    def test_the_message_is_carried_for_diagnosis(self):
        token = start_firing_collection()
        try:
            asyncio.run(_warning_rule().aexecute("my secret is here"))
            firings = collected_firings()
        finally:
            stop_firing_collection(token)

        assert firings[0].message, "a firing with no reason is barely better than silence"


@pytest.mark.parametrize("action", ["warn", "mask"])
def test_both_non_halting_actions_are_observable(action):
    """The two that let the run finish. ``block`` raises and was always visible;
    these two were not."""
    g = Guardrail(
        name=f"rule_{action}",
        guardrail_type=GuardrailType.pii,
        position=GuardrailPosition.input,
        config={"entities": ["ssn"]},
        action=action,
    )

    token = start_firing_collection()
    try:
        asyncio.run(g.aexecute("ssn 123-45-6789"))
        firings = collected_firings()
    finally:
        stop_firing_collection(token)

    assert len(firings) == 1
    assert firings[0].fired() is True
