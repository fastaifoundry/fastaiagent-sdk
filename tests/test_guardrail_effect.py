"""What a guardrail check did to the run: ``effect_of`` and the event outcome (1.89.0).

``action_taken`` is the resolved action and ignores ``blocking``: an observe-only
``mask`` reports ``masked`` though nothing was applied, and a re-ask reported the
same as a block. The effect reads the two together. Every row of the plan's
truth table, on real check results (no mocks).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fastaiagent.guardrail.actions import EFFECTS, effect_of
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.ui.events import _outcome, event_outcome, rule_snapshot


def _boom(_text: str) -> bool:
    raise RuntimeError("detector down")


def _rule(action: str, *, blocking: bool, state: str) -> Guardrail:
    if state.startswith("errored"):
        return Guardrail(
            name="r",
            guardrail_type=GuardrailType.code,
            position=GuardrailPosition.output,
            fn=_boom,
            action=action,
            blocking=blocking,
            on_error="allow" if state == "errored_allow" else "block",
        )
    return Guardrail(
        name="r",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": "SECRET", "tripwire_message": "I can't share that."},
        action=action,
        blocking=blocking,
    )


def _check(g: Guardrail, state: str) -> Any:
    return asyncio.run(g._acheck("the code is SECRET" if state == "fail" else "all clear"))


# (action, blocking, state, allow_reask) -> (effect, outcome)
TABLE = [
    ("block", True, "clean", False, "none", "passed"),
    ("warn", False, "clean", False, "none", "passed"),
    ("block", True, "errored_allow", False, "unchecked", "errored"),
    ("block", False, "errored_allow", False, "observed", "errored"),
    ("block", True, "errored_block", False, "blocked", "errored"),
    ("mask", False, "errored_block", False, "observed", "errored"),
    ("block", True, "fail", False, "blocked", "blocked"),
    ("warn", True, "fail", False, "warned", "warned"),
    ("mask", True, "fail", False, "rewritten", "filtered"),
    ("override", True, "fail", False, "rewritten", "filtered"),
    ("reask", True, "fail", True, "reasked", "reasked"),
    ("reask", True, "fail", False, "blocked", "blocked"),
    ("block", False, "fail", False, "observed", "warned"),
    ("mask", False, "fail", False, "observed", "warned"),
    ("reask", False, "fail", True, "observed", "warned"),
]


@pytest.mark.parametrize(("action", "blocking", "state", "allow_reask", "effect", "outcome"), TABLE)
def test_the_truth_table(
    action: str, blocking: bool, state: str, allow_reask: bool, effect: str, outcome: str
) -> None:
    g = _rule(action, blocking=blocking, state=state)
    result = _check(g, state)
    got = effect_of(g, result, allow_reask=allow_reask)
    assert got == effect
    assert got in EFFECTS
    assert event_outcome(g, result, got) == outcome


def test_an_observe_only_mask_still_reports_masked_on_the_result() -> None:
    """The result keeps the resolved action (the wire's meaning); the effect: nothing applied."""
    g = _rule("mask", blocking=False, state="fail")
    result = _check(g, "fail")
    assert result.action_taken == "masked"
    assert effect_of(g, result) == "observed"


def test_the_outcome_compatibility_wrapper() -> None:
    g = _rule("reask", blocking=True, state="fail")
    result = _check(g, "fail")
    assert _outcome(g, result) == "blocked"
    assert _outcome(g, result, allow_reask=True) == "reasked"


def test_rule_snapshot_never_carries_a_judge_credential() -> None:
    g = Guardrail(
        name="judge",
        guardrail_type=GuardrailType.llm_judge,
        position=GuardrailPosition.output,
        config={"prompt": "Is it polite?"},
    )
    # A key put in after construction (construction refuses it, 1.89.0 S1).
    g.config["llm"] = {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-live-123"}
    snap = rule_snapshot(g)
    assert snap["config"]["llm"] == {"provider": "openai", "model": "gpt-4o-mini"}
    assert "sk-live-123" not in str(snap)
    assert snap["origin"] == "local"
    assert snap["blocking"] is True and snap["on_error"] == "block"
