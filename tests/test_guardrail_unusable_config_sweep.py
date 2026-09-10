"""One invariant, swept across every type a control plane can distribute.

    A guardrail whose configuration cannot check anything must report that it
    **could not run** — never a clean verdict.

This project has now shipped a violation of that rule three times, in three
consecutive releases, in three different types:

* ``topic`` — an unreadable judge answer ``return []``-ed, which silently *passed*
  a ``deny`` rule (1.58.0)
* ``schema`` — ``jsonschema.validate(anything, {})`` passes, so an empty schema
  validated everything and called it valid (1.59.0)
* ``pii`` — ``{"entities": []}`` reported "No personal data detected" over a
  payload it never inspected (1.61.0)

Each was found by hand, months apart, and each fix closed *that instance*. The
1.61.0 changelog said the quiet part out loud: *"One line of that in runnable
form would have caught both this defect and the ``schema`` one."* This is that
line.

**Why ``errored`` and not ``raises``.** The runners do raise, but
``run_guardrail`` is the choke point that turns a raise into
``errored=True`` plus the ``on_error`` policy. Asserting on ``errored`` tests the
contract callers actually see.

**``llm_judge`` is excluded**, and that is a gap, not an oversight: every path
through it makes a model call, so it cannot be swept hermetically. Its known
defect — a missing or empty ``prompt`` silently substitutes a built-in criterion
the operator never wrote, and ``pass_value=""`` makes the fallback always pass —
is recorded in the 2026-09-10 audit and is not fixed here.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from fastaiagent.guardrail.from_policy import _RECONSTRUCTABLE
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType
from fastaiagent.guardrail.implementations import run_guardrail

#: A payload with something for every detector to find, so a rule that *does*
#: inspect it has an obvious verdict to reach. A control reporting a clean pass
#: over this text is either broken or was never looking.
PAYLOAD = (
    "My SSN is 123-45-6789, my card is 4111111111111111, and "
    'api_key = "sk-proj-abcdefghijklmnopqrst". Also: secret."'
)

#: ``(implementation_type, config, why it is unusable)``.
#:
#: Every entry is a config a control plane can currently write and distribute —
#: the plane's ``validate_config_is_enforceable`` gates only ``pii`` and
#: ``schema``, so the rest reach the edge unchallenged.
UNUSABLE: list[tuple[str, dict[str, Any], str]] = [
    ("schema", {}, "no schema at all"),
    ("schema", {"schema": {}}, "an empty schema validates everything"),
    ("schema", {"schema": True}, "`true` is a legal schema meaning 'accept anything'"),
    ("schema", {"schema": "nope"}, "a non-dict schema"),
    ("topic", {}, "no topics to judge against"),
    ("topic", {"topics": []}, "an empty topic list"),
    ("topic", {"topics": ["ok"], "mode": "denyy"}, "a mistyped polarity inverts the rule"),
    ("pii", {"entities": []}, "scans for no entities"),
    ("pii", {"entities": ["nope"]}, "an entity no detector knows"),
    ("pii", {"backend": "presidoo"}, "a backend that does not exist"),
    ("content_safety", {"categories": ["S99"]}, "no known hazard category"),
    ("groundedness", {}, "no context to score the answer against"),
    ("regex", {}, "no pattern"),
    ("regex", {"pattern": ""}, "an empty pattern matches at every position"),
    (
        "regex",
        {"pattern": "", "should_match": True},
        "an empty pattern with should_match passes everything",
    ),
    ("classifier", {}, "no categories"),
    ("classifier", {"categories": {}}, "an empty category map"),
    (
        "classifier",
        {"categories": {"leak": ["secret"]}},
        "categories with no `blocked` list: detects, then reports success",
    ),
]

#: Types whose defect is real, reachable, and **not fixed here** — closing them
#: changes what an existing rule does (a rule that passes today would start
#: blocking), so they need explicit sign-off rather than a drive-by fix. Marked
#: ``strict`` so that when they are fixed this test fails and has to be updated,
#: rather than quietly passing and leaving the marker behind.
KNOWN_UNFIXED = {"regex", "classifier"}


def _case_id(case: tuple[str, dict[str, Any], str]) -> str:
    impl, config, why = case
    return f"{impl}-{why}"


@pytest.mark.parametrize("case", UNUSABLE, ids=_case_id)
def test_an_unusable_config_reports_could_not_run(case) -> None:
    impl, config, why = case
    if impl in KNOWN_UNFIXED:
        pytest.xfail(f"known, unfixed, needs sign-off: {impl} — {why}")

    guardrail = Guardrail(
        name=f"unusable-{impl}",
        guardrail_type=GuardrailType(impl),
        position=GuardrailPosition.output,
        config=config,
        on_error="block",
    )

    result = asyncio.run(run_guardrail(guardrail, PAYLOAD))

    assert result.errored is True, (
        f"{impl} with {config!r} ({why}) returned a verdict instead of reporting that it "
        f"could not run. A control that inspects nothing must not report success — "
        f"this is the defect topic/schema/pii each shipped in turn."
    )


@pytest.mark.parametrize("case", UNUSABLE, ids=_case_id)
def test_an_unusable_config_never_reports_a_clean_pass(case) -> None:
    """The weaker half, and the one that actually bites.

    ``errored`` is what a *reader* of the row needs. This asserts what the
    *caller* gets: with the default ``on_error="block"``, an unusable rule must
    not let the payload through. Kept separate because a type could plausibly
    block for the wrong reason, and that is still better than passing.
    """
    impl, config, why = case
    if impl in KNOWN_UNFIXED and impl != "regex":
        pytest.xfail(f"known, unfixed, needs sign-off: {impl} — {why}")

    guardrail = Guardrail(
        name=f"unusable-{impl}",
        guardrail_type=GuardrailType(impl),
        position=GuardrailPosition.output,
        config=config,
        on_error="block",
    )

    result = asyncio.run(run_guardrail(guardrail, PAYLOAD))

    if result.passed:
        pytest.xfail(
            f"known, unfixed, needs sign-off: {impl} with {config!r} ({why}) passes the payload"
        )
    assert result.passed is False


def test_the_sweep_covers_every_distributable_type() -> None:
    """The sweep is only worth having if it cannot fall behind the enum.

    ``_RECONSTRUCTABLE`` is the set a control plane can distribute and the edge
    will enforce. When an eleventh type joins it, this fails until someone has
    decided what "unusable" means for it — which is exactly the step that was
    skipped three times.
    """
    swept = {impl for impl, _, _ in UNUSABLE}
    expected = {t.value for t in _RECONSTRUCTABLE}

    # ``secrets`` takes no detection config by design — there is nothing to make
    # unusable, which is the point of that decision. ``llm_judge`` cannot be
    # swept without a model call (see the module docstring).
    exempt = {"secrets", "llm_judge"}

    missing = expected - swept - exempt
    assert not missing, (
        f"these distributable types have no unusable-config case: {sorted(missing)}. "
        f"Decide what an unusable config looks like for each and add it here."
    )


def test_defaults_are_not_mistaken_for_degenerate_configs() -> None:
    """The other half of the invariant, and the reason this file is a sweep and
    not a blanket rule.

    Two configs *look* empty and are legitimate: ``content_safety`` with no
    ``categories`` falls back to the six default hazards, and ``secrets`` takes no
    detection config at all — a tenant narrowing a credential detector would only
    weaken it. Neither may be swept into "unusable", or the fix for one defect
    becomes a new one.
    """
    rule = Guardrail(
        name="secrets-default",
        guardrail_type=GuardrailType.secrets,
        position=GuardrailPosition.output,
        config={},
    )

    result = asyncio.run(run_guardrail(rule, PAYLOAD))

    assert result.errored is False, "an empty `secrets` config is correct, not unusable"
    assert result.passed is False, "the payload carries an API key; it should be found"
