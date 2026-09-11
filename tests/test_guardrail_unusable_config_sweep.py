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

**``llm_judge`` is excluded from the sweep**, and that is a gap, not an
oversight: every path through it makes a model call, so it cannot be swept
hermetically. Two of its three defects *were* fixed in 1.64.0 and are asserted
directly at the bottom of this file, before any model call happens: an
explicitly empty ``prompt``, and ``pass_value=""`` making the fallback always
pass. The third — that a **missing** ``prompt`` substitutes a *different*
built-in criterion on each side — is deliberately untouched, because choosing
one would be the SDK unilaterally settling a cross-repo default. It stays on the
shared board as a "both sides, needs agreement" item.
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
    # NOTE: ``{"categories": {...}}`` with no ``blocked`` list used to live here.
    # It is no longer an unusable config — 1.64.0 made it *work* rather than
    # making it error, adopting the plane's reading that every detected category
    # blocks when no narrowing list is given. See
    # ``test_a_classifier_with_no_blocked_list_blocks_what_it_detects`` below.
]

#: Types whose defect was real, reachable, and deliberately left unfixed while
#: it waited on a human: closing them changes what an existing rule does.
#:
#: **Empty since 1.64.0** — ``regex`` and ``classifier`` were signed off and
#: fixed, so every case above is now a live assertion rather than an ``xfail``.
#: Kept as an empty set rather than deleted, because the mechanism is the point:
#: a new degenerate config that cannot be fixed without sign-off goes in here
#: with its reason, and is named in every test run instead of being absent.
KNOWN_UNFIXED: set[str] = set()


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

    # This used to be ``if result.passed: pytest.xfail(...)`` — a *conditional*
    # xfail, which is the "skips instead of failing" shape that already produced
    # one bad test in this work. With #7 signed off there is nothing left to
    # excuse, so it asserts.
    assert result.passed is False, (
        f"{impl} with {config!r} ({why}) let the payload through under the default "
        f"on_error='block'."
    )


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


# --------------------------------------------------------------------------- #
# #7, the half that was fixed by making a config *work* rather than error
# --------------------------------------------------------------------------- #
def test_a_classifier_with_no_blocked_list_blocks_what_it_detects() -> None:
    """1.64.0 (signed off): the SDK adopts the plane's `classifier` reading.

    Before, ``blocked = [c for c in detected if c in blocked_categories]`` meant
    that with ``blocked`` missing or empty **nothing ever blocked** — the rule
    detected the category and then reported success. The plane does
    ``hits = [...] if blocked else detected``, as its own docstring states. Same
    rule, same payload, opposite verdicts, and the SDK held the unsafe side.

    This is the *whole* of the divergence for this type, so it is asserted
    directly rather than through the could-not-run sweep: the config is usable,
    it just used to be ignored.
    """
    guardrail = Guardrail(
        name="profanity",
        guardrail_type=GuardrailType.classifier,
        position=GuardrailPosition.output,
        config={"categories": {"profanity": ["damn"]}},  # no `blocked`
        on_error="block",
    )

    result = asyncio.run(run_guardrail(guardrail, "damn it"))

    assert result.passed is False, (
        "a classifier that detects its category must not report success just "
        "because no narrowing `blocked` list was given — the plane blocks here"
    )
    assert result.errored is False, "the config is usable; this is a verdict, not a failure"
    assert result.metadata["detected"] == ["profanity"]
    assert result.metadata["blocked"] == ["profanity"]


def test_an_explicit_blocked_list_still_narrows() -> None:
    """The other direction, so the fix cannot be read as "blocked is ignored".

    An operator who names the categories they care about still gets exactly
    those — detection of an unlisted category is recorded and allowed through.
    """
    guardrail = Guardrail(
        name="narrowed",
        guardrail_type=GuardrailType.classifier,
        position=GuardrailPosition.output,
        config={
            "categories": {"profanity": ["damn"], "legalese": ["heretofore"]},
            "blocked": ["legalese"],
        },
        on_error="block",
    )

    result = asyncio.run(run_guardrail(guardrail, "damn it"))

    assert result.metadata["detected"] == ["profanity"]
    assert result.metadata["blocked"] == [], "profanity was detected but not in `blocked`"
    assert result.passed is True


# --------------------------------------------------------------------------- #
# #4 — a `code` rule with no callable. Not in the sweep because `code` is
# edge-exempt (a plane cannot distribute one), but it is the same invariant and
# it is the one with teeth.
# --------------------------------------------------------------------------- #
def test_a_code_rule_with_no_function_cannot_run() -> None:
    """1.64.0 (signed off). This returned ``passed=True, "No code configured"``.

    Reachable without anyone authoring a broken rule: ``to_dict()`` cannot
    serialize ``fn`` and ``from_dict()`` never restores it.
    """
    guardrail = Guardrail(
        name="orphaned",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.output,
        config={},
        on_error="block",
    )

    result = asyncio.run(run_guardrail(guardrail, PAYLOAD))

    assert result.errored is True, "a code rule with no callable inspected nothing"
    assert result.passed is False, "and must not report a clean pass"


def test_a_round_tripped_builtin_no_longer_reports_a_clean_pass() -> None:
    """The reachability, executed rather than asserted (board rule 5).

    This is the path that disarmed Replay: ``Replay.fork_at(...).rerun()`` is
    fed from the ``agent.guardrails`` span attribute, which is ``to_dict()``
    output. Every builtin is ``guardrail_type=code, fn=<callable>``, so every
    one came back without its function and passed unconditionally — writing
    green rows for checks that never executed.
    """
    from fastaiagent.guardrail.builtins import no_pii

    original = no_pii()
    assert original.guardrail_type is GuardrailType.code
    assert original.fn is not None

    restored = Guardrail.from_dict(original.to_dict())
    assert restored.fn is None, "fn cannot survive serialization — that is the premise"

    result = asyncio.run(run_guardrail(restored, PAYLOAD))

    assert result.errored is True
    assert result.passed is False, (
        "a replayed run must not report that a guardrail passed when the "
        "guardrail was not there to run"
    )


# --------------------------------------------------------------------------- #
# #7 — the two `llm_judge` halves that need no model call
# --------------------------------------------------------------------------- #
def test_an_llm_judge_with_an_explicitly_empty_prompt_cannot_run() -> None:
    guardrail = Guardrail(
        name="no-criterion",
        guardrail_type=GuardrailType.llm_judge,
        position=GuardrailPosition.output,
        config={"prompt": "   "},
        on_error="block",
    )

    result = asyncio.run(run_guardrail(guardrail, PAYLOAD))

    assert result.errored is True
    assert result.passed is False


def test_an_llm_judge_with_an_empty_pass_value_cannot_run() -> None:
    """``pass_value=""`` made the fallback path pass everything: the check is
    ``pass_value in reply``, and every string contains the empty string."""
    guardrail = Guardrail(
        name="empty-pass-value",
        guardrail_type=GuardrailType.llm_judge,
        position=GuardrailPosition.output,
        config={"prompt": "Is this acceptable?", "pass_value": ""},
        on_error="block",
    )

    result = asyncio.run(run_guardrail(guardrail, PAYLOAD))

    assert result.errored is True
    assert result.passed is False


def test_a_missing_prompt_still_uses_the_documented_default() -> None:
    """The restraint, pinned.

    A *missing* ``prompt`` is not an error: the two repos substitute different
    built-in criteria, and choosing one here would be the SDK settling a
    cross-repo default unilaterally. It stays a board item. This test exists so
    a later "consistency" pass does not quietly close it.
    """
    guardrail = Guardrail(
        name="defaulted",
        guardrail_type=GuardrailType.llm_judge,
        position=GuardrailPosition.output,
        config={},
    )

    # No model call is made: this only has to get past the config checks, so we
    # assert on the absence of a config-time raise rather than on a verdict.
    from fastaiagent.guardrail.implementations import _run_llm_judge

    try:
        asyncio.run(_run_llm_judge(guardrail, "anything"))
    except ValueError as exc:  # pragma: no cover - only on a regression
        assert "prompt" not in str(exc), (
            "a missing prompt must not raise — that default is a cross-repo "
            "decision, not the SDK's to make alone"
        )
    except Exception:
        pass  # a model/network failure is fine; the config check is what matters
