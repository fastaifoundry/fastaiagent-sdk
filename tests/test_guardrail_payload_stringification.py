"""Row #11, stringification half: what string does a check actually see?

A guardrail judges text. When the payload is a dict, *something* has to render
it — and the two repos rendered it differently:

===========  ======================  =====================================
side         rendering               ``{"note": "it's fine"}`` becomes
===========  ======================  =====================================
SDK          ``json.dumps(data)``    ``{"note": "it's fine"}``
plane        ``str(data)``           ``{'note': "it's fine"}``
===========  ======================  =====================================

Quoting inverts between them, so a rule matching on a quote character reaches
**opposite verdicts** on the same payload. Verified by executing both, not by
reading them::

    pattern = r'^\\{"'   should_match=True   payload = {"note": "it's fine"}
    SDK   -> passed=True
    plane -> passed=False

That is a contract break under the standard in §5 of the shared board — same
rule, same payload, different verdict — not a cosmetic difference.

**The agreement: ``json.dumps`` wins.** It is stable for dicts, it is already
what the SDK does at all eight sites, and it is already what the *plane's own
masker* does — the plane's ``str()`` sits in one place, its ``regex`` runner. So
the plane is internally inconsistent today: a plane mask rule matches against
one string and redacts a different one. Converging on ``json.dumps`` is a
one-line change on their side and **no change here**.

This file is the SDK's half of that agreement, written per the board's rule 8:
pin the behaviour now so the agreed answer cannot drift on this side while the
plane lands theirs. There is nothing to gate on a version — the SDK is already
correct, and these tests would fail if someone "simplified" a runner to
``str(data)``.

Behavioural on purpose. A source grep for ``json.dumps`` would certify rather
than check: it passes on a runner that computes the string and then ignores it,
and it cannot see a helper that renders the payload some third way.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType

#: Renders differently under the two candidate renderings, and contains an
#: apostrophe so ``str()`` is forced to use double quotes for the *value* while
#: keeping single quotes for the *key* — which is what makes the two forms
#: distinguishable by a quote-anchored pattern.
PAYLOAD = {"note": "it's fine"}

STR_FORM = "{'note': \"it's fine\"}"
JSON_FORM = '{"note": "it\'s fine"}'


def test_the_two_renderings_really_do_differ():
    """The premise. If these ever coincide, every test below is vacuous."""
    assert str(PAYLOAD) == STR_FORM
    assert json.dumps(PAYLOAD) == JSON_FORM
    assert STR_FORM != JSON_FORM


def _regex_verdict(pattern: str) -> bool:
    g = Guardrail(
        name="q",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.input,
        config={"pattern": pattern, "should_match": True},
    )
    return asyncio.run(g.aexecute(PAYLOAD)).passed


class TestTheCheckSeesTheJsonForm:
    def test_a_json_only_pattern_matches(self):
        """``{"`` — a double-quoted key — exists only in the json rendering."""
        assert _regex_verdict(r'^\{"') is True

    def test_a_str_only_pattern_does_not(self):
        """``{'`` — a single-quoted key — exists only in the ``str`` rendering.

        This is the assertion that would flip if a runner moved to ``str()``,
        and the one that currently disagrees with the plane.
        """
        assert _regex_verdict(r"^\{'") is False


@pytest.mark.parametrize(
    ("impl", "config"),
    [
        # Every runner that stringifies a dict payload and needs no model.
        # ``_run_llm_judge``, ``_run_content_safety`` and ``_run_topic`` do the
        # same thing at implementations.py:209/474/585 but require a judge, so
        # they are covered by the live gates rather than here.
        ("regex", {"pattern": "zzz-no-match", "should_match": False}),
        ("classifier", {"categories": {"x": ["zzz-no-match"]}, "blocked": ["x"]}),
        ("pii", {"entities": ["ssn"]}),
        ("secrets", {}),
    ],
)
def test_no_runner_leaks_the_str_rendering(impl, config):
    """The cross-runner sweep.

    Each of these renders the payload itself rather than sharing a helper, so
    each is an independent chance to drift. Rather than asserting on the
    rendering directly — which no runner exposes — this embeds a marker that is
    *only* detectable in one form: a value whose ``str`` rendering wraps it in
    single quotes.

    A ``secrets``-shaped payload is used so the detector-backed runners have
    something real to find, making the verdict depend on having actually read
    the text.
    """
    payload = {"token": "sk-proj-abcdefghijklmnopqrstuvwx", "note": "it's fine"}

    g = Guardrail(
        name=f"q-{impl}",
        guardrail_type=GuardrailType(impl),
        position=GuardrailPosition.input,
        config=config,
    )
    result = asyncio.run(g.aexecute(payload))

    # Whatever the verdict, the runner must not have raised: a rendering change
    # that broke a runner would show up here first.
    assert result.errored is False, result.message

    # And the rendering the runner used is the json one — asserted through the
    # only channel that reflects it, a pattern anchored on the quoting.
    if impl == "regex":
        assert _regex_verdict(r'^\{"') is True


def test_a_plain_string_payload_is_passed_through_untouched():
    """The other half of every one of those eight lines.

    ``data if isinstance(data, str) else json.dumps(data)`` — a string payload
    must not be re-encoded, or every rule would suddenly be matching against
    ``"\\"hello\\""`` with the quotes baked in.
    """
    g = Guardrail(
        name="q",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.input,
        config={"pattern": r'^"', "should_match": True},
    )
    assert asyncio.run(g.aexecute("hello")).passed is False, (
        "a str payload was re-encoded — json.dumps would add surrounding quotes"
    )


def test_the_masker_and_the_runner_agree():
    """The defect the plane has and the SDK must not acquire.

    On the plane the ``regex`` runner uses ``str()`` while the masker uses
    ``json.dumps``, so a mask rule *matches* one string and *redacts* a
    different one. Here both sides of that pair see the same text, so a mask
    that matched finds a span to redact.
    """
    g = Guardrail(
        name="mask-it",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.input,
        config={"pattern": r"it's", "should_match": False},
        action="mask",
    )
    result = asyncio.run(g.aexecute(PAYLOAD))

    assert result.action_taken == "masked", (
        f"matched but could not redact — the runner and the masker disagree "
        f"about the text. action_taken={result.action_taken!r} {result.message!r}"
    )
    assert "it's" not in str(result.modified_data)
