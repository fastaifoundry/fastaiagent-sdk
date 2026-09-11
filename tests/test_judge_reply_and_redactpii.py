"""Two second copies that drifted from the thing they were copied from.

Both are the same defect shape, which is why they ship together.

**8b — the judge's reply travelled in an error string.**
``hazard_taxonomy.parse_scores`` and ``grounding.parse_verdict`` interpolated up
to 200 characters of the model's reply into their ``ValueError``. That becomes
``GuardrailResult.error``, and the control plane copies it into
``guardrail_executions.result_detail`` — durable, tenant-visible, aggregated by
``GET /guardrails/analytics``. A judge that refuses in prose quoting the payload,
or one prompt-injected into echoing its input, wrote customer content there
permanently. It is the rule ``detectors.py`` states for that exact table, broken
from the other end: the control that inspects sensitive data must not become a
place that stores it.

``topics.py`` never did this. Two of three judges disagreed with the third and
the third was right. The plane fixed its copies first (``2b66686``); these two
modules are **theirs** and the SDK mirrors them, so this is a re-mirror.

**9b — ``RedactPII`` carried its own PII regexes.**
Its card pattern was a bare ``\\b(?:\\d[ \\-]?){13,19}\\b`` with no Luhn check,
so any 13-19 digit run was redacted as a card. ``before_model`` mutates message
content in place, so that corruption reached the model, memory, and the
guardrail re-ask history. ``detect_pii`` — already the detector behind
``no_pii()``, the ``pii`` type and ``PIILeakage`` — Luhn-validates, and was one
import away.
"""

from __future__ import annotations

import asyncio

import pytest

from fastaiagent.agent.middleware import RedactPII
from fastaiagent.guardrail import grounding, hazard_taxonomy

#: A judge reply that quotes the customer's payload back. Both shapes below are
#: real: a model that refuses in prose, and one prompt-injected into echoing.
POISONED_REPLY = (
    "I can't help with that request about the patient whose SSN is 123-45-6789 "
    "and whose card is 4111111111111111."
)
SECRETS_IN_REPLY = ("123-45-6789", "4111111111111111", "patient")


# --------------------------------------------------------------- 8b


class TestTheJudgesReplyDoesNotTravel:
    def test_content_safety_no_json_object(self):
        with pytest.raises(ValueError) as exc:
            hazard_taxonomy.parse_scores(POISONED_REPLY, ["S1"])
        for secret in SECRETS_IN_REPLY:
            assert secret not in str(exc.value)

    def test_content_safety_unparseable_json(self):
        """A reply that *looks* like JSON but is not. The old code put the raw
        text in; the new one reports the parser's own complaint."""
        with pytest.raises(ValueError) as exc:
            hazard_taxonomy.parse_scores(
                '{"S1": 0.9, ssn: 123-45-6789, card: 4111111111111111}', ["S1"]
            )
        assert "123-45-6789" not in str(exc.value)
        assert "4111111111111111" not in str(exc.value)

    def test_groundedness_no_json_object(self):
        with pytest.raises(ValueError) as exc:
            grounding.parse_verdict(POISONED_REPLY)
        for secret in SECRETS_IN_REPLY:
            assert secret not in str(exc.value)

    def test_groundedness_unparseable_json(self):
        with pytest.raises(ValueError) as exc:
            grounding.parse_verdict('{"score": 0.2, unsupported: 123-45-6789}')
        assert "123-45-6789" not in str(exc.value)

    @pytest.mark.parametrize(
        ("fn", "arg", "expected"),
        [
            (
                hazard_taxonomy.parse_scores,
                POISONED_REPLY,
                "content-safety judge returned no JSON object",
            ),
            (grounding.parse_verdict, POISONED_REPLY, "groundedness judge returned no JSON object"),
        ],
    )
    def test_the_wording_matches_the_plane_verbatim(self, fn, arg, expected):
        """The plane owns these two modules; the SDK mirrors them.

        Pinned as literals because the point is cross-repo agreement, and the SDK
        cannot import the plane to compare. The plane's own
        ``test_judge_mirror.py`` asserts the same strings from its side, gated on
        SDK >= 1.63.0.
        """
        with pytest.raises(ValueError) as exc:
            fn(arg) if fn is grounding.parse_verdict else fn(arg, ["S1"])
        assert str(exc.value) == expected

    def test_the_row_still_records_why_the_check_could_not_run(self):
        """The half that makes this a redaction and not a blind spot.

        ``errored`` is derived from the presence of an error, and an operator
        still has to be able to tell "the judge returned prose" from "the judge
        returned malformed JSON". Dropping the reply must not drop the reason.
        """
        with pytest.raises(ValueError) as no_json:
            hazard_taxonomy.parse_scores(POISONED_REPLY, ["S1"])
        with pytest.raises(ValueError) as bad_json:
            hazard_taxonomy.parse_scores('{"S1": bad}', ["S1"])

        assert str(no_json.value) != str(bad_json.value), (
            "the two failures must stay distinguishable"
        )
        assert "no JSON object" in str(no_json.value)
        assert "unparseable JSON" in str(bad_json.value)
        assert str(bad_json.value) != "content-safety judge returned unparseable JSON: "

    def test_topic_was_always_correct(self):
        """The control case — it is what made the fix obvious rather than a
        judgement call. Pinned so a later 'consistency' pass cannot break it."""
        from fastaiagent.guardrail import topics

        with pytest.raises(ValueError) as exc:
            topics.parse_topics(POISONED_REPLY, [{"name": "medical", "description": "d"}])
        for secret in SECRETS_IN_REPLY:
            assert secret not in str(exc.value)


# --------------------------------------------------------------- 9b


class TestRedactPIIUsesTheSharedDetector:
    def test_an_order_number_survives_and_a_real_card_does_not(self):
        """The whole point of the Luhn check, in one assertion.

        Both are 16 digits. Only one is a card.
        """
        out = RedactPII()._redact("Order 1234567890123456 shipped; charged card 4111111111111111.")

        assert "1234567890123456" in out, "an order number is not a credit card"
        assert "4111111111111111" not in out, "a Luhn-valid card must still be redacted"

    @pytest.mark.parametrize(
        "text",
        [
            "reach me at bob@acme.com",
            "ssn 123-45-6789",
            "call 555-123-4567",
        ],
    )
    def test_the_entities_it_always_caught_are_still_caught(self, text):
        assert "[REDACTED]" in RedactPII()._redact(text)

    def test_it_shares_the_detector_rather_than_copying_it(self):
        """Behavioural, not a source grep: the middleware and the guardrail
        detector must agree about what counts as PII."""
        from fastaiagent._internal.safety_detectors import DEFAULT_PII_ENTITIES, detect_pii

        text = "Order 1234567890123456 for bob@acme.com, card 4111111111111111, ssn 123-45-6789."
        detector_found = {m.value for m in detect_pii(text, entities=list(DEFAULT_PII_ENTITIES))}
        redacted = RedactPII()._redact(text)

        for value in detector_found:
            assert value not in redacted, f"detector found {value!r}; middleware left it in"
        assert "1234567890123456" in redacted, "and neither should redact the order number"

    def test_custom_patterns_are_untouched(self):
        """``patterns=`` keeps its old meaning — your regexes, applied verbatim,
        with no Luhn opinion. Only the default path moved."""
        m = RedactPII(patterns=[r"\b\d{4}\b"])
        assert m._redact("code 1234 here") == "code [REDACTED] here"
        # the shared detector is NOT consulted on this path
        assert m._redact("bob@acme.com") == "bob@acme.com"

    def test_a_custom_placeholder_is_honoured_on_both_paths(self):
        assert "***" in RedactPII(placeholder="***")._redact("ssn 123-45-6789")
        assert "***" in RedactPII(patterns=[r"x"], placeholder="***")._redact("x")

    def test_entities_can_be_narrowed(self):
        m = RedactPII(entities=("email",))
        out = m._redact("bob@acme.com and ssn 123-45-6789")
        assert "bob@acme.com" not in out
        assert "123-45-6789" in out, "only the named entity should be redacted"

    @pytest.mark.parametrize("content", [[{"type": "image"}], None, "", 42])
    def test_non_string_content_does_not_raise(self, content):
        """A multimodal message's ``content`` is a ``list[ContentPart]``. The old
        ``pat.sub`` raised TypeError on it — a middleware crashing the run it was
        added to protect."""
        assert RedactPII()._redact(content) == content

    def test_it_still_redacts_in_place_through_before_model(self):
        """The integration that matters: ``before_model`` mutates the shared
        message objects, which is how the corruption reached memory and the
        re-ask history in the first place."""
        from fastaiagent.llm.message import Message, MessageRole

        messages = [Message(role=MessageRole.user, content="ssn 123-45-6789")]
        out = asyncio.run(RedactPII().before_model(None, messages))

        assert "123-45-6789" not in out[0].content
        assert out[0] is messages[0], "mutation is in place, by design"

    def test_overlapping_matches_are_replaced_once(self):
        """``mask_spans`` merges overlaps and edits right-to-left, so a value
        matched twice is redacted cleanly instead of leaving fragments."""
        out = RedactPII()._redact("card 4111111111111111 card 4111111111111111")
        assert "4111" not in out
        assert out.count("[REDACTED]") == 2
