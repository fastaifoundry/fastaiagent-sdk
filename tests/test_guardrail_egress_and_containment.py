"""Two guardrail defects found in the 2026-09-10 cross-repo audit.

Both survived a fully green board (683 tests across the two repos, zero skips),
because neither behaviour had a test.

**Egress.** ``apply_export_policy`` gated span *attributes* from the start, and
nothing gated span *events*. That mattered without anyone writing an event by
hand: OTel's ``record_exception`` fires automatically for an exception raised
inside a span, a blocked guardrail raises inside the ``agent.*`` span, and
``str(GuardrailBlockedError)`` **is** the guardrail's ``result.message`` — which
for an ``llm_judge`` rule is the judge's entire raw response and for
``groundedness`` quotes the model's answer. So payload-derived text left the
machine with ``FASTAIAGENT_TRACE_PAYLOADS=0`` set.

**Containment.** ``apply_action`` was called outside ``run_guardrail``'s
``try``, so anything it raised escaped as a bare ``AttributeError`` from
``agent.run()``: no ``errored`` flag, no ``on_error``, not even a
``GuardrailBlockedError``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.guardrail.implementations import run_guardrail
from fastaiagent.trace.redaction import (
    SENSITIVE_EVENT_ATTR_KEYS,
    RedactionPolicy,
    apply_event_export_policy,
    get_redaction_policy,
    set_redaction_policy,
)

#: A payload-derived guardrail message of the shape ``llm_judge`` really produces
#: (``message=response.content`` — the judge's whole reply, quoting the customer's
#: text back).
JUDGE_LEAK = (
    '{"verdict": "FAIL", "reason": "The answer states the patient\'s SSN is '
    '123-45-6789 and their card is 4111111111111111."}'
)


@pytest.fixture(autouse=True)
def _reset_policy():
    saved = get_redaction_policy()
    set_redaction_policy(None)
    yield
    set_redaction_policy(saved)


def _exception_event(message: str = JUDGE_LEAK) -> dict[str, Any]:
    """The event shape ``trace.storage`` writes for a recorded exception."""
    return {
        "name": "exception",
        "timestamp": "1757500000000000000",
        "attributes": {
            "exception.type": "GuardrailBlockedError",
            "exception.message": message,
            "exception.stacktrace": (
                f"Traceback (most recent call last):\n  ...\nGuardrailBlockedError: {message}\n"
            ),
        },
    }


# ---------------------------------------------------------------- egress


class TestEventEgressGate:
    def test_the_guardrail_message_does_not_reach_the_wire(self, monkeypatch):
        """The regression this whole fix exists for.

        The operator set the payload gate; the judge's quoted SSN must not leave.
        """
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        out = apply_event_export_policy([_exception_event()])

        serialized = json.dumps(out)
        assert "123-45-6789" not in serialized
        assert "4111111111111111" not in serialized
        assert "exception.message" not in out[0]["attributes"]

    def test_the_stacktrace_goes_too(self, monkeypatch):
        """Not redundant with the message key: a formatted traceback ends with
        ``<Type>: <message>``, so dropping only the message leaves the text."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        out = apply_event_export_policy([_exception_event()])

        assert "exception.stacktrace" not in out[0]["attributes"]

    def test_the_exception_type_survives(self, monkeypatch):
        """A class name is structural. An operator with payloads off should still
        be able to see *that* a guardrail blocked, and where."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        out = apply_event_export_policy([_exception_event()])

        assert out[0]["attributes"]["exception.type"] == "GuardrailBlockedError"
        assert out[0]["name"] == "exception"
        assert out[0]["timestamp"] == "1757500000000000000"

    def test_events_are_untouched_when_payloads_are_exported(self, monkeypatch):
        """The default. Payloads on and no policy is the zero-copy fast path."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")
        events = [_exception_event()]

        out = apply_event_export_policy(events)

        assert out is events
        assert out[0]["attributes"]["exception.message"] == JUDGE_LEAK

    def test_a_capture_policy_masks_the_message_instead_of_dropping_it(self, monkeypatch):
        """Payloads on + a redaction policy: the message survives, the SSN does not."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")
        set_redaction_policy(RedactionPolicy(patterns=[r"\d{3}-\d{2}-\d{4}"], mode="capture"))

        out = apply_event_export_policy([_exception_event()])

        msg = out[0]["attributes"]["exception.message"]
        assert "123-45-6789" not in msg
        assert "verdict" in msg  # the rest of the message is still there

    def test_a_read_only_policy_does_not_apply_on_egress(self, monkeypatch):
        """``read`` mode is the UI hook, not the export hook — mirrors
        ``apply_export_policy``'s own mode gating."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")
        set_redaction_policy(RedactionPolicy(patterns=[r"\d{3}-\d{2}-\d{4}"], mode="read"))

        out = apply_event_export_policy([_exception_event()])

        assert out[0]["attributes"]["exception.message"] == JUDGE_LEAK

    def test_the_input_is_never_mutated(self, monkeypatch):
        """local.db keeps full fidelity; only what leaves is filtered."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
        events = [_exception_event()]

        apply_event_export_policy(events)

        assert events[0]["attributes"]["exception.message"] == JUDGE_LEAK

    @pytest.mark.parametrize(
        "event",
        [
            "not-a-dict",
            {"name": "exception"},
            {"name": "exception", "attributes": None},
            {"name": "exception", "attributes": "not-a-dict"},
        ],
    )
    def test_malformed_events_pass_through_rather_than_raise(self, event, monkeypatch):
        """This runs on the export path — it must never be the reason a batch fails."""
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        assert apply_event_export_policy([event]) == [event]

    def test_non_exception_events_are_left_alone(self, monkeypatch):
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
        event = {"name": "retry", "timestamp": "1", "attributes": {"attempt": 2}}

        assert apply_event_export_policy([event])[0]["attributes"] == {"attempt": 2}

    def test_the_registry_names_both_message_carrying_keys(self):
        """Pinned so a later reader cannot narrow it to the message alone."""
        assert SENSITIVE_EVENT_ATTR_KEYS == frozenset({"exception.message", "exception.stacktrace"})


class TestPlatformExporterFiltersEvents:
    """Pins the call site behaviourally, not by grepping its source.

    ``platform_export`` used to run ``apply_export_policy`` over
    ``item["attributes"]`` and ship ``item["events"]`` verbatim.
    """

    @staticmethod
    def _span_with_a_blocked_guardrail():
        from fastaiagent.trace.storage import SpanData

        return SpanData(
            span_id="s1",
            trace_id="t1",
            parent_span_id=None,
            name="agent.support",
            span_type="agent",
            start_time="2026-09-10T00:00:00+00:00",
            end_time="2026-09-10T00:00:01+00:00",
            status="ERROR",
            attributes={"agent.output": "the customer's SSN is 123-45-6789"},
            events=[_exception_event()],
        )

    def test_neither_channel_carries_the_payload_when_the_gate_is_on(self, monkeypatch):
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
        from fastaiagent.trace.platform_export import to_wire

        item = to_wire(self._span_with_a_blocked_guardrail())

        serialized = json.dumps(item)
        assert "123-45-6789" not in serialized, (
            "the guardrail message reached the wire through the exception event"
        )
        assert "agent.output" not in item["attributes"]  # the pre-existing gate
        assert item["events"][0]["attributes"]["exception.type"] == "GuardrailBlockedError"

    def test_both_channels_flow_when_payloads_are_exported(self, monkeypatch):
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")
        from fastaiagent.trace.platform_export import to_wire

        item = to_wire(self._span_with_a_blocked_guardrail())

        assert item["attributes"]["agent.output"].endswith("123-45-6789")
        assert item["events"][0]["attributes"]["exception.message"] == JUDGE_LEAK


class TestOtelExporterFiltersEvents:
    def test_a_third_party_exporter_sees_no_exception_message(self, monkeypatch):
        """The same gap on the ``add_exporter`` path: ``_rebuild_span`` copied
        ``events`` by reference, so a Datadog/Jaeger exporter received the text."""
        pytest.importorskip("opentelemetry.sdk.trace")
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        from opentelemetry.sdk.trace import Event

        from fastaiagent.trace.otel import _filtered_events

        class _Span:
            events = [
                Event(
                    name="exception",
                    attributes={
                        "exception.type": "GuardrailBlockedError",
                        "exception.message": JUDGE_LEAK,
                    },
                    timestamp=1,
                )
            ]

        out = _filtered_events(_Span())

        assert out is not None, "events must be rebuilt when the gate is on"
        assert "exception.message" not in dict(out[0].attributes)
        assert dict(out[0].attributes)["exception.type"] == "GuardrailBlockedError"

    def test_no_rebuild_when_there_is_nothing_to_filter(self, monkeypatch):
        pytest.importorskip("opentelemetry.sdk.trace")
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")

        from fastaiagent.trace.otel import _filtered_events

        class _Span:
            events = []

        assert _filtered_events(_Span()) is None


# ------------------------------------------------------ action containment


def _malformed_mask_rule(on_error: str = "block") -> Guardrail:
    """A rule whose action fails while its check succeeds.

    ``secrets`` is the one maskable type whose runner never reads ``config``
    (deliberately — it takes no detection config), so a non-dict config reaches
    ``mask_payload`` without the runner raising first. Every other maskable type
    touches config in the runner and errors there instead.
    """
    return Guardrail(
        name="leaked_credentials",
        guardrail_type=GuardrailType.secrets,
        position=GuardrailPosition.output,
        config="not-a-dict",  # type: ignore[arg-type]
        action="mask",
        on_error=on_error,
    )


SECRET_PAYLOAD = 'api_key = "sk-proj-abcdefghijklmnopqrst"'


class TestActionFailureContainment:
    def test_an_action_failure_is_contained_not_raised(self):
        """Before the fix this escaped as ``AttributeError: 'str' object has no
        attribute 'get'`` straight out of ``run_guardrail``."""
        result = asyncio.run(run_guardrail(_malformed_mask_rule(), SECRET_PAYLOAD))

        assert result.passed is False
        assert result.errored is True
        assert result.action_taken == "blocked"
        assert "could not apply action" in (result.message or "")

    def test_on_error_allow_does_not_rescue_an_action_failure(self):
        """The load-bearing asymmetry.

        ``on_error`` answers "what does it mean when the check could not RUN".
        Here the check ran and said FAIL — only the consequence could not be
        applied. A mask that raised is strictly worse than a mask that found
        nothing, and that already degrades to a block.
        """
        result = asyncio.run(run_guardrail(_malformed_mask_rule("allow"), SECRET_PAYLOAD))

        assert result.passed is False, "an action failure must not fail open"
        assert result.action_taken == "blocked"

    def test_the_verdict_that_preceded_the_failure_is_recorded(self):
        result = asyncio.run(run_guardrail(_malformed_mask_rule(), SECRET_PAYLOAD))

        assert result.metadata["verdict_before_action"] is False
        assert "'str' object has no attribute 'get'" in result.metadata["action_error"]

    def test_a_runner_failure_still_honours_on_error(self):
        """Regression guard: the *other* branch is unchanged. A check that could
        not run still gets the operator's fail policy."""

        def boom(_text: str) -> bool:
            raise RuntimeError("detector unavailable")

        for on_error, expected_passed in (("allow", True), ("block", False)):
            rule = Guardrail(
                name="detector",
                guardrail_type=GuardrailType.code,
                position=GuardrailPosition.output,
                fn=boom,
                on_error=on_error,  # type: ignore[arg-type]
            )
            result = asyncio.run(run_guardrail(rule, "anything"))
            assert result.passed is expected_passed
            assert result.errored is True

    def test_the_block_propagates_as_a_guardrail_error_not_a_type_error(self):
        """End to end through the executor: the caller sees the documented
        exception type, not whatever the mask happened to raise."""
        from fastaiagent.guardrail.executor import execute_guardrails

        with pytest.raises(GuardrailBlockedError):
            asyncio.run(
                execute_guardrails(
                    [_malformed_mask_rule()], SECRET_PAYLOAD, GuardrailPosition.output
                )
            )


class TestEventLoggingIsBestEffort:
    def test_a_broken_event_store_does_not_fail_the_check(self, monkeypatch, tmp_path):
        """``log_guardrail_event`` is ``try``/*finally* with no ``except``, so a
        locked or unwritable ``local.db`` used to abort the agent run — turning an
        observability problem into an outage."""
        from fastaiagent._internal.config import get_config

        config = get_config()
        monkeypatch.setattr(config, "ui_enabled", True, raising=False)
        # A directory where a file is expected: init_local_db cannot open it.
        unwritable = tmp_path / "blocked.db"
        unwritable.mkdir()
        monkeypatch.setattr(config, "local_db_path", str(unwritable), raising=False)

        rule = Guardrail(
            name="ok",
            guardrail_type=GuardrailType.regex,
            position=GuardrailPosition.output,
            config={"pattern": "nope"},
        )

        result = rule.execute("clean text")

        assert result.passed is True, "a broken event store must not change the verdict"

    def test_unserializable_metadata_does_not_cost_the_event(self, tmp_path):
        """A ``code`` guardrail may return anything in ``metadata``. Without
        ``default=str`` the insert raised; now that the caller catches, that would
        silently lose the whole row instead of one value."""
        from fastaiagent.ui.events import log_guardrail_event

        class Unserializable:
            def __repr__(self) -> str:
                return "<opaque>"

        rule = Guardrail(
            name="code_rule",
            guardrail_type=GuardrailType.code,
            position=GuardrailPosition.output,
        )
        result = GuardrailResult(passed=True, metadata={"obj": Unserializable()})

        # Must not raise.
        log_guardrail_event(
            rule, result, data="x", db_path=str(tmp_path / "local.db"), agent_name="a"
        )


class TestSpanStatusIsFilteredToo:
    """The third content channel, and the last one.

    A span's ``Status`` carries a free-text description, and for a guardrail span
    that description **is** the rule's failure message. ``_rebuild_span`` copied
    ``status`` by reference, so filtering attributes and events still let that text
    reach a Datadog or Jaeger exporter with the payload gate on.
    """

    @staticmethod
    def _errored_span():
        from opentelemetry.trace import Status, StatusCode

        class _Span:
            status = Status(StatusCode.ERROR, JUDGE_LEAK)

        return _Span()

    def test_the_description_is_withheld_when_the_gate_is_on(self, monkeypatch):
        pytest.importorskip("opentelemetry.sdk.trace")
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        from opentelemetry.trace import StatusCode

        from fastaiagent.trace.otel import _filtered_status

        out = _filtered_status(self._errored_span())

        assert out is not None, "the status must be rebuilt when the gate is on"
        assert out.description is None
        assert out.status_code is StatusCode.ERROR, "the code is structural and must survive"

    def test_a_capture_policy_masks_the_description(self, monkeypatch):
        pytest.importorskip("opentelemetry.sdk.trace")
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")
        set_redaction_policy(RedactionPolicy(patterns=[r"\d{3}-\d{2}-\d{4}"], mode="capture"))

        from fastaiagent.trace.otel import _filtered_status

        out = _filtered_status(self._errored_span())

        assert out is not None
        assert "123-45-6789" not in (out.description or "")

    def test_untouched_when_payloads_are_exported(self, monkeypatch):
        pytest.importorskip("opentelemetry.sdk.trace")
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "1")

        from fastaiagent.trace.otel import _filtered_status

        assert _filtered_status(self._errored_span()) is None

    def test_a_span_with_no_description_needs_no_rebuild(self, monkeypatch):
        pytest.importorskip("opentelemetry.sdk.trace")
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        from opentelemetry.trace import Status, StatusCode

        from fastaiagent.trace.otel import _filtered_status

        class _Span:
            status = Status(StatusCode.OK)

        assert _filtered_status(_Span()) is None
