"""The remaining SDK-owned findings from the 2026-09-10 audit that needed no sign-off.

Six defects, one theme: each is a rule the code *states* somewhere — in a
docstring, a changelog, a comment, or a sibling function — and does not hold.

* ``run_sync`` dropped the caller's ``contextvars`` context, so a shipped feature
  (``fa.guardrail_context`` → ``groundedness``) worked in a script and blocked the
  run from Jupyter, pytest-asyncio, or a framework proxy.
* ``require_platform`` had no ``E2E_REQUIRED`` escape hatch, so the only
  wire-level test of the guardrail feature skipped on every PR *by configuration*.
* ``mask_payload`` kept a second copy of the pii backend resolution, which 1.61.0
  claimed to have removed.
* ``unsupported_claims`` was capped by count but not by volume, while the export
  allowlist's argument for admitting it says "five clipped claims".
* A second ``reask`` rule fell through to a hard block.
* Guardrail event metadata was the one local write a ``RedactionPolicy`` could not
  reach.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import fastaiagent as fa
from fastaiagent.guardrail.executor import _exportable_detail, execute_guardrails
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.trace.redaction import RedactionPolicy, get_redaction_policy, set_redaction_policy


@pytest.fixture(autouse=True)
def _reset_policy():
    saved = get_redaction_policy()
    set_redaction_policy(None)
    yield
    set_redaction_policy(saved)


class TestRunSyncCarriesTheCallersContext:
    """The bug that only appears when someone else owns the event loop."""

    def test_context_survives_the_thread_offload(self):
        from fastaiagent._internal.async_utils import run_sync
        from fastaiagent.guardrail.context import get_guardrail_context

        async def from_inside_a_running_loop():
            with fa.guardrail_context(context="the retrieved docs"):

                async def probe():
                    return get_guardrail_context()

                return run_sync(probe())

        assert asyncio.run(from_inside_a_running_loop()) == {"context": "the retrieved docs"}

    def test_the_two_paths_agree(self):
        """A plain script always worked. That asymmetry is what hid this."""
        from fastaiagent._internal.async_utils import run_sync
        from fastaiagent.guardrail.context import get_guardrail_context

        async def probe():
            return get_guardrail_context()

        with fa.guardrail_context(context="docs", answer="a"):
            plain = run_sync(probe())

        async def nested():
            with fa.guardrail_context(context="docs", answer="a"):
                return run_sync(probe())

        assert plain == asyncio.run(nested())

    def test_a_groundedness_rule_can_find_its_context_from_a_running_loop(self):
        """The user-visible consequence: without this the rule raises, and with the
        default ``on_error="block"`` it blocks the run it was meant to score."""
        from fastaiagent.guardrail import grounding

        async def nested():
            with fa.guardrail_context(context="Paris is the capital of France."):
                from fastaiagent._internal.async_utils import run_sync

                async def extract():
                    return grounding.extract_pair({}, "Paris is the capital.")

                return run_sync(extract())

        context, answer = asyncio.run(nested())
        assert context == "Paris is the capital of France."
        assert answer == "Paris is the capital."


class TestTheE2EGateCanBeDemanded:
    """`require_env` fails when `E2E_REQUIRED=1`; `require_platform` only skipped."""

    def test_e2e_required_turns_the_platform_skip_into_a_failure(self, monkeypatch):
        from tests.e2e.conftest import require_platform

        monkeypatch.setenv("E2E_SKIP_PLATFORM", "1")
        monkeypatch.setenv("E2E_REQUIRED", "1")

        # Catch BaseException, not fail.Exception: pre-fix this raised *Skipped*,
        # and a narrow `raises` would let that propagate and skip this test — which
        # is the exact failure mode under test.
        with pytest.raises(BaseException) as exc:
            require_platform()
        assert isinstance(exc.value, pytest.fail.Exception), (
            f"E2E_REQUIRED=1 must turn the platform skip into a failure; "
            f"got {type(exc.value).__name__} instead"
        )
        assert "E2E_REQUIRED" in str(exc.value)

    def test_it_still_skips_by_default(self, monkeypatch):
        from tests.e2e.conftest import require_platform

        monkeypatch.setenv("E2E_SKIP_PLATFORM", "1")
        monkeypatch.delenv("E2E_REQUIRED", raising=False)

        with pytest.raises(pytest.skip.Exception) as exc:
            require_platform()
        assert "E2E_SKIP_PLATFORM" in str(exc.value)

    def test_it_is_a_no_op_when_the_platform_is_in_play(self, monkeypatch):
        from tests.e2e.conftest import require_platform

        monkeypatch.delenv("E2E_SKIP_PLATFORM", raising=False)
        require_platform()  # must not raise


class TestMaskPayloadUsesTheSharedBackendResolver:
    def test_there_is_only_one_backend_resolution(self):
        """1.61.0's changelog: *"one place to be wrong rather than two"*. It was
        true of ``_run_pii`` and false of ``mask_payload``."""
        import inspect

        from fastaiagent.guardrail import actions

        source = inspect.getsource(actions.mask_payload)
        assert "_resolve_pii_backend" in source
        assert 'config.get("backend")' not in source, (
            "mask_payload must resolve the backend through the shared resolver, "
            "not with its own inline copy"
        )

    def test_an_unknown_backend_raises_the_mirrored_error(self):
        """The observable difference: the plane-mirrored wording, not
        ``detect_pii``'s own."""
        from fastaiagent.guardrail.actions import mask_payload

        rule = Guardrail(
            name="pii-mask",
            guardrail_type=GuardrailType.pii,
            position=GuardrailPosition.output,
            config={"backend": "presidoo"},
            action="mask",
        )

        with pytest.raises(ValueError, match="pii guardrail backend must be one of"):
            asyncio.run(mask_payload(rule, "bob@acme.com"))


class TestUnsupportedClaimsAreCappedByVolumeToo:
    def test_a_single_enormous_claim_is_clipped_on_export(self):
        """The allowlist admits this key on the argument that it is *five clipped
        claims*. The count was capped; the volume was not, so one entry could carry
        the whole answer."""
        rule = Guardrail(
            name="grounded",
            guardrail_type=GuardrailType.groundedness,
            position=GuardrailPosition.output,
        )
        whole_answer = "x" * 5000
        result = GuardrailResult(
            passed=False,
            metadata={"score": 0.1, "threshold": 0.7, "unsupported_claims": [whole_answer]},
        )

        from fastaiagent.guardrail.executor import _MAX_CLAIM_CHARS

        detail = _exportable_detail(rule, result)

        assert detail is not None
        assert len(detail["unsupported_claims"][0]) == _MAX_CLAIM_CHARS

    def test_the_local_result_keeps_full_fidelity(self):
        """Only what *leaves* is clipped — local capture is untouched."""
        result = GuardrailResult(passed=False, metadata={"unsupported_claims": ["y" * 5000]})
        assert len(result.metadata["unsupported_claims"][0]) == 5000

    def test_short_claims_are_untouched(self):
        rule = Guardrail(
            name="grounded",
            guardrail_type=GuardrailType.groundedness,
            position=GuardrailPosition.output,
        )
        result = GuardrailResult(
            passed=False, metadata={"unsupported_claims": ["the sky is green"]}
        )

        detail = _exportable_detail(rule, result)

        assert detail is not None
        assert detail["unsupported_claims"] == ["the sky is green"]


class TestASecondReaskRuleDoesNotHardBlock:
    @staticmethod
    def _reask_rule(name: str, position: GuardrailPosition = GuardrailPosition.output) -> Guardrail:
        return Guardrail(
            name=name,
            guardrail_type=GuardrailType.regex,
            position=position,
            config={"pattern": "bad"},
            action="reask",
        )

    def test_two_failing_reask_rules_still_hand_back_a_reask(self):
        """Before the fix the second rule skipped the reask branch entirely, fell
        through to ``halts()`` — True for ``reask`` on a blocking rule — and raised,
        silently converting the pair into a hard block."""
        outcome = asyncio.run(
            execute_guardrails(
                [self._reask_rule("first"), self._reask_rule("second")],
                "this is bad",
                GuardrailPosition.output,
                allow_reask=True,
            )
        )

        assert outcome.reask is not None
        assert outcome.reask.action_taken == "reask"

    def test_the_first_failure_is_the_one_handed_back(self):
        """First-wins on *which* failure, because the caller can only re-drive the
        model once per attempt."""
        outcome = asyncio.run(
            execute_guardrails(
                [self._reask_rule("first"), self._reask_rule("second")],
                "this is bad",
                GuardrailPosition.output,
                allow_reask=True,
            )
        )

        assert outcome.reask is not None
        assert outcome.results[0].action_taken == "reask"

    def test_a_single_reask_rule_is_unchanged(self):
        outcome = asyncio.run(
            execute_guardrails(
                [self._reask_rule("only")],
                "this is bad",
                GuardrailPosition.output,
                allow_reask=True,
            )
        )
        assert outcome.reask is not None

    def test_reask_still_blocks_where_the_caller_cannot_retry(self):
        """`allow_reask=False` is every position but the agent's output path."""
        from fastaiagent._internal.errors import GuardrailBlockedError

        with pytest.raises(GuardrailBlockedError):
            asyncio.run(
                execute_guardrails(
                    [
                        self._reask_rule("first", GuardrailPosition.input),
                        self._reask_rule("second", GuardrailPosition.input),
                    ],
                    "this is bad",
                    GuardrailPosition.input,
                )
            )


class TestGuardrailEventMetadataHonoursTheRedactionPolicy:
    @staticmethod
    def _write_event(tmp_path, monkeypatch, *, data: str) -> list[dict]:
        """Write one `filtered` event and read the stored metadata back.

        ``ui_enabled`` has to be forced on: it is off by default, and
        ``log_guardrail_event`` returns early without it — which would make every
        assertion below pass over an empty table.
        """
        from fastaiagent._internal.config import get_config
        from fastaiagent.ui.db import init_local_db
        from fastaiagent.ui.events import log_guardrail_event

        monkeypatch.setattr(get_config(), "ui_enabled", True, raising=False)
        db = str(tmp_path / "local.db")

        rule = Guardrail(
            name="mask_pii",
            guardrail_type=GuardrailType.pii,
            position=GuardrailPosition.output,
            action="mask",
        )
        result = GuardrailResult(
            passed=False,
            action="mask",
            action_taken="masked",
            modified_data="the ssn is [REDACTED]",
        )
        log_guardrail_event(rule, result, data=data, db_path=db, agent_name="a")

        helper = init_local_db(db)
        try:
            rows = helper.fetchall("SELECT metadata FROM guardrail_events")
        finally:
            helper.close()

        stored = [json.loads(r["metadata"]) for r in rows]
        assert stored, "no event was written — the assertions below would pass vacuously"
        assert "before" in stored[0], "the before/after diff is the thing under test"
        return stored

    def test_the_before_half_of_a_mask_diff_is_redacted(self, tmp_path, monkeypatch):
        """`before` is the payload *prior* to redaction — the very PII the rule
        exists to remove — and was the one local write a policy could not reach.
        ``trace.storage`` has redacted span attributes on capture all along."""
        set_redaction_policy(RedactionPolicy(patterns=[r"\d{3}-\d{2}-\d{4}"], mode="capture"))

        stored = self._write_event(tmp_path, monkeypatch, data="the ssn is 123-45-6789")

        assert "123-45-6789" not in json.dumps(stored), (
            "the policy must reach guardrail event metadata"
        )
        assert "the ssn is" in stored[0]["before"], "only the match is masked, not the row"

    def test_no_policy_means_full_fidelity(self, tmp_path, monkeypatch):
        """Local capture is full fidelity by design — Replay depends on it, and the
        payload flag is an *export* boundary. Without a policy nothing changes."""
        stored = self._write_event(tmp_path, monkeypatch, data="ssn 123-45-6789")

        assert "123-45-6789" in json.dumps(stored)
