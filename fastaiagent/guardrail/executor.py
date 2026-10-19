"""Guardrail execution — blocking and parallel modes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.actions import degrade_to_block, effect_of, halts
from fastaiagent.guardrail.guardrail import (
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
    record_firing,
)

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail
    from fastaiagent.ui.events import EventBatch, GuardrailEventRecord

_log = logging.getLogger(__name__)


@dataclass
class GuardrailOutcome:
    """What a position's guardrails decided, and the payload to carry forward.

    ``mask`` and ``override`` rewrite the payload, so this can no longer be a
    function that either returns verdicts or raises — it has to be able to hand
    back a **modified payload**. :attr:`data` is what the caller should use from
    here on: the original when nothing rewrote it, the rewritten value when
    something did.

    Iterates and indexes as the ``list[GuardrailResult]`` this used to be, so
    callers that only wanted the verdicts keep working unchanged.
    """

    results: list[GuardrailResult] = field(default_factory=list)
    data: str | dict[str, Any] = ""
    modified: bool = False
    reask: GuardrailResult | None = None
    """The first rule that asked to re-prompt the model, if any. Only the agent's
    output path can honour it — every other position has no model turn to redo,
    so ``halts()`` blocks there instead."""
    records: list[GuardrailEventRecord] = field(default_factory=list)
    """One record per check run here, with its effect and local event id (1.89.0)."""

    def refuse(self, reason: str) -> None:
        """The caller cannot use this outcome's rewrite or re-ask, and will block.

        Call immediately before raising. Every rewritten or re-asked check is
        recorded as ``blocked`` with ``reason``: its result (so
        ``GuardrailBlockedError.results`` tells the truth), its firing on
        ``AgentResult.guardrails``, and its local event row. The guardrail span
        already emitted is unchanged. Best-effort: never raises.
        """
        try:
            event_ids: list[str] = []
            for record in self.records:
                if record.effect not in ("rewritten", "reasked"):
                    continue
                record.effect = "blocked"
                record.outcome = "blocked"
                if record.result is not None:
                    degrade_to_block(record.result, reason)
                if record.firing is not None:
                    record.firing.effect = "blocked"
                if record.event_id:
                    event_ids.append(record.event_id)
            if event_ids:
                from fastaiagent.ui.events import finalize_guardrail_events

                finalize_guardrail_events(
                    event_ids, effect="blocked", outcome="blocked", final_reason=reason
                )
        except Exception:
            _log.debug("Could not record a refused guardrail outcome", exc_info=True)

    def __iter__(self) -> Iterator[GuardrailResult]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> GuardrailResult:
        return self.results[index]

    def __bool__(self) -> bool:
        return True


#: Per-type allowlist of ``GuardrailResult.metadata`` keys the SDK runtime sends
#: to a control plane on the guardrail span.
#:
#: A check's metadata is captured locally at full fidelity, but most of it is
#: derived from the payload — ``toxic_words`` holds the offending words,
#: ``matches`` a regex fragment (for a PII rule, the matched value itself),
#: ``unsupported_claims`` model output over customer content. None of that may
#: leave the machine as a side effect of reporting a verdict, so this is an
#: allowlist rather than a filter: a type absent from it exports nothing.
#:
#: Most entries here are payload-free by construction. ``topic``'s ``mode`` is an
#: enum, its ``topics`` is the rule's own config — authored centrally, so the
#: plane already has it — and its ``matched`` is intersected back against
#: ``topics`` by ``topics.parse_topics``, so a model cannot smuggle content into
#: it. ``content_safety`` and ``groundedness`` export scores, the bars they were
#: judged against, and category codes: derived numbers and rule config, no text.
#:
#: ⚠ ``groundedness.unsupported_claims`` is the exception — it quotes the
#: answer, so it **is** payload-derived, and it is here deliberately. The
#: judgement (agreed with the plane, 2026-09-10): the payload gate is the right
#: control for it rather than exclusion. ``FASTAIAGENT_TRACE_PAYLOADS=0`` already
#: means "no customer content leaves"; a deployment with payloads *on* has
#: consented to span inputs and outputs, which is strictly more content than five
#: clipped claims — and ``grounding.parse_verdict`` caps the list at five.
#: Excluding it would leave an operator able to see the claims at
#: ``/guardrails/{id}/test`` but not for the run that actually failed, which is
#: the one worth debugging.
#:
#: **Do not move ``unsupported_claims`` out from behind the payload gate on the
#: grounds that its neighbours are safe.** They are safe; it is not. The whole
#: ``detail`` attribute is registered in
#: :data:`~fastaiagent.trace.redaction.SENSITIVE_ATTR_KEYS`, and that
#: registration is what makes this entry defensible.
#:
#: Widening this is a deliberate act: ``tests/test_guardrail_topics.py`` pins the
#: contents so a new entry has to be argued for, not typed.
EXPORTABLE_DETAIL_KEYS: dict[GuardrailType, frozenset[str]] = {
    # ``backend`` (1.84.0) is the engine name — ``"decisions"`` — and is present
    # only off the default chat engine; rule-derived, like ``pii``'s.
    GuardrailType.topic: frozenset({"mode", "matched", "topics", "backend"}),
    GuardrailType.content_safety: frozenset(
        {"taxonomy", "scores", "thresholds", "tripped", "unscored", "backend"}
    ),
    # Only the Decisions-API engine writes these (a probability and the rule's
    # own threshold); the chat engine's ``llm_judge`` result carries no metadata.
    GuardrailType.llm_judge: frozenset({"backend", "probability", "threshold"}),
    # ``unsupported_claims`` is payload-derived — see the ⚠ above.
    GuardrailType.groundedness: frozenset({"score", "threshold", "unsupported_claims"}),
    # Entity detection. Counts and entity names only — the same shape the plane's
    # ``detectors.summarize_pii`` / ``summarize_secrets`` persist, and for the
    # same reason: the row is durable and tenant-visible, so the control that
    # finds personal data must not become a standing database of it. Note what is
    # deliberately absent — ``PIIMatch.value`` (the matched text), the offsets,
    # and even ``SecretMatch.masked``.
    GuardrailType.pii: frozenset({"backend", "entities", "found", "counts", "total"}),
    GuardrailType.secrets: frozenset({"found", "counts", "total"}),
}


#: Per-entry character cap applied to ``groundedness.unsupported_claims`` on the
#: way out. The list is capped at five by ``grounding.parse_verdict``, and the ⚠
#: note above rests on that — but a *count* cap leaves the volume unbounded, so a
#: judge (or an injected one) could return the whole answer, or the whole
#: retrieved context, as a single "claim". Long enough to identify the claim,
#: short enough that it cannot become a channel for the payload.
#:
#: Applied **here**, not in ``grounding.py``: that module is mirrored from the
#: plane and clipping there would widen a cross-repo divergence to fix an egress
#: concern. The bound is claimed by this allowlist, so it is enforced by this
#: allowlist — the local result keeps full fidelity either way.
_MAX_CLAIM_CHARS = 300


def _exportable_detail(guardrail: Guardrail, result: GuardrailResult) -> dict[str, object] | None:
    """The subset of ``result.metadata`` this guardrail may put on its span."""
    allowed = EXPORTABLE_DETAIL_KEYS.get(guardrail.guardrail_type)
    if not allowed or not result.metadata:
        return None
    detail = {k: v for k, v in result.metadata.items() if k in allowed}
    claims = detail.get("unsupported_claims")
    if isinstance(claims, list):
        detail["unsupported_claims"] = [str(c)[:_MAX_CLAIM_CHARS] for c in claims]
    return detail or None


def _stamp_span(span: Any, guardrail: Guardrail, result: GuardrailResult) -> None:
    """Set a guardrail span's outcome attributes and status. Best-effort."""
    try:
        from fastaiagent.trace.span import _stamp_guardrail_outcome

        if result.errored:
            check_result = "error"
        elif result.passed:
            check_result = "pass"
        else:
            check_result = "block"
        _stamp_guardrail_outcome(
            span,
            name=guardrail.name,
            position=guardrail.position.value,
            passed=result.passed,
            checks=json.dumps([{"name": guardrail.name, "result": check_result}]),
            errored=result.errored,
            message=result.message,
            action=result.action,
            action_taken=result.action_taken,
            severity=guardrail.severity,
            floor=guardrail.floor,
            detail=_exportable_detail(guardrail, result),
        )
    except Exception:  # pragma: no cover - observability must never break a run
        _log.debug("Failed to stamp guardrail span for %r", guardrail.name, exc_info=True)


async def _unrunnable(
    guardrail: Guardrail, data: str | dict[str, Any], reason: str
) -> GuardrailResult:
    """The result of a check that could not run, decided by ``on_error`` — the
    same verdict ``run_guardrail`` gives a check whose runner raised (CLAUDE.md
    §2.4: never a clean pass)."""
    from fastaiagent.guardrail.actions import apply_action

    result = GuardrailResult(
        passed=guardrail.on_error == "allow",
        errored=True,
        message=f"{guardrail.name} could not run (on_error={guardrail.on_error}): {reason}",
        metadata={"error": reason, "on_error": guardrail.on_error},
    )
    return await apply_action(guardrail, data, result)


async def _check_in_span(
    guardrail: Guardrail,
    data: str | dict[str, Any],
    *,
    contain: bool = False,
    could_not_run: str | None = None,
) -> tuple[GuardrailResult, str | None]:
    """Run one check inside its own guardrail span; return ``(result, span_id)``.

    Emitted on **pass and block** so the console shows green passes too. The
    span nests under whatever span is currently active (the agent/turn span),
    since OTel context propagates through ``await``.

    Since 1.89.0 the span **wraps the check** (it used to be opened after it):
    its duration is the check's real latency, and a model call the check makes
    (a judge's) is its child, not its sibling. Keys, values and status are
    unchanged, and set by the same helper as :func:`fastaiagent.emit_guardrail`,
    which a foreign runtime that borrows :func:`fastaiagent.run_guardrail` calls
    with *its* tracer instead.

    ``contain`` turns an exception from the check into an errored, blocked
    result (the observe-only path, which used to get it from ``gather``).
    """
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import _span_id_hex

    async def check() -> GuardrailResult:
        if could_not_run is not None:
            return await _unrunnable(guardrail, data, could_not_run)
        try:
            return await guardrail._acheck(data)
        except Exception as e:
            if not contain:
                raise
            return GuardrailResult(
                passed=False, message=str(e), errored=True, action_taken="blocked"
            )

    # Observability must never break the check: if the span cannot be opened,
    # the check still runs, untraced.
    try:
        cm = get_tracer("fastaiagent.guardrail").start_as_current_span(
            f"guardrail.{guardrail.name}", record_exception=False, set_status_on_exception=False
        )
        span = cm.__enter__()
    except Exception:
        _log.debug("Could not open guardrail span for %r", guardrail.name, exc_info=True)
        return await check(), None
    try:
        result = await check()
        _stamp_span(span, guardrail, result)
        return result, _span_id_hex(span)
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # pragma: no cover - observability must never break a run
            _log.debug("Could not close guardrail span for %r", guardrail.name, exc_info=True)


def _record(
    outcome: GuardrailOutcome,
    batch: EventBatch | None,
    guardrail: Guardrail,
    result: GuardrailResult,
    *,
    judged: Any,
    span_id: str | None,
    allow_reask: bool,
) -> None:
    """Record one check: the run's firing list, the local event row, and the
    record ``refuse()`` can later rewrite. Never raises."""
    from fastaiagent.ui.events import GuardrailEventRecord, event_outcome

    try:
        effect = effect_of(guardrail, result, allow_reask=allow_reask)
        firing = record_firing(guardrail, result, effect=effect)
        event_id = (
            batch.add(guardrail, result, data=judged, effect=effect, guardrail_span_id=span_id)
            if batch is not None
            else ""
        )
        outcome.records.append(
            GuardrailEventRecord(
                event_id=event_id,
                guardrail_name=guardrail.name,
                position=guardrail.position.value,
                effect=effect,
                outcome=event_outcome(guardrail, result, effect),
                firing=firing,
                result=result,
            )
        )
    except Exception:
        _log.debug("Could not record guardrail %r", guardrail.name, exc_info=True)


async def execute_guardrails(
    guardrails: list[Guardrail],
    data: str | dict[str, Any],
    position: GuardrailPosition,
    *,
    allow_reask: bool = False,
    agent_name: str | None = None,
    attempt: int | None = None,
    could_not_run: str | None = None,
) -> GuardrailOutcome:
    """Execute guardrails for a given position.

    ``could_not_run`` (1.91.0): the reason none of these checks can see what
    they would judge — a ``tool_call`` rule over a tool the model's provider
    runs, whose arguments exist only inside the model call. Each check is then
    recorded as errored and decided by its ``on_error``, exactly as a check
    whose runner raised: ``"block"`` halts, ``"allow"`` lets the call through
    marked unchecked. It is never a pass.

    Blocking guardrails are run first (sequentially). Non-blocking guardrails
    are run in parallel. Raises ``GuardrailBlockedError`` when a blocking
    guardrail fails **and its action halts the run** — see
    :func:`fastaiagent.guardrail.actions.halts`.

    A failure no longer always stops the run. ``warn`` records and continues;
    ``mask`` and ``override`` rewrite the payload and continue, and the rewritten
    value is what the *next* rule in the sequence sees and what
    :attr:`GuardrailOutcome.data` hands back to the caller.

    ``reask`` is always reported on :attr:`GuardrailOutcome.reask`, but only a
    caller that passes ``allow_reask=True`` gets to act on it: the agent's output
    path owns a model turn it can re-drive, and every other position does not, so
    there the rule halts. The plane makes the same choice for the same reason —
    it runs no agent loop, so it records the intent and fails closed.

    Each guardrail that runs emits one child span (on pass and block) carrying
    its outcome, so connected traces show a per-span CHECKS row.

    Each check is also recorded once (1.89.0): on the run's
    ``AgentResult.guardrails``, on :attr:`GuardrailOutcome.records`, and, when
    tracing is on, as a local event row with the text it judged. ``agent_name``
    names the agent on those rows (the current span may be a tool span) and
    ``attempt`` numbers a re-ask.
    """
    # Filter guardrails by position
    applicable = [g for g in guardrails if g.position == position]
    if not applicable:
        return GuardrailOutcome(results=[], data=data)

    blocking = [g for g in applicable if g.blocking]
    non_blocking = [g for g in applicable if not g.blocking]

    outcome = GuardrailOutcome(results=[], data=data)

    from fastaiagent.ui.events import EventBatch, recording_enabled

    batch = EventBatch(agent_name=agent_name, attempt=attempt) if recording_enabled() else None
    try:
        await _run_position(
            outcome,
            blocking,
            non_blocking,
            batch,
            allow_reask=allow_reask,
            could_not_run=could_not_run,
        )
    finally:
        # A block still records: the rows are flushed on the way out.
        if batch is not None:
            batch.flush()
    return outcome


async def _run_position(
    outcome: GuardrailOutcome,
    blocking: list[Guardrail],
    non_blocking: list[Guardrail],
    batch: EventBatch | None,
    *,
    allow_reask: bool,
    could_not_run: str | None = None,
) -> None:
    # Run blocking guardrails sequentially
    for guardrail in blocking:
        judged = outcome.data
        # The span closes before any raise below, so blocks are traced too.
        result, span_id = await _check_in_span(guardrail, judged, could_not_run=could_not_run)
        outcome.results.append(result)
        _record(
            outcome,
            batch,
            guardrail,
            result,
            judged=judged,
            span_id=span_id,
            allow_reask=allow_reask,
        )
        if result.rewrote_payload():
            # The next rule judges what the caller will actually use, not what
            # the model originally produced.
            outcome.data = result.modified_data  # type: ignore[assignment]
            outcome.modified = True
        if result.action_taken == "reask":
            # First-wins on *which* failure the caller is handed back, because it
            # can only re-drive the model once per attempt. But the decision to
            # continue is **per rule**: gating it on ``outcome.reask is None`` too
            # meant a second failing reask rule skipped this branch entirely, fell
            # through to ``halts()`` — which is True for ``reask`` on a blocking
            # rule — and hard-blocked. Adding a second reask rule silently turned
            # the pair into a block and bypassed the retry loop.
            if outcome.reask is None:
                outcome.reask = result
            if allow_reask and guardrail.blocking:
                # The caller can re-drive the model, so hand the failure back
                # instead of raising. If the re-ask never converges the caller
                # blocks — a re-ask that does not resolve must not become a pass.
                continue
        if halts(guardrail, result):
            raise GuardrailBlockedError(
                guardrail_name=guardrail.name,
                message=result.message or f"Blocked by guardrail: {guardrail.name}",
                results=outcome.results,
            )

    # Run non-blocking guardrails in parallel. They judge the payload as it
    # stands after the blocking rules, but their own rewrites are recorded as
    # evidence and never applied: an observe-only rule does not change the run.
    if non_blocking:
        judged = outcome.data
        # Each task gets its own span (gather copies the context per task).
        checked = await asyncio.gather(
            *(
                _check_in_span(g, judged, contain=True, could_not_run=could_not_run)
                for g in non_blocking
            )
        )
        for guardrail, (result, span_id) in zip(non_blocking, checked):
            outcome.results.append(result)
            _record(
                outcome,
                batch,
                guardrail,
                result,
                judged=judged,
                span_id=span_id,
                allow_reask=allow_reask,
            )
