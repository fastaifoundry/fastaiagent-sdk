"""Guardrail execution — blocking and parallel modes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.actions import halts
from fastaiagent.guardrail.guardrail import GuardrailPosition, GuardrailResult, GuardrailType

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail


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
    GuardrailType.topic: frozenset({"mode", "matched", "topics"}),
    GuardrailType.content_safety: frozenset(
        {"taxonomy", "scores", "thresholds", "tripped", "unscored"}
    ),
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


def _emit_guardrail_span(guardrail: Guardrail, result: GuardrailResult) -> None:
    """Emit one child span carrying a guardrail's outcome.

    Emitted on **pass and block** so the console shows green passes too. The
    span nests under whatever span is currently active (the agent/turn span),
    since OTel context propagates through ``await``. Best-effort — a tracing
    failure must never break guardrail execution.

    This is the **SDK runtime's** emit half: it pairs the compute done by
    ``guardrail.aexecute`` with the SDK's own tracer. A foreign runtime that
    borrows :func:`fastaiagent.run_guardrail` calls
    :func:`fastaiagent.emit_guardrail` with *its* tracer instead — same span
    shape, its own exporter.
    """
    try:
        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import emit_guardrail

        if result.errored:
            check_result = "error"
        elif result.passed:
            check_result = "pass"
        else:
            check_result = "block"
        checks = json.dumps([{"name": guardrail.name, "result": check_result}])
        emit_guardrail(
            get_tracer("fastaiagent.guardrail"),
            name=guardrail.name,
            position=guardrail.position.value,
            passed=result.passed,
            checks=checks,
            errored=result.errored,
            message=result.message,
            action=result.action,
            action_taken=result.action_taken,
            severity=guardrail.severity,
            floor=guardrail.floor,
            detail=_exportable_detail(guardrail, result),
        )
    except Exception:  # pragma: no cover - observability must never break a run
        pass


async def execute_guardrails(
    guardrails: list[Guardrail],
    data: str | dict[str, Any],
    position: GuardrailPosition,
    *,
    allow_reask: bool = False,
) -> GuardrailOutcome:
    """Execute guardrails for a given position.

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
    """
    # Filter guardrails by position
    applicable = [g for g in guardrails if g.position == position]
    if not applicable:
        return GuardrailOutcome(results=[], data=data)

    blocking = [g for g in applicable if g.blocking]
    non_blocking = [g for g in applicable if not g.blocking]

    outcome = GuardrailOutcome(results=[], data=data)

    # Run blocking guardrails sequentially
    for guardrail in blocking:
        result = await guardrail.aexecute(outcome.data)
        outcome.results.append(result)
        # Emit the span before raising so blocks are traced too.
        _emit_guardrail_span(guardrail, result)
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
        tasks = [g.aexecute(outcome.data) for g in non_blocking]
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
        for guardrail, r in zip(non_blocking, parallel_results):
            if isinstance(r, GuardrailResult):
                outcome.results.append(r)
                _emit_guardrail_span(guardrail, r)
            elif isinstance(r, Exception):
                failed = GuardrailResult(
                    passed=False, message=str(r), errored=True, action_taken="blocked"
                )
                outcome.results.append(failed)
                _emit_guardrail_span(guardrail, failed)

    return outcome
