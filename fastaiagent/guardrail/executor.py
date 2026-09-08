"""Guardrail execution — blocking and parallel modes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.actions import halts
from fastaiagent.guardrail.guardrail import GuardrailPosition, GuardrailResult

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
        if result.action_taken == "reask" and outcome.reask is None:
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
