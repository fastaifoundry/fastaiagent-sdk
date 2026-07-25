"""Guardrail execution — blocking and parallel modes."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.guardrail import GuardrailPosition, GuardrailResult

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail


def _emit_guardrail_span(guardrail: Guardrail, result: GuardrailResult) -> None:
    """Emit one child span carrying a guardrail's outcome.

    Emitted on **pass and block** so the console shows green passes too. The
    span nests under whatever span is currently active (the agent/turn span),
    since OTel context propagates through ``await``. Best-effort — a tracing
    failure must never break guardrail execution.
    """
    try:
        from opentelemetry.trace import Status, StatusCode

        from fastaiagent.trace.otel import get_tracer
        from fastaiagent.trace.span import set_guardrail_attributes

        if result.errored:
            check_result = "error"
        elif result.passed:
            check_result = "pass"
        else:
            check_result = "block"
        checks = json.dumps([{"name": guardrail.name, "result": check_result}])
        tracer = get_tracer("fastaiagent.guardrail")
        with tracer.start_as_current_span(f"guardrail.{guardrail.name}") as span:
            set_guardrail_attributes(
                span,
                name=guardrail.name,
                position=guardrail.position.value,
                passed=result.passed,
                checks=checks,
                errored=result.errored,
            )
            if result.passed:
                # A degraded pass (errored + on_error="allow") keeps an OK
                # status — the run continued — but the errored attribute and
                # the "error" check result keep it distinguishable.
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(
                    Status(
                        StatusCode.ERROR,
                        result.message or f"Blocked by guardrail: {guardrail.name}",
                    )
                )
    except Exception:  # pragma: no cover - observability must never break a run
        pass


async def execute_guardrails(
    guardrails: list[Guardrail],
    data: str | dict[str, Any],
    position: GuardrailPosition,
) -> list[GuardrailResult]:
    """Execute guardrails for a given position.

    Blocking guardrails are run first (sequentially).
    Non-blocking guardrails are run in parallel.
    Raises GuardrailBlockedError if any blocking guardrail fails.

    Each guardrail that runs emits one child span (on pass and block) carrying
    its outcome, so connected traces show a per-span CHECKS row.
    """
    # Filter guardrails by position
    applicable = [g for g in guardrails if g.position == position]
    if not applicable:
        return []

    blocking = [g for g in applicable if g.blocking]
    non_blocking = [g for g in applicable if not g.blocking]

    results: list[GuardrailResult] = []

    # Run blocking guardrails sequentially
    for guardrail in blocking:
        result = await guardrail.aexecute(data)
        results.append(result)
        # Emit the span before raising so blocks are traced too.
        _emit_guardrail_span(guardrail, result)
        if not result.passed:
            raise GuardrailBlockedError(
                guardrail_name=guardrail.name,
                message=result.message or f"Blocked by guardrail: {guardrail.name}",
                results=results,
            )

    # Run non-blocking guardrails in parallel
    if non_blocking:
        tasks = [g.aexecute(data) for g in non_blocking]
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
        for guardrail, r in zip(non_blocking, parallel_results):
            if isinstance(r, GuardrailResult):
                results.append(r)
                _emit_guardrail_span(guardrail, r)
            elif isinstance(r, Exception):
                failed = GuardrailResult(passed=False, message=str(r))
                results.append(failed)
                _emit_guardrail_span(guardrail, failed)

    return results
