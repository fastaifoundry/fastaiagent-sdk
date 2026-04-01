"""Guardrail execution — blocking and parallel modes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.guardrail import GuardrailPosition, GuardrailResult

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail


async def execute_guardrails(
    guardrails: list[Guardrail],
    data: str | dict[str, Any],
    position: GuardrailPosition,
) -> list[GuardrailResult]:
    """Execute guardrails for a given position.

    Blocking guardrails are run first (sequentially).
    Non-blocking guardrails are run in parallel.
    Raises GuardrailBlockedError if any blocking guardrail fails.
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
        for r in parallel_results:
            if isinstance(r, GuardrailResult):
                results.append(r)
            elif isinstance(r, Exception):
                results.append(GuardrailResult(passed=False, message=str(r)))

    return results
