"""The guardrail loop shared by the LangChain, CrewAI and PydanticAI proxies.

A proxy wraps somebody else's runnable: it owns the verdict, not the payload
and not the loop. So ``warn`` records and continues, and everything else that
fails stops the call with ``GuardrailBlocked``. ``mask``, ``override`` and
``reask`` block too, and say why: they need the SDK's own agent loop. See
:func:`fastaiagent.guardrail.actions.harness_halts`.

Each check is recorded **once** (1.89.0), with the effect it really had. The
proxies used to call ``g.execute()``, which could log a row, and then log a
second one on failure; a masked check was recorded as ``filtered`` though the
call was blocked.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def run_proxy_guardrails(
    text: str,
    guardrails: list[Any] | None,
    *,
    side: str,
    agent_name: str | None,
    framework: str,
) -> None:
    """Run ``guardrails`` over ``text`` for a framework proxy; raise on a halt."""
    if not guardrails:
        return

    from fastaiagent._internal.async_utils import run_sync
    from fastaiagent.guardrail.actions import effect_of, harness_halts
    from fastaiagent.integrations._registry import GuardrailBlocked
    from fastaiagent.ui.events import EventBatch, recording_enabled

    batch = EventBatch(agent_name=agent_name) if recording_enabled() else None
    try:
        for g in guardrails:
            result = run_sync(g._acheck(text))
            reason = harness_halts(g, result)
            effect = effect_of(g, result)
            final_reason: str | None = None
            if reason is not None:
                effect = "blocked"
                if result.action_taken in ("masked", "overridden", "reask"):
                    final_reason = reason
            if batch is not None:
                batch.add(
                    g,
                    result,
                    data=text,
                    effect=effect,
                    extra_metadata={"framework": framework, "side": side},
                    final_reason=final_reason,
                )
            if reason is not None:
                raise GuardrailBlocked(f"{side} blocked by {g.name}: {reason}")
    finally:
        if batch is not None:
            batch.flush()
