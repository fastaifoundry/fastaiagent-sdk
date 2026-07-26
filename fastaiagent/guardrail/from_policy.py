"""Build runtime guardrails from plane-authored policy rules.

A connected SDK pulls ``guardrail_rules`` from ``GET /public/v1/policy`` (cached on
the connection). This module turns those rules into enforceable :class:`Guardrail`
objects, so a guardrail *authored centrally* on the Enterprise plane is *enforced
at the edge* by the runtime — no local code required. This is the read-down half
of the guardrail story (the SDK also pushes local guardrails up as part of an
agent's definition).

A rule is reconstructed by mapping it onto the SDK's own guardrail runners
(:mod:`fastaiagent.guardrail.implementations`), which already enforce ``regex``,
``schema``, ``classifier`` and ``llm_judge`` checks directly from ``config`` — the
same runners a locally-defined guardrail uses. A ``code`` rule, whose logic lives
in a server-side callable the SDK doesn't have, cannot be reconstructed from
config and is skipped (logged at debug) rather than silently passing.
"""

from __future__ import annotations

import logging
from typing import Any

from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType

logger = logging.getLogger(__name__)

# Plane ``guardrail_type`` (input/output/tool) → SDK position. The plane models a
# single "tool" phase; the SDK splits it into tool_call/tool_result, so a plane
# "tool" rule is enforced on the call (argument inspection) by default.
_POSITION = {
    "input": GuardrailPosition.input,
    "output": GuardrailPosition.output,
    "tool": GuardrailPosition.tool_call,
    "tool_call": GuardrailPosition.tool_call,
    "tool_result": GuardrailPosition.tool_result,
}

# Implementation types the SDK can enforce locally from a rule's ``config`` alone.
# ``code`` is excluded on purpose: its logic is a server-side callable we don't
# have, and the config-embedded code path is refused for security.
_RECONSTRUCTABLE = {
    GuardrailType.regex,
    GuardrailType.schema,
    GuardrailType.classifier,
    GuardrailType.llm_judge,
}

# (version, agent_id) -> built guardrails. Rebuilt when the policy version changes
# (the plane hashes rules into ``version``, so an edit invalidates the cache).
_CACHE: dict[tuple[str | None, str | None], list[Guardrail]] = {}


def clear_cache() -> None:
    """Drop the memoized plane guardrails (called on connect/disconnect/refresh)."""
    _CACHE.clear()


def guardrail_from_policy_rule(rule: dict[str, Any]) -> Guardrail | None:
    """Convert one ``/policy`` guardrail_rule into a runtime :class:`Guardrail`,
    or None when its logic can't be enforced locally (a ``code`` rule, or an
    unknown implementation type)."""
    impl = rule.get("implementation_type", "code")
    try:
        gtype = GuardrailType(impl)
    except ValueError:
        logger.debug("Plane guardrail has unknown implementation_type=%r; skipping.", impl)
        return None
    if gtype not in _RECONSTRUCTABLE:
        logger.debug("Plane guardrail impl=%r not locally reconstructable; skipping.", impl)
        return None

    name = rule.get("name") or "plane_guardrail"
    position = _POSITION.get(rule.get("guardrail_type", "output"), GuardrailPosition.output)
    blocking = rule.get("validation_mode", "blocking") != "parallel"
    on_error = rule.get("on_error", "block")
    if on_error not in ("allow", "block"):
        on_error = "block"

    return Guardrail(
        name=name,
        guardrail_type=gtype,
        position=position,
        config=rule.get("config") or {},
        blocking=blocking,
        description=rule.get("description") or "authored on the plane",
        on_error=on_error,
    )


def plane_guardrails_for_agent(agent_id: str | None) -> list[Guardrail]:
    """Runtime guardrails distributed to this connected agent via the cached
    policy. A rule with an empty ``agent_ids`` is domain-wide; a non-empty list
    scopes it to those agents. Returns ``[]`` when not connected or nothing applies.
    """
    from fastaiagent.client import _plane_guardrail_rules, _plane_policy_version

    rules = _plane_guardrail_rules()
    if not rules:
        return []

    key = (_plane_policy_version(), agent_id)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    built: list[Guardrail] = []
    for rule in rules:
        agent_ids = rule.get("agent_ids") or []
        if agent_ids and agent_id not in agent_ids:
            continue  # scoped to other agents
        try:
            guardrail = guardrail_from_policy_rule(rule)
        except Exception:  # a malformed rule must never break a run
            logger.debug("Failed to build guardrail from rule %r", rule.get("name"), exc_info=True)
            guardrail = None
        if guardrail is not None:
            built.append(guardrail)

    _CACHE[key] = built
    return built
