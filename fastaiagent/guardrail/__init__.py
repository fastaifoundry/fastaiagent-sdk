"""Guardrail system with 10 implementation types and built-in factories."""

from fastaiagent.guardrail.actions import ACTIONS, ACTIONS_TAKEN
from fastaiagent.guardrail.builtins import (
    allowed_domains,
    allowed_topics,
    banned_topics,
    cost_limit,
    grounded,
    json_valid,
    no_hallucination,
    no_pii,
    no_prompt_injection,
    no_secrets,
    openai_moderation,
    responsible_ai,
    toxicity_check,
)
from fastaiagent.guardrail.context import (
    clear_guardrail_context,
    get_guardrail_context,
    guardrail_context,
    set_guardrail_context,
)
from fastaiagent.guardrail.executor import GuardrailOutcome, execute_guardrails
from fastaiagent.guardrail.from_policy import (
    guardrail_from_policy_rule,
    plane_guardrails_for_agent,
)
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)
from fastaiagent.guardrail.implementations import run_guardrail

__all__ = [
    "Guardrail",
    "GuardrailResult",
    "GuardrailPosition",
    "GuardrailType",
    # What a failure costs: block | warn | mask | override | reask.
    "ACTIONS",
    "ACTIONS_TAKEN",
    # Run-scoped context for the rules that need more than the payload
    # (``groundedness`` scores an answer against the context it was given).
    "guardrail_context",
    "set_guardrail_context",
    "get_guardrail_context",
    "clear_guardrail_context",
    # Runtime: compute + emit, for the SDK's own agent loop.
    "execute_guardrails",
    "GuardrailOutcome",
    # Primitives: compute only, no span. A foreign runtime pairs these with
    # fastaiagent.emit_guardrail on its own tracer — compute ≠ emit.
    "run_guardrail",
    "guardrail_from_policy_rule",
    "plane_guardrails_for_agent",
    "no_pii",
    "no_prompt_injection",
    "openai_moderation",
    "json_valid",
    "toxicity_check",
    "cost_limit",
    "allowed_domains",
    # Responsible-AI "Trust Layer"
    "no_secrets",
    "grounded",
    "no_hallucination",
    "banned_topics",
    "allowed_topics",
    "responsible_ai",
]
