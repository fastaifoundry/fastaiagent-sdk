"""Guardrail system with 5 implementation types and built-in factories."""

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
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)

__all__ = [
    "Guardrail",
    "GuardrailResult",
    "GuardrailPosition",
    "GuardrailType",
    "execute_guardrails",
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
