"""Built-in guardrail factories."""

from __future__ import annotations

import json
import re

from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)

# PII patterns
_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "credit_card"),
]


def no_pii(position: GuardrailPosition = GuardrailPosition.output) -> Guardrail:
    """Create a guardrail that blocks PII (SSN, email, phone, credit card)."""

    def check_pii(text: str) -> GuardrailResult:
        found = []
        for pattern, pii_type in _PII_PATTERNS:
            if re.search(pattern, text):
                found.append(pii_type)
        if found:
            return GuardrailResult(
                passed=False,
                message=f"PII detected: {', '.join(found)}",
                metadata={"pii_types": found},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="no_pii",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Blocks output containing PII (SSN, email, phone, credit card)",
        fn=check_pii,
    )


def json_valid(position: GuardrailPosition = GuardrailPosition.output) -> Guardrail:
    """Create a guardrail that validates output is valid JSON."""

    def check_json(text: str) -> GuardrailResult:
        try:
            json.loads(text)
            return GuardrailResult(passed=True)
        except (json.JSONDecodeError, TypeError):
            return GuardrailResult(passed=False, message="Output is not valid JSON")

    return Guardrail(
        name="json_valid",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Validates that output is valid JSON",
        fn=check_json,
    )


def toxicity_check(
    position: GuardrailPosition = GuardrailPosition.output,
) -> Guardrail:
    """Create a keyword-based toxicity guardrail."""
    toxic_words = [
        "hate",
        "kill",
        "attack",
        "destroy",
        "threat",
        "racist",
        "sexist",
        "slur",
    ]

    def check_toxicity(text: str) -> GuardrailResult:
        text_lower = text.lower()
        found = [w for w in toxic_words if w in text_lower]
        if found:
            return GuardrailResult(
                passed=False,
                message=f"Potentially toxic content detected: {', '.join(found)}",
                metadata={"toxic_words": found},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="toxicity_check",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Blocks potentially toxic content",
        fn=check_toxicity,
    )


def cost_limit(
    max_usd: float,
    position: GuardrailPosition = GuardrailPosition.output,
) -> Guardrail:
    """Create a guardrail that checks accumulated cost."""

    def check_cost(text: str) -> GuardrailResult:
        # Cost checking is typically done by the agent executor
        # This guardrail is a policy marker
        return GuardrailResult(
            passed=True,
            metadata={"max_usd": max_usd},
        )

    return Guardrail(
        name="cost_limit",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description=f"Enforces cost limit of ${max_usd}",
        config={"max_usd": max_usd},
        fn=check_cost,
    )


def allowed_domains(
    domains: list[str],
    position: GuardrailPosition = GuardrailPosition.tool_call,
) -> Guardrail:
    """Create a guardrail that restricts URLs to allowed domains."""

    def check_domains(text: str) -> GuardrailResult:
        import re

        urls = re.findall(r"https?://([^/\s]+)", text)
        blocked = []
        for url_domain in urls:
            if not any(url_domain.endswith(d) for d in domains):
                blocked.append(url_domain)
        if blocked:
            return GuardrailResult(
                passed=False,
                message=f"Blocked domains: {', '.join(blocked)}. Allowed: {', '.join(domains)}",
                metadata={"blocked_domains": blocked},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="allowed_domains",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description=f"Restricts URLs to domains: {', '.join(domains)}",
        config={"domains": domains},
        fn=check_domains,
    )
