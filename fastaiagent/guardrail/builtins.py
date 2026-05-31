"""Built-in guardrail factories.

PII / prompt-injection / moderation guardrails share their detection logic with
the eval scorers via :mod:`fastaiagent._internal.safety_detectors` — one core
detector, two surfaces.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from fastaiagent._internal.safety_detectors import (
    DEFAULT_PII_ENTITIES,
    detect_pii,
    detect_prompt_injection,
    moderate_text,
)
from fastaiagent.guardrail.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailResult,
    GuardrailType,
)


def no_pii(
    position: GuardrailPosition = GuardrailPosition.output,
    *,
    entities: tuple[str, ...] = DEFAULT_PII_ENTITIES,
    backend: str = "regex",
) -> Guardrail:
    """Create a guardrail that blocks PII (email, phone, SSN, credit card).

    Delegates to :func:`fastaiagent._internal.safety_detectors.detect_pii`:
    credit cards are Luhn-validated to suppress false positives, and extra
    entity types (``ip``, ``iban``) are available via ``entities=``.
    """

    def check_pii(text: str) -> GuardrailResult:
        matches = detect_pii(text, entities=entities, backend=backend)
        if matches:
            counts = Counter(m.entity for m in matches)
            found = sorted(counts)
            return GuardrailResult(
                passed=False,
                message=f"PII detected: {', '.join(found)}",
                metadata={"pii_types": found, "counts": dict(counts)},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="no_pii",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Blocks text containing PII (email, phone, SSN, credit card)",
        fn=check_pii,
    )


def no_prompt_injection(
    position: GuardrailPosition = GuardrailPosition.input,
    *,
    mode: str = "heuristic",
    llm: Any = None,
) -> Guardrail:
    """Create a guardrail that blocks prompt-injection / jailbreak attempts.

    Delegates to
    :func:`fastaiagent._internal.safety_detectors.detect_prompt_injection`.
    Defaults to the ``input`` position (the usual attack surface) and the
    zero-dependency heuristic mode; ``mode="llm"`` opts into a classifier call.
    """

    def check_injection(text: str) -> GuardrailResult:
        res = detect_prompt_injection(text, mode=mode, llm=llm)
        if res.detected:
            return GuardrailResult(
                passed=False,
                score=res.score,
                message=res.reason,
                metadata={"matched_patterns": res.matched_patterns},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="no_prompt_injection",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Blocks prompt-injection / jailbreak attempts",
        fn=check_injection,
    )


def openai_moderation(
    position: GuardrailPosition = GuardrailPosition.output,
    *,
    client: Any = None,
    model: str = "omni-moderation-latest",
) -> Guardrail:
    """Create a guardrail that blocks content flagged by OpenAI moderation.

    Delegates to :func:`fastaiagent._internal.safety_detectors.moderate_text`.
    Requires the ``openai`` package and an API key.
    """

    def check_moderation(text: str) -> GuardrailResult:
        res = moderate_text(text, client=client, model=model)
        if res.flagged:
            flagged = [k for k, v in res.categories.items() if v]
            return GuardrailResult(
                passed=False,
                message=res.reason,
                metadata={"flagged_categories": flagged},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="openai_moderation",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Blocks content flagged by the OpenAI moderation endpoint",
        fn=check_moderation,
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
