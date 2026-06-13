"""Built-in guardrail factories.

PII / prompt-injection / moderation guardrails share their detection logic with
the eval scorers via :mod:`fastaiagent._internal.safety_detectors` — one core
detector, two surfaces.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from fastaiagent._internal.safety_detectors import (
    DEFAULT_PII_ENTITIES,
    detect_pii,
    detect_prompt_injection,
    detect_secrets,
    detect_toxicity,
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
    *,
    mode: str = "keyword",
    llm: Any = None,
    threshold: float = 0.5,
) -> Guardrail:
    """Create a toxicity guardrail.

    Defaults to the zero-dependency keyword check (unchanged behaviour). Opt
    into a much stronger LLM classifier with ``mode="llm"`` — it scores 0..1 and
    blocks when the score meets ``threshold`` (lower = stricter). Delegates to
    :func:`fastaiagent._internal.safety_detectors.detect_toxicity`.
    """

    def check_toxicity(text: str) -> GuardrailResult:
        res = detect_toxicity(text, mode=mode, llm=llm, threshold=threshold)
        if res.toxic:
            return GuardrailResult(
                passed=False,
                score=res.score,
                message=res.reason or "Potentially toxic content detected",
                metadata={"toxic_words": res.matched, "toxicity_score": res.score},
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


# --------------------------------------------------------------------------- #
# Responsible-AI "Trust Layer" guardrails
# --------------------------------------------------------------------------- #


def no_secrets(position: GuardrailPosition = GuardrailPosition.output) -> Guardrail:
    """Block leaked secrets / credentials.

    Delegates to :func:`fastaiagent._internal.safety_detectors.detect_secrets`:
    private keys, AWS / GitHub / Slack / Google / OpenAI / Stripe tokens, JWTs,
    and generic ``api_key = "..."`` assignments. Detected values are masked in
    the guardrail metadata so the secret is never re-leaked.
    """

    def check_secrets(text: str) -> GuardrailResult:
        matches = detect_secrets(text)
        if matches:
            kinds = sorted({m.kind for m in matches})
            return GuardrailResult(
                passed=False,
                message=f"Secrets detected: {', '.join(kinds)}",
                metadata={"secret_kinds": kinds, "matches": [m.to_dict() for m in matches]},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="no_secrets",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Blocks leaked secrets / credentials (keys, tokens, private keys)",
        fn=check_secrets,
    )


def grounded(
    reference: str | Callable[[], str],
    *,
    llm: Any = None,
    threshold: float = 0.7,
    position: GuardrailPosition = GuardrailPosition.output,
) -> Guardrail:
    """Block ungrounded / hallucinated output.

    Verifies the output's factual claims against a ``reference`` (your source
    text). ``reference`` is a string or a zero-arg callable returning the current
    reference — e.g. ``lambda: latest_retrieved_context`` — evaluated at check
    time so it can track per-run retrieval. Shares its engine with the eval
    ``Faithfulness`` scorer via
    :func:`fastaiagent._internal.safety_detectors.score_groundedness`.

    Note: output guardrails receive only the output text, so the reference must
    be supplied here rather than auto-wired from the agent's retrieval.
    """
    from fastaiagent._internal.async_utils import run_sync
    from fastaiagent._internal.safety_detectors import score_groundedness

    def check_grounded(text: str) -> GuardrailResult:
        ref = reference() if callable(reference) else reference
        if not ref:
            return GuardrailResult(
                passed=True, message="No reference available; groundedness skipped"
            )
        res = run_sync(score_groundedness(text, str(ref), llm=llm))
        return GuardrailResult(
            passed=res.score >= threshold,
            score=res.score,
            message=res.reason,
            metadata=res.to_dict(),
        )

    return Guardrail(
        name="grounded",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description="Blocks output not grounded in the provided reference",
        fn=check_grounded,
    )


# Alias — same guardrail, framed as hallucination prevention.
no_hallucination = grounded


def _classify_topics(
    text: str, topics: list[str], *, llm: Any = None, mode: str = "llm"
) -> list[str]:
    """Return the subset of ``topics`` the text relates to."""
    if mode == "keyword":
        low = text.lower()
        return [t for t in topics if t.lower() in low]
    if mode != "llm":
        raise ValueError(f"Unknown topic mode {mode!r}. Use 'llm' or 'keyword'.")

    from fastaiagent._internal.async_utils import run_sync

    return run_sync(_classify_topics_llm(text, topics, llm=llm))


async def _classify_topics_llm(text: str, topics: list[str], *, llm: Any = None) -> list[str]:
    import re

    from fastaiagent.llm import LLMClient, SystemMessage, UserMessage

    client = llm or LLMClient()
    prompt = (
        "Which of the following topics does the text relate to? "
        "Choose only from the list; return an empty list if none apply.\n\n"
        f"Topics: {', '.join(topics)}\n\n"
        f"Text:\n{text}\n\n"
        'Respond with JSON only: {"topics": ["..."]}'
    )
    try:
        resp = await client.acomplete(
            [
                SystemMessage("You are a topic classifier. Respond with JSON only."),
                UserMessage(prompt),
            ]
        )
        raw = (resp.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        chosen = json.loads(raw).get("topics", [])
        return [t for t in topics if t in chosen]
    except Exception:
        # Fail open — never crash the agent on a classifier error.
        return []


def banned_topics(
    topics: list[str],
    *,
    llm: Any = None,
    mode: str = "llm",
    position: GuardrailPosition = GuardrailPosition.output,
) -> Guardrail:
    """Block content that falls under any banned topic (blacklist).

    ``mode="llm"`` (default) classifies semantically; ``mode="keyword"`` does a
    zero-dependency literal match.
    """

    def check(text: str) -> GuardrailResult:
        hits = _classify_topics(text, topics, llm=llm, mode=mode)
        if hits:
            return GuardrailResult(
                passed=False,
                message=f"Banned topic(s): {', '.join(hits)}",
                metadata={"banned_topics": hits},
            )
        return GuardrailResult(passed=True)

    return Guardrail(
        name="banned_topics",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description=f"Blocks banned topics: {', '.join(topics)}",
        config={"topics": topics},
        fn=check,
    )


def allowed_topics(
    topics: list[str],
    *,
    llm: Any = None,
    mode: str = "llm",
    position: GuardrailPosition = GuardrailPosition.output,
) -> Guardrail:
    """Allow only content within the given topics (whitelist).

    ``mode="llm"`` (default) classifies semantically; ``mode="keyword"`` does a
    zero-dependency literal match.
    """

    def check(text: str) -> GuardrailResult:
        on = _classify_topics(text, topics, llm=llm, mode=mode)
        if on:
            return GuardrailResult(passed=True, metadata={"matched_topics": on})
        return GuardrailResult(
            passed=False,
            message=f"Off-topic — allowed topics: {', '.join(topics)}",
            metadata={"allowed_topics": topics},
        )

    return Guardrail(
        name="allowed_topics",
        guardrail_type=GuardrailType.code,
        position=position,
        blocking=True,
        description=f"Restricts content to topics: {', '.join(topics)}",
        config={"topics": topics},
        fn=check,
    )


def responsible_ai(
    *,
    pii: bool = True,
    prompt_injection: bool = True,
    secrets: bool = True,
    toxicity: bool = False,
    moderation: bool = False,
    grounded_to: str | Callable[[], str] | None = None,
    banned: list[str] | None = None,
    allowed: list[str] | None = None,
    llm: Any = None,
    threshold: float = 0.7,
) -> list[Guardrail]:
    """Compose a Responsible-AI "Trust Layer" as a list of guardrails.

    Spread the result into ``Agent(guardrails=responsible_ai(...))``. The
    zero-dependency checks (prompt-injection on input, PII + secrets on output)
    are on by default; the LLM-backed checks (groundedness, topic controls, LLM
    toxicity, OpenAI moderation) are opt-in, so the default bundle adds **no**
    extra LLM calls.

    Example::

        agent = Agent(name="safe", llm=llm, guardrails=responsible_ai(
            grounded_to=lambda: latest_context,
            banned=["politics", "legal advice"],
            toxicity=True,
            llm=llm,
        ))
    """
    rails: list[Guardrail] = []
    if prompt_injection:
        rails.append(no_prompt_injection())
    if pii:
        rails.append(no_pii())
    if secrets:
        rails.append(no_secrets())
    if toxicity:
        rails.append(toxicity_check(mode="llm" if llm is not None else "keyword", llm=llm))
    if moderation:
        rails.append(openai_moderation())
    if grounded_to is not None:
        rails.append(grounded(grounded_to, llm=llm, threshold=threshold))
    if banned:
        rails.append(banned_topics(banned, llm=llm))
    if allowed:
        rails.append(allowed_topics(allowed, llm=llm))
    return rails
