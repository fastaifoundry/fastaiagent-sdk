"""What a guardrail failure costs — the action spectrum.

Guardrails answer three independent questions, and this module owns the third:

===============  ====================================  ===================================
Axis             Field                                 Question
===============  ====================================  ===================================
Scheduling       ``validation_mode`` → ``blocking``    Does it run inline, and can it halt?
Degradation      ``on_error``                          What does an *un-runnable* check mean?
**Consequence**  **``action``**                        **What does a genuine failure cost?**
===============  ====================================  ===================================

``blocking=False`` means "run concurrently, don't wait". ``action="warn"`` is a
different thing: the check still runs inline and the caller still waits for the
verdict — it just doesn't stop the run.

The one safety rule, mirrored from the plane's ``guardrail_service``: every
caller branches on what the action **actually did** (``action_taken``), never on
what it was configured to do. "An errored check always blocks" and "a mask with
nothing to mask blocks" then fall out for free instead of being special-cased at
every call site.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import regex as _regex

from fastaiagent.guardrail.guardrail import GuardrailResult, GuardrailType

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail

#: The five consequences a rule may carry. Matches the plane's ``ACTIONS``.
ACTIONS: tuple[str, ...] = ("block", "warn", "mask", "override", "reask")

#: What actually happened. Matches the plane's six ``action_taken`` strings
#: byte-for-byte, including the asymmetry that ``reask`` is not past tense.
ACTIONS_TAKEN: tuple[str, ...] = (
    "none",
    "blocked",
    "warned",
    "masked",
    "overridden",
    "reask",
)

#: Only these types locate the offending text, so only these can mask; the rest
#: return a bare verdict, and a mask with nothing to redact degrades to a block.
#: ``regex`` and ``classifier`` re-run their pattern; ``pii`` and ``secrets``
#: replace the spans their detectors return. Matches the plane's
#: ``MASKABLE_IMPLEMENTATION_TYPES``, and :func:`mask_payload` reads it — it was
#: documentation-only until 1.60.0, describing a rule it did not enforce.
MASKABLE_TYPES: tuple[GuardrailType, ...] = (
    GuardrailType.regex,
    GuardrailType.classifier,
    GuardrailType.pii,
    GuardrailType.secrets,
)

#: Severities the plane may send. Anything else (including ``None``) is unset.
SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")

DEFAULT_MASK_TOKEN = "[REDACTED]"

_NO_SPAN_SUFFIX = " (masking was configured but no maskable span was found; blocked instead)"


def coerce_action(value: Any) -> str:
    """Return a known action, defaulting to ``"block"``.

    Anything unrecognised, ``None``, or absent becomes ``"block"``. The strict
    reading is the safe one for a safety control: it keeps an older SDK correct
    against a newer plane that has learned an action this build cannot perform,
    rather than silently letting the payload through. The plane coerces the same
    way on its own push path.
    """
    return value if isinstance(value, str) and value in ACTIONS else "block"


def coerce_severity(value: Any) -> str | None:
    """Return a known severity, or ``None`` when unset/unrecognised."""
    return value if isinstance(value, str) and value in SEVERITIES else None


def _mask_token(config: dict[str, Any]) -> str:
    return str(config.get("mask_token") or DEFAULT_MASK_TOKEN)


async def mask_payload(guardrail: Guardrail, data: str | dict[str, Any]) -> str | None:
    """Redact the offending spans in ``data``, or ``None`` if nothing was masked.

    Defined for :data:`MASKABLE_TYPES` — the types that locate the offending text
    rather than returning a bare verdict. ``regex`` and ``classifier`` re-run
    their pattern; ``pii`` and ``secrets`` re-run their detector and replace the
    spans it returns, which is why they are the first non-pattern types that can
    mask at all. Returns ``None`` (which the caller must treat as "block") when:

    * the guardrail is any other type;
    * the rule is ``should_match=True`` — its failure is that the pattern is
      *absent*, so there is nothing to redact;
    * substitution changed nothing.

    The replacement is a **callable**, so a ``mask_token`` containing ``\\1`` is
    inserted literally instead of being expanded as a backreference.
    """
    if guardrail.guardrail_type not in MASKABLE_TYPES:
        return None

    config = guardrail.config or {}
    token = _mask_token(config)

    def replace(_m: Any) -> str:
        return token

    text = data if isinstance(data, str) else json.dumps(data)

    if guardrail.guardrail_type == GuardrailType.regex:
        pattern = config.get("pattern", "")
        if not pattern or config.get("should_match", False):
            return None
        flags = _regex.IGNORECASE if config.get("case_insensitive", False) else 0
        # Same ReDoS bound as _run_regex (security_audit_2 N13): the ``regex``
        # engine on a worker thread under a hard timeout. A runaway substitution
        # returns None, which degrades the mask to a block — fail closed.
        from fastaiagent.guardrail.implementations import _resolve_regex_timeout

        timeout = _resolve_regex_timeout(config)
        try:
            out = await asyncio.to_thread(
                _regex.sub, pattern, replace, text, 0, flags, timeout=timeout
            )
        except (TimeoutError, _regex.error):
            return None
        return out if out != text else None

    if guardrail.guardrail_type == GuardrailType.classifier:
        categories = config.get("categories", {}) or {}
        blocked_categories = config.get("blocked", [])
        out = text
        for category, keywords in categories.items():
            if blocked_categories and category not in blocked_categories:
                continue
            for keyword in keywords:
                out = _regex.sub(_regex.escape(keyword), replace, out, flags=_regex.IGNORECASE)
        return out if out != text else None

    if guardrail.guardrail_type in (GuardrailType.pii, GuardrailType.secrets):
        # Span replacement, not a second pattern pass: the detectors return real
        # offsets. They are re-run here rather than threaded through the result
        # because ``mask_payload`` is handed the payload, not the verdict — and
        # for `regex` above the same re-derivation is already the design. The one
        # cost is a `presidio` rule paying for a second `AnalyzerEngine`; the
        # shipped redaction templates use the regex backend.
        from fastaiagent._internal.safety_detectors import (
            detect_pii,
            detect_secrets,
            mask_spans,
        )
        from fastaiagent.guardrail.implementations import (
            _resolve_pii_backend,
            _resolve_pii_entities,
        )

        if guardrail.guardrail_type == GuardrailType.pii:
            spans = [
                (m.start, m.end)
                for m in detect_pii(
                    text,
                    entities=_resolve_pii_entities(config),
                    # Through the shared resolver, not an inline lowercase. 1.61.0
                    # moved ``_run_pii`` onto it and claimed "one place to be wrong
                    # rather than two" — this call site was the second place, and
                    # kept its own copy. It is only unreachable today because the
                    # runner validates first and ``apply_action`` short-circuits an
                    # errored result; that is an accident of ordering, not a design.
                    backend=_resolve_pii_backend(config),
                )
            ]
        else:
            spans = [(m.start, m.end) for m in detect_secrets(text)]
        out = mask_spans(text, spans, token)
        return out if out != text else None

    return None


async def apply_action(
    guardrail: Guardrail,
    data: str | dict[str, Any],
    result: GuardrailResult,
) -> GuardrailResult:
    """Stamp ``action`` / ``action_taken`` / ``modified_data`` onto ``result``.

    Called for every guardrail run, pass or fail. A clean pass reports
    ``action_taken="none"`` whatever the configured action is.

    An **errored** check always blocks, whatever the action says: nothing is
    known about the payload, so there is nothing to mask and nothing to warn
    about with confidence. ``on_error`` still decides whether the error is a
    failure at all — that is applied earlier, in ``run_guardrail``.
    """
    action = coerce_action(guardrail.action)
    result.action = action

    if result.passed:
        result.action_taken = "none"
        return result

    if result.errored:
        result.action_taken = "blocked"
        return result

    if action == "warn":
        result.action_taken = "warned"
    elif action == "override":
        config = guardrail.config or {}
        result.modified_data = (
            config.get("override_message")
            or config.get("tripwire_message")
            or result.message
            or f"Blocked by guardrail: {guardrail.name}"
        )
        result.action_taken = "overridden"
    elif action == "mask":
        modified = await mask_payload(guardrail, data)
        if modified is None:
            result.action_taken = "blocked"
            result.message = f"{result.message or guardrail.name}{_NO_SPAN_SUFFIX}"
        else:
            result.modified_data = modified
            result.action_taken = "masked"
    elif action == "reask":
        result.action_taken = "reask"
    else:
        result.action_taken = "blocked"

    return result


def halts(guardrail: Guardrail, result: GuardrailResult) -> bool:
    """Does this outcome stop the run?

    Two independent gates: ``blocking`` decides whether the rule is inline at
    all, ``action_taken`` decides what its failure cost. An observe-only rule
    never halts, whatever its action — it still produces evidence.

    ``reask`` halts by default. Only the agent's output path, which owns a model
    turn it can redo, opts out of that; every other position has nothing to
    re-drive, so failing closed is the correct reading.
    """
    return not result.passed and guardrail.blocking and result.action_taken in ("blocked", "reask")


def harness_halts(guardrail: Guardrail, result: GuardrailResult) -> str | None:
    """Should a foreign-framework proxy stop this call, and what should it say?

    The LangChain / CrewAI / PydanticAI proxies wrap somebody else's runnable.
    They own the verdict but not the payload and not the loop, so only two of
    the five actions mean anything there:

    * ``warn`` — record and carry on, exactly as in the SDK's own runtime.
    * ``block`` — stop, as before.

    ``mask``, ``override`` and ``reask`` need to rewrite the payload or re-drive
    the model, neither of which a proxy can do. They **block**, and say why,
    rather than letting text the rule wanted redacted through untouched.

    Returns the reason to block, or ``None`` to continue.
    """
    if result.passed or not getattr(guardrail, "blocking", True):
        return None
    taken = result.action_taken
    if taken == "warned":
        return None
    if taken in ("masked", "overridden"):
        return (
            f"{result.message or guardrail.name} (this guardrail rewrites the payload, "
            "which a framework proxy cannot do; blocked instead — run it on a "
            "fastaiagent Agent to get the rewrite)"
        )
    if taken == "reask":
        return (
            f"{result.message or guardrail.name} (this guardrail re-asks the model, "
            "which needs the fastaiagent agent loop; blocked instead)"
        )
    return result.message or f"Blocked by guardrail: {guardrail.name}"


def degrade_to_block(result: GuardrailResult, reason: str) -> GuardrailResult:
    """Turn a payload-rewriting outcome into a block, saying why.

    Used where a rewrite cannot be applied faithfully — a masked stream whose
    text has already been yielded, multimodal input whose parts the rewrite
    would drop, a tool result carrying image/PDF content parts. Passing the
    payload through untouched would defeat the control, so we fail closed.
    """
    result.action_taken = "blocked"
    result.modified_data = None
    result.message = f"{result.message or 'Guardrail failed'} ({reason}; blocked instead)"
    return result
