"""Keeping credentials out of everything that leaves the process (1.89.0).

Two tools for two different leaks.

**Shareable LLM configs (S1).** A model-backed guardrail's ``config["llm"]`` and
a Chain decision node's ``decision["llm"]`` are ``LLMClient`` kwargs stored as a
plain dict, so the rule can be serialized: into the root span's
``agent.guardrails``, into the agent definition pushed to the plane, and back
out of the plane to every agent it is distributed to. Such a dict is portable
data, so it must never hold a credential. The process that *runs* the rule
supplies its own key. That is the same contract ``LLMClient.to_dict()`` keeps,
and LangChain keeps it too (``SecretStr`` plus ``lc_secrets``, serialized as an
env-var reference).

The control is an **allowlist**, not a list of names that look secret. A
denylist leaks the first credential nobody thought to name, which is exactly
how ``agent.guardrails`` escaped ``SENSITIVE_ATTR_KEYS``.

* :func:`check_llm_config` runs at construction and **refuses**.
* :func:`allowlist_llm_config` runs on rebuild (a rule stored by an older SDK
  must still run) and on serialization. It **drops** and says so.

**Provider error text (P9).** OpenAI echoes a rejected key back in its error
body, and an exception message is copied into span events, status
descriptions, logs and the plane's ``trace.error``. :func:`scrub_secrets`
cleans such text where the exception is built, so every channel downstream
receives it already clean.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote, quote_plus

logger = logging.getLogger(__name__)

#: The shareable half of ``LLMClient``: every field its ``to_dict()`` emits,
#: plus Bedrock's ``region`` (read from its extra kwargs). Nothing that
#: authenticates belongs here. Pinned against ``LLMClient.to_dict()`` by
#: ``tests/test_secret_serialization_sweep.py``.
LLM_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "provider",
        "model",
        "base_url",
        "temperature",
        "max_tokens",
        "max_retries",
        "top_p",
        "stop",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "parallel_tool_calls",
        "pdf_mode",
        "max_pdf_pages",
        "max_image_size_mb",
        "region",
        # Serialized by LLMClient.to_dict() since 1.91.0 and none a secret; left
        # out of this list, a rule's ``llm`` config could not carry them (X22).
        "reasoning",
        "cache",
        "api",
    }
)

#: What a scrubbed secret is replaced with.
REDACTED = "[REDACTED]"


def _disallowed(cfg: dict[str, Any]) -> list[str]:
    return sorted(str(k) for k in cfg if k not in LLM_CONFIG_FIELDS)


def check_llm_config(cfg: Any, where: str) -> Any:
    """Refuse a shareable ``llm`` config holding anything outside the allowlist.

    Returns a copy of ``cfg``. A non-dict is returned unchanged: it is not a
    shareable config, and the caller decides what it means. The error names
    the refused *fields*, never their values.
    """
    if not isinstance(cfg, dict):
        return cfg
    bad = _disallowed(cfg)
    if bad:
        raise ValueError(
            f"{where}: the 'llm' config cannot hold {bad}. It is part of a rule that is "
            f"serialized into traces and pushed to the control plane, so it takes only "
            f"shareable LLMClient fields ({', '.join(sorted(LLM_CONFIG_FIELDS))}). "
            f"Supply credentials through the provider's environment variable (for "
            f"example OPENAI_API_KEY). Where the API accepts a live client, "
            f"llm=LLMClient(api_key=...) also works: it keeps the rule local and is "
            f"never serialized."
        )
    return dict(cfg)


def allowlist_llm_config(cfg: Any, where: str, *, level: int = logging.WARNING) -> Any:
    """``cfg`` without the fields outside the allowlist, logging what was dropped.

    For configs that arrive already built: a rule stored by an SDK older than
    1.89.0, read back from a trace or the plane, must still run, so it loses
    the field rather than failing. The rule then resolves its key from the
    environment, as every rebuilt ``LLMClient`` does.
    """
    if not isinstance(cfg, dict):
        return cfg
    bad = _disallowed(cfg)
    if bad:
        logger.log(
            level,
            "%s: dropped %s from the 'llm' config. A shareable rule never carries "
            "credentials; this process resolves its own key from the environment.",
            where,
            bad,
        )
    return {k: v for k, v in cfg.items() if k in LLM_CONFIG_FIELDS}


# ---------------------------------------------------------------------------
# Secret text scrubbing (P9)
# ---------------------------------------------------------------------------

#: A key the provider echoed back half-masked: ``sk-proj-****abcd``. OpenAI does
#: this for real keys; only very short ones come back whole.
_MASKED_ECHO = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_\-]{1,24}\*{3,}[A-Za-z0-9_\-]{0,8}")

#: Shorter "secrets" are not replaced verbatim: at that length a substring
#: replace would mangle ordinary words.
_MIN_SECRET_LEN = 8

#: Environment variables that hold credentials the SDK itself reads. Provider
#: presets add their own (see :func:`known_credential_values`).
_CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "FASTAIAGENT_API_KEY",
)


def known_credential_values() -> list[str]:
    """The credential values present in this process's environment.

    Read on every call, not cached: a key can be set after import, and
    reading a couple of dozen variables costs nothing next to an exception.
    """
    names = list(_CREDENTIAL_ENV_VARS)
    try:
        from fastaiagent.llm.providers import list_presets

        names.extend(p.env_var for p in list_presets() if p.env_var)
    except Exception:  # pragma: no cover — scrubbing must never fail on a registry issue
        logger.debug("Could not read provider presets for secret scrubbing", exc_info=True)
    values = []
    for name in dict.fromkeys(names):
        value = os.environ.get(name)
        if value:
            values.append(value)
    return values


def scrub_secrets(text: str, secrets: Iterable[str | None] = (), *, patterns: bool = True) -> str:
    """``text`` with credentials replaced by ``[REDACTED]``.

    Three passes, in order:

    1. Each known secret (8+ characters), plus its URL-encoded forms, verbatim.
    2. A provider's half-masked echo of a key (``sk-proj-****abcd``).
    3. The secret-shaped patterns of the ``secrets`` guardrail (OpenAI,
       Anthropic, AWS, GitHub, Slack, Google, Stripe, JWT, private keys, and
       ``api_key = "..."`` assignments).

    Only credential text is replaced. The rest of the message, such as the
    provider's explanation, the status and the request id, stays readable.

    ``patterns=False`` runs pass 1 only. Use it for text that is the user's
    own payload rather than an error message, where a pattern match could
    rewrite content the user meant to keep.
    """
    if not text:
        return text
    from fastaiagent._internal.safety_detectors import detect_secrets, mask_spans

    out = text
    known = {s for s in secrets if s and len(s) >= _MIN_SECRET_LEN}
    for secret in sorted(known, key=len, reverse=True):
        for form in {secret, quote(secret, safe=""), quote_plus(secret)}:
            out = out.replace(form, REDACTED)
    if not patterns:
        return out
    out = _MASKED_ECHO.sub(REDACTED, out)
    found = detect_secrets(out)
    if found:
        out = mask_spans(out, [(m.start, m.end) for m in found], REDACTED)
    return out


def scrub_event_dicts(events: Any) -> Any:
    """Stored-shape span events (``{"name", "timestamp", "attributes"}``) with
    credentials scrubbed from their exception text.

    The fallback behind :meth:`LLMClient._provider_error`: an exception raised
    by user or library code, rather than built by the SDK, can still carry a
    key in ``exception.message`` or ``exception.stacktrace``. Only those
    message-bearing keys are touched. Returns ``events`` itself when nothing
    changed, so the common case costs no copy.
    """
    if not events:
        return events
    from fastaiagent.trace.redaction import SENSITIVE_EVENT_ATTR_KEYS

    secrets = known_credential_values()
    out: list[Any] = []
    changed = False
    for event in events:
        attrs = event.get("attributes") if isinstance(event, dict) else None
        if not isinstance(attrs, dict):
            out.append(event)
            continue
        new_attrs: dict[str, Any] | None = None
        for key in SENSITIVE_EVENT_ATTR_KEYS:
            value = attrs.get(key)
            if isinstance(value, str):
                clean = scrub_secrets(value, secrets)
                if clean != value:
                    new_attrs = new_attrs if new_attrs is not None else dict(attrs)
                    new_attrs[key] = clean
        if new_attrs is None:
            out.append(event)
        else:
            out.append({**event, "attributes": new_attrs})
            changed = True
    return out if changed else events


__all__ = [
    "LLM_CONFIG_FIELDS",
    "REDACTED",
    "allowlist_llm_config",
    "check_llm_config",
    "known_credential_values",
    "scrub_event_dicts",
    "scrub_secrets",
]
