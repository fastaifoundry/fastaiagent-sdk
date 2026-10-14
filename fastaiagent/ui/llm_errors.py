"""What the Playground tells you when a model call fails (1.89.0, P8).

Every failure read "LLM call failed." plus a correlation id: a wrong key, an
unknown model, a rate limit, Claude refusing ``temperature`` and Ollama not
running all looked the same. The opaque message was a deliberate choice
(security review H7: provider text can carry keys and ids). Provider errors
are scrubbed of credentials where they are built since 1.89.0 (P9), so the
reason can be shown, scrubbed again, without ids, and truncated.

Never shown: the provider's error code or type, request or organisation ids,
exception class names, the traceback. ``FASTAIAGENT_UI_PROVIDER_ERRORS=redacted``
restores the opaque message (decision D1).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Literal

FailureKind = Literal[
    "auth",
    "not_found",
    "bad_request",
    "rate_limited",
    "provider_error",
    "connection",
    "timeout",
    "config",
    "internal",
]

_ID_PATTERNS = (
    re.compile(r"\breq_[A-Za-z0-9]+"),
    re.compile(r"\borg-[A-Za-z0-9]+"),
    re.compile(r"request[_ -]?id[=: ]+\S+", re.IGNORECASE),
)
_MAX_MESSAGE = 400


@dataclass
class LLMFailure:
    kind: FailureKind
    provider_status: int | None
    provider_message: str | None
    hint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provider_status": self.provider_status,
            "provider_message": self.provider_message,
            "hint": self.hint,
        }


def errors_redacted() -> bool:
    """``FASTAIAGENT_UI_PROVIDER_ERRORS=redacted`` hides the provider's reason."""
    return os.environ.get("FASTAIAGENT_UI_PROVIDER_ERRORS", "detail").strip().lower() == "redacted"


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in out and len(out) < 8:
        out.append(cur)
        cur = cur.__cause__ or cur.__context__
    return out


def _clean(text: str | None, secrets: list[str | None]) -> str | None:
    if not text:
        return None
    from fastaiagent._internal.scrub import known_credential_values, scrub_secrets

    for pattern in _ID_PATTERNS:
        text = pattern.sub("[id]", text)
    text = scrub_secrets(text, [*secrets, *known_credential_values()])
    text = " ".join(text.split())
    return text[:_MAX_MESSAGE] + ("…" if len(text) > _MAX_MESSAGE else "")


def _kind_for_status(status: int | None) -> FailureKind:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status in (400, 422):
        return "bad_request"
    if status == 429:
        return "rate_limited"
    if status is not None and status >= 500:
        return "provider_error"
    return "internal"


def _sampling_hint(provider: str, model: str, message: str, params: dict[str, Any]) -> str | None:
    from fastaiagent.llm.param_rules import sampling_rule

    low = message.lower()
    if "temperature" not in low and "top_p" not in low:
        return None
    rule = sampling_rule(provider, model)
    if "both" in low or (
        rule.exclusive and params.get("temperature") is not None and params.get("top_p") is not None
    ):
        return "This model takes temperature or top_p, not both: turn one of them off."
    if "temperature" in low and rule.temperature == "default_only":
        default = f"{rule.default_temperature:g}"
        return f"This model accepts only its default temperature ({default}): turn Temperature off."
    if "top_p" in low and rule.top_p == "unsupported":
        return "This model does not accept top_p: turn Top P off."
    if "top_p" in low and rule.top_p == "default_only":
        default = f"{rule.default_top_p:g}"
        return f"This model accepts only its default top_p ({default}): turn Top P off."
    return rule.note or None


def describe_llm_failure(
    exc: BaseException,
    *,
    provider: str,
    model: str,
    base_url: str | None = None,
    params: dict[str, Any] | None = None,
    secrets: list[str | None] | None = None,
) -> LLMFailure:
    """Classify a failed model call and say why, safely (see module docstring)."""
    import httpx

    from fastaiagent._internal.errors import LLMError, LLMProviderError, LLMRateLimitError

    secrets = list(secrets or [])
    params = params or {}
    chain = _chain(exc)

    for e in chain:
        if isinstance(e, LLMProviderError):
            status = e.status_code
            kind = _kind_for_status(status)
            message = _clean(e.provider_message, secrets)
            hint: str | None = None
            if kind == "auth":
                hint = (
                    f"The provider rejected the credential for '{provider}'. Check its API key "
                    "on this server, or the token in the Connection panel."
                )
            elif kind == "not_found":
                hint = f"The provider does not know model '{model}', or this key cannot use it."
            elif kind == "bad_request":
                hint = _sampling_hint(provider, model, message or "", params)
            elif kind == "rate_limited":
                hint = "Rate limited by the provider. Wait a moment and run again."
            elif kind == "provider_error":
                hint = "The provider failed on its side. Run again shortly."
            return LLMFailure(kind, status, message, hint)
        if isinstance(e, LLMRateLimitError):
            return LLMFailure(
                "rate_limited", 429, _clean(str(e), secrets), "Rate limited by the provider."
            )

    for e in chain:
        if isinstance(e, httpx.TimeoutException):
            return LLMFailure("timeout", None, None, "The provider did not answer in time.")
        if isinstance(e, httpx.RemoteProtocolError):
            # The peer closed the connection mid-reply (a dropped stream).
            where = base_url or "The provider"
            return LLMFailure(
                "connection", None, None, f"{where} closed the connection before the reply ended."
            )
        if isinstance(e, (httpx.ConnectError, httpx.UnsupportedProtocol, httpx.NetworkError)):
            where = base_url or "the provider"
            hint = f"Could not connect to {where}."
            if provider in ("ollama", "lmstudio", "vllm"):
                hint += f" Is the {provider} server running?"
            if isinstance(e, httpx.UnsupportedProtocol):
                hint = (
                    f"'{provider}' needs an Endpoint (base URL): set one in the Connection panel."
                )
                return LLMFailure("config", None, None, hint)
            return LLMFailure("connection", None, None, hint)

    for e in chain:
        if isinstance(e, LLMError) and not isinstance(e, LLMProviderError):
            # Our own configuration errors (no streaming for a provider, a
            # missing endpoint): the message is the SDK's, scrubbed anyway.
            return LLMFailure("config", None, _clean(str(e), secrets), None)

    return LLMFailure("internal", None, None, None)


__all__ = ["FailureKind", "LLMFailure", "describe_llm_failure", "errors_redacted"]
