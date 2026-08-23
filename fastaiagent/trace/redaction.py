"""Opt-in redaction for trace span attributes.

Two redaction modes:

* **Capture-time** (``mode="capture"`` or ``mode="both"``): regex
  patterns are applied to sensitive attribute keys *before* the JSON
  blob is written to SQLite. Downstream OTel exporters added via
  :func:`fastaiagent.trace.otel.add_exporter` also receive redacted
  attributes, since redaction happens in the same processor before
  the batch span exporter sees the span.
* **Read-time** (``mode="read"`` or ``mode="both"``): the UI trace
  endpoints apply redaction on the way out when the caller passes
  ``?redact=true``. Useful for operator screen-shares without
  changing what's stored.

**Defaults to OFF.** No redaction happens until a caller invokes
:func:`set_redaction_policy`. This is intentional per the v1.14
top-3-recommendation decision log:
``claude_files/top3recommendation.md`` §1.5 decision 1.

For a coarser "redact every payload field" knob, see
``FASTAIAGENT_TRACE_PAYLOADS=0`` in ``fastaiagent.trace.span`` — that
drops payload-bearing fields entirely instead of masking them.

Example::

    from fastaiagent.trace.redaction import RedactionPolicy, set_redaction_policy

    set_redaction_policy(RedactionPolicy(
        patterns=[r"sk-[A-Za-z0-9]{32,}", r"\\b\\d{4}-\\d{4}-\\d{4}-\\d{4}\\b"],
        replacement="[REDACTED]",
        mode="capture",
    ))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

RedactionMode = Literal["off", "capture", "read", "both"]

# Span attribute keys whose values may contain sensitive payload content.
#
# This is the AUTHORITATIVE payload registry. It drives two things:
#   1. Redaction — masks these keys' string values (capture/read/export).
#   2. Export payload gate — ``apply_export_policy`` drops these keys entirely
#      when ``FASTAIAGENT_TRACE_PAYLOADS=0`` (egress-gated model, N3).
# Local capture is always full fidelity, so any attribute holding free-text /
# user / model content MUST be listed here or it will egress even when the
# operator opted out. Structural metadata (counts, latencies, ids, types,
# model/provider names, tool schemas) is deliberately absent so it always flows.
# Adding a new payload attribute anywhere in the codebase is a one-line change
# here — keep this in sync.
SENSITIVE_ATTR_KEYS: frozenset[str] = frozenset(
    {
        # GenAI semantic-convention payloads (see ``fastaiagent.trace.span``).
        "gen_ai.request.messages",
        "gen_ai.request.tools",
        "gen_ai.response.content",
        "gen_ai.response.tool_calls",
        # Agent inputs/outputs/system prompt captured by ``Agent._arun_traced``.
        "agent.input",
        "agent.output",
        "agent.system_prompt",
        "fastaiagent.agent.input",
        "fastaiagent.agent.output",
        # Tool invocation payloads.
        "tool.input",
        "tool.args",
        "tool.output",
        "tool.result",
        # Chain payloads — ``Chain.aexecute`` writes JSON-serialized state.
        "chain.input",
        "chain.output",
        # Retrieval / KB payloads — the query and returned document content can
        # both carry user data and proprietary corpus text. Set by
        # ``fastaiagent.kb._tracing`` and the LangChain retriever integration.
        "retrieval.query",
        "retrieval.doc_ids",
        "retrieval.documents",
        # Research template free-form payloads.
        "fastaiagent.research.brief",
        "fastaiagent.research.findings",
        # Memory observability payloads — recalled snippets / extracted-fact
        # detail can contain PII (FactExtractionBlock extracts facts about the
        # user). Numeric ``memory.scores`` is structural and stays clear.
        "memory.query",
        "memory.snippets",
        "memory.detail",
        # Framework-integration payloads (crewai / pydantic-ai). Structural
        # fields (names, roles, models, process, counts) are intentionally
        # excluded; only free-text content is listed.
        "crewai.agent.backstory",
        "crewai.agent.goal",
        "crewai.agent.output",
        "crewai.crew.inputs",
        "crewai.crew.output",
        "crewai.task.description",
        "crewai.task.output",
        "pydanticai.agent.system_prompt",
    }
)


@dataclass(frozen=True)
class RedactionPolicy:
    """Configuration for trace redaction.

    Attributes:
        patterns: Iterable of regex strings. Compiled once on policy
            construction. Matches are replaced with ``replacement``.
        replacement: String to substitute for each match. Defaults to
            ``"[REDACTED]"``.
        apply_to_keys: Iterable of attribute keys to scan. Defaults to
            :data:`SENSITIVE_ATTR_KEYS`. Override to redact additional
            custom attributes, or to narrow the scope.
        mode: One of ``"off"``, ``"capture"``, ``"read"``, ``"both"``.
            Defaults to ``"capture"`` since "off" already implies
            constructing no policy — passing ``mode="off"`` is mostly
            useful for temporarily disabling an installed policy.
    """

    patterns: tuple[str, ...] = ()
    replacement: str = "[REDACTED]"
    apply_to_keys: frozenset[str] = SENSITIVE_ATTR_KEYS
    mode: RedactionMode = "capture"

    _compiled: tuple[re.Pattern[str], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Pre-compile regexes once — ``frozen=True`` blocks normal
        # assignment, so use object.__setattr__ to populate the cache.
        compiled = tuple(re.compile(p) for p in self.patterns)
        object.__setattr__(self, "_compiled", compiled)

    def redact_string(self, value: str) -> str:
        """Apply every compiled pattern to ``value`` and return the result."""
        out = value
        for pat in self._compiled:
            out = pat.sub(self.replacement, out)
        return out


# Module-level singleton — ``None`` means "no redaction". Guarded by a lock
# only because read-mode redaction can race with set/get in multi-threaded
# UI servers (the hot path is read-only, so contention is rare).
_policy: RedactionPolicy | None = None
_policy_lock = Lock()


def set_redaction_policy(policy: RedactionPolicy | None) -> None:
    """Install (or remove) the global redaction policy.

    Pass ``None`` to disable redaction entirely. The change takes effect
    immediately for subsequent spans; spans already written to SQLite
    are not modified.
    """
    global _policy
    with _policy_lock:
        _policy = policy


def get_redaction_policy() -> RedactionPolicy | None:
    """Return the installed policy or ``None`` if redaction is disabled."""
    with _policy_lock:
        return _policy


def _walk_and_redact(value: Any, policy: RedactionPolicy) -> Any:
    """Apply ``policy`` to every string nested inside ``value``.

    Walks dicts and lists recursively so JSON-serialized payloads (e.g.
    ``gen_ai.request.messages`` is a JSON string of a list-of-dicts that
    may itself be re-parsed into a dict) get their contents redacted.

    Strings: regex substitution.
    Dicts / lists: recurse.
    Everything else: pass through unchanged.
    """
    if isinstance(value, str):
        return policy.redact_string(value)
    if isinstance(value, dict):
        return {k: _walk_and_redact(v, policy) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_and_redact(v, policy) for v in value]
    return value


def redact_attributes(
    attrs: dict[str, Any],
    policy: RedactionPolicy | None = None,
) -> dict[str, Any]:
    """Return a copy of ``attrs`` with sensitive values redacted.

    Non-sensitive keys are left as-is by reference (shallow copy at
    the top level). Sensitive keys' values are deep-walked and
    redacted via :func:`_walk_and_redact`.

    If ``policy`` is ``None``, falls back to the installed module-level
    policy. If that's also ``None``, returns ``attrs`` unchanged
    (zero-overhead path).
    """
    effective = policy or get_redaction_policy()
    if effective is None or effective.mode == "off" or not effective._compiled:
        return attrs

    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if key in effective.apply_to_keys:
            out[key] = _walk_and_redact(value, effective)
        else:
            out[key] = value
    return out


def _capture_redact(attrs: dict[str, Any]) -> dict[str, Any]:
    """Capture-mode hook used by ``LocalStorageProcessor.on_end``.

    No-op when the installed policy's ``mode`` is not ``"capture"`` or
    ``"both"`` (or when there's no policy at all). Otherwise returns a
    redacted copy.
    """
    policy = get_redaction_policy()
    if policy is None or policy.mode not in ("capture", "both"):
        return attrs
    return redact_attributes(attrs, policy)


def _read_redact(attrs: dict[str, Any]) -> dict[str, Any]:
    """Read-mode hook used by the UI trace endpoints.

    No-op when the installed policy's ``mode`` is not ``"read"`` or
    ``"both"`` (or when there's no policy at all). Otherwise returns a
    redacted copy.
    """
    policy = get_redaction_policy()
    if policy is None or policy.mode not in ("read", "both"):
        return attrs
    return redact_attributes(attrs, policy)


def apply_export_policy(attrs: dict[str, Any]) -> dict[str, Any]:
    """Transform span attributes for EGRESS (control plane + OTel exporters).

    The single "on the way out" filter for the egress-gated privacy model
    (security_audit_2 N3 + N4). Local capture is always full fidelity; this is
    where content is held back from leaving the machine:

    1. **Payload gate** — when ``export_payloads_enabled()`` is False
       (``FASTAIAGENT_TRACE_PAYLOADS=0``), the payload-bearing keys
       (:data:`SENSITIVE_ATTR_KEYS`) are dropped entirely.
    2. **Redaction** — a capture/both-mode :class:`RedactionPolicy`, if
       installed, masks whatever payloads remain. This is what makes redaction
       reach third-party exporters (N4), not just ``local.db``.

    Returns a new dict; ``attrs`` is never mutated. Zero-copy fast path when
    payloads are exported and no policy is installed.
    """
    from fastaiagent.trace.span import export_payloads_enabled

    payloads_ok = export_payloads_enabled()
    policy = get_redaction_policy()
    redacting = policy is not None and policy.mode in ("capture", "both")
    if payloads_ok and not redacting:
        return attrs

    out = dict(attrs)
    if not payloads_ok:
        for key in SENSITIVE_ATTR_KEYS:
            out.pop(key, None)
    if redacting:
        out = redact_attributes(out, policy)
    return out
