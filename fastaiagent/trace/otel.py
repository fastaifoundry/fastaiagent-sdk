"""OTel tracer provider setup and management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter

_provider: Any = None

# Whether the SDK's provider should try to become OTel's *global* provider.
# Turned off by ``connect(export_traces=False)`` so a foreign runtime that owns
# its own exporter keeps its global provider — one exporter per process.
_claim_global: bool = True


def suppress_global_provider() -> None:
    """Stop :func:`get_tracer_provider` from calling ``set_tracer_provider``.

    For a process where another framework (LangChain, CrewAI, LlamaIndex…)
    already owns the global tracer provider and its exporter. The SDK still
    creates its own provider for its own spans — it just doesn't try to claim
    the global slot. Call this *before* the provider is first created; OTel's
    global set is first-wins and cannot be undone afterwards.
    """
    global _claim_global
    _claim_global = False


def get_tracer_provider() -> Any:
    """Get or create the OTel TracerProvider singleton."""
    global _provider
    if _provider is None:
        from opentelemetry.sdk.trace import TracerProvider

        from fastaiagent.trace.storage import LocalStorageProcessor

        _provider = TracerProvider()
        _provider.add_span_processor(LocalStorageProcessor())

        if _claim_global:
            from opentelemetry import trace as otel_trace

            otel_trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str = "fastaiagent") -> Any:
    """Get a tracer instance."""
    return get_tracer_provider().get_tracer(name)


def _filtered_status(span: Any) -> Any:
    """Apply the egress payload gate to a span's status description, or ``None``.

    The third channel, and the last one. A span's ``Status`` carries a free-text
    *description*, and for a guardrail span that description **is** the rule's
    failure message (``trace.span``: ``Status(StatusCode.ERROR, message)``) — for
    an ``llm_judge`` rule the judge's raw reply, for ``groundedness`` the claims it
    quoted out of the model's answer.

    ``_rebuild_span`` copies ``status`` by reference, so filtering attributes and
    events still let that text reach a Datadog or Jaeger exporter with
    ``FASTAIAGENT_TRACE_PAYLOADS=0`` set. The SDK→plane path escapes this by
    accident of its wire model, which carries ``status`` as a bare code string and
    discards the description.

    The **code** is structural and always survives — an operator with payloads off
    still sees that the span errored, and the guardrail attributes still say which
    rule and that it blocked. Only the description is withheld.
    """
    from opentelemetry.trace import Status

    from fastaiagent.trace.redaction import _walk_and_redact, get_redaction_policy
    from fastaiagent.trace.span import export_payloads_enabled

    status = getattr(span, "status", None)
    description = getattr(status, "description", None)
    if status is None or not description:
        return None

    if not export_payloads_enabled():
        return Status(status.status_code)

    policy = get_redaction_policy()
    if policy is not None and policy.mode in ("capture", "both") and policy._compiled:
        masked = _walk_and_redact(description, policy)
        if masked != description:
            return Status(status.status_code, masked)
    return None


def _rebuild_span(
    span: Any, attributes: dict[str, Any], events: Any = None, status: Any = None
) -> Any:
    """Return a ``ReadableSpan`` identical to ``span`` but with new attributes/events/status.

    ``ReadableSpan.attributes`` is immutable, so redacting/stripping for export
    means reconstructing the span. All other fields are copied by reference.
    """
    from opentelemetry.sdk.trace import ReadableSpan

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=span.events if events is None else events,
        links=span.links,
        kind=span.kind,
        instrumentation_scope=span.instrumentation_scope,
        status=span.status if status is None else status,
        start_time=span.start_time,
        end_time=span.end_time,
    )


def _filtered_events(span: Any) -> Any:
    """Apply the egress payload gate to a span's events, or ``None`` if unchanged.

    The dict-shaped sibling lives in ``redaction.apply_event_export_policy``; OTel
    spans carry real ``Event`` objects, so this rebuilds them while borrowing the
    same key registry. Returning ``None`` keeps the zero-copy fast path — the
    common case is a span with no events at all.
    """
    from opentelemetry.sdk.trace import Event

    from fastaiagent.trace.redaction import (
        SENSITIVE_EVENT_ATTR_KEYS,
        _walk_and_redact,
        get_redaction_policy,
    )
    from fastaiagent.trace.span import export_payloads_enabled

    events = getattr(span, "events", None)
    if not events:
        return None

    payloads_ok = export_payloads_enabled()
    policy = get_redaction_policy()
    masker = (
        policy
        if (policy is not None and policy.mode in ("capture", "both") and policy._compiled)
        else None
    )
    if payloads_ok and masker is None:
        return None

    rebuilt = []
    changed = False
    for event in events:
        attrs = dict(getattr(event, "attributes", None) or {})
        for key in SENSITIVE_EVENT_ATTR_KEYS:
            if key not in attrs:
                continue
            if not payloads_ok:
                attrs.pop(key, None)
                changed = True
            elif masker is not None:
                attrs[key] = _walk_and_redact(attrs[key], masker)
                changed = True
        rebuilt.append(Event(name=event.name, attributes=attrs, timestamp=event.timestamp))
    return rebuilt if changed else None


class _EgressFilteredExporter:
    """Wraps a user exporter so spans are payload-filtered/redacted on the way out.

    security_audit_2 N4: previously redaction was applied only to the copy that
    ``LocalStorageProcessor`` writes to SQLite, so exporters registered here
    received raw, unredacted spans — contradicting the documented behavior. This
    runs every span through :func:`apply_export_policy` (payload gate + redaction)
    before delegating, so third-party exporters honor the same egress rules as
    the control-plane exporter.
    """

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Any) -> Any:
        from fastaiagent.trace.redaction import apply_export_policy

        filtered = []
        for span in spans:
            try:
                attrs = dict(span.attributes or {})
                new_attrs = apply_export_policy(attrs)
                new_events = _filtered_events(span)
                new_status = _filtered_status(span)
                unchanged = new_attrs == attrs and new_events is None and new_status is None
                filtered.append(
                    span if unchanged else _rebuild_span(span, new_attrs, new_events, new_status)
                )
            except Exception:
                # Fail closed: if we can't rebuild a filtered span, drop it from
                # the export batch rather than leak an unfiltered one. It remains
                # in local.db regardless.
                logger.debug("Egress filter failed for a span; dropping it", exc_info=True)
        return self._inner.export(filtered)

    def shutdown(self) -> Any:
        return self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> Any:
        return self._inner.force_flush(timeout_millis)


def add_exporter(exporter: SpanExporter) -> None:
    """Add any OTel-compatible exporter (Datadog, Jaeger, etc.).

    The exporter is wrapped so payloads are stripped/redacted for export per the
    egress-gated privacy model (N3/N4) — see :class:`_EgressFilteredExporter`.
    """
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    wrapped = _EgressFilteredExporter(exporter)
    get_tracer_provider().add_span_processor(BatchSpanProcessor(wrapped))


def reset() -> None:
    """Reset the tracer provider (for testing)."""
    global _provider, _claim_global
    _claim_global = True
    if _provider is not None:
        try:
            _provider.shutdown()
        except Exception:
            logger.debug("Failed to shutdown tracer provider", exc_info=True)
    _provider = None
