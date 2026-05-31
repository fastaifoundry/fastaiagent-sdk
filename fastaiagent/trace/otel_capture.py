"""Capture and richly render spans from any in-process OTel instrumentor.

fastaiagent already persists *every* span that reaches its
:class:`~fastaiagent.trace.storage.LocalStorageProcessor` — but third-party
OpenTelemetry / OpenInference / OpenLLMetry instrumentors only reach it when
fastaiagent owns the global tracer provider, and their foreign attribute
conventions render thin in the Local UI.

:func:`enable_otel_capture` closes both gaps with **one opt-in call**:

1. **Ordering** — it adaptively joins whichever tracer provider is already
   global (attaching a :class:`LocalStorageProcessor` to a foreign SDK
   provider), or claims the global slot if none has, so foreign spans are
   captured regardless of import order.
2. **Display** — it flips on write-time normalization
   (:func:`fastaiagent.trace.storage.set_normalize_enabled`) so foreign keys are
   mapped onto canonical ``gen_ai.*`` keys as spans are stored.

``import fastaiagent`` behavior is unchanged until this is called. The call is
idempotent (mirrors the ``integrations.*.enable()`` pattern).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_enabled = False


def enable_otel_capture(*, framework: str | None = None) -> None:
    """Capture & richly render spans from any in-process OTel instrumentor.

    Call once at startup — ideally before enabling third-party instrumentors,
    though the adaptive-attach makes it robust either way. Idempotent.

    ``framework`` optionally overrides the framework badge stamped on root
    spans; when omitted the badge is derived per-span from the instrumentation
    scope name.
    """
    global _enabled

    from opentelemetry import trace as otel_trace

    from fastaiagent.trace import otel
    from fastaiagent.trace.storage import LocalStorageProcessor, set_normalize_enabled

    if _enabled:
        # Already on — just refresh the (possibly new) framework override.
        set_normalize_enabled(True, framework=framework)
        return

    current = otel_trace.get_tracer_provider()
    ours = otel._provider  # None until fastaiagent has initialized its provider

    if ours is not None and current is ours:
        # We already own the global provider; LocalStorageProcessor is attached.
        pass
    elif hasattr(current, "add_span_processor"):
        # A real SDK TracerProvider set by someone else — join it so its spans
        # also reach our store. (Native fastaiagent spans keep flowing through
        # our own provider, so this does not double-store.)
        try:
            current.add_span_processor(LocalStorageProcessor())
        except Exception:
            logger.warning(
                "enable_otel_capture: could not attach LocalStorageProcessor to the "
                "active tracer provider; foreign spans may not be captured.",
                exc_info=True,
            )
    else:
        # No real provider yet (no-op / proxy) — create and register ours.
        otel.get_tracer_provider()

    set_normalize_enabled(True, framework=framework)
    _enabled = True


def disable_otel_capture() -> None:
    """Stop foreign-span normalization (symmetry + test teardown).

    This flips normalization back off so subsequently stored foreign spans are
    persisted raw again. Note: OpenTelemetry has no API to *detach* a span
    processor, so a :class:`LocalStorageProcessor` previously attached to a
    foreign provider stays attached — capture continues, only the enrichment
    stops. Re-enabling later is cheap and idempotent.
    """
    global _enabled

    from fastaiagent.trace.storage import set_normalize_enabled

    if not _enabled:
        return
    set_normalize_enabled(False)
    _enabled = False
