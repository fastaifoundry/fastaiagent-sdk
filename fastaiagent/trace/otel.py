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


def add_exporter(exporter: SpanExporter) -> None:
    """Add any OTel-compatible exporter (Datadog, Jaeger, etc.)."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))


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
