"""Async non-blocking trace export to FastAIAgent Platform."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class PlatformSpanExporter(SpanExporter):
    """Exports spans to the FastAIAgent Platform.

    Used with OTel BatchSpanProcessor which handles batching and
    calls export() from a background thread. On failure, logs a
    warning — the local SQLite store already has the data.
    """

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Send a batch of spans to the platform."""
        from fastaiagent.client import _connection

        if not _connection.is_connected:
            return SpanExportResult.SUCCESS

        trace_spans = _convert_spans(spans)
        if not trace_spans:
            return SpanExportResult.SUCCESS

        try:
            import httpx

            url = f"{_connection.target}/public/v1/traces/ingest"
            payload = {
                "project": _connection.project_id or _connection.project,
                "spans": trace_spans,
            }
            with httpx.Client(timeout=10) as client:
                client.post(url, json=payload, headers=_connection.headers)
        except Exception:
            logger.debug("Failed to export spans to platform", exc_info=True)

        # Always return SUCCESS — local SQLite has the trace
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _convert_spans(spans: Sequence[ReadableSpan]) -> list[dict[str, Any]]:
    """Convert OTel ReadableSpan objects to platform JSON format."""
    results = []
    for span in spans:
        ctx = span.get_span_context()
        trace_id = format(ctx.trace_id, "032x")
        span_id = format(ctx.span_id, "016x")

        parent_id = None
        if span.parent and span.parent.span_id:
            parent_id = format(span.parent.span_id, "016x")

        attrs = dict(span.attributes) if span.attributes else {}

        events = []
        if span.events:
            for e in span.events:
                raw_attrs = getattr(e, "attributes", None)
                event_attrs = (
                    {str(k): v for k, v in raw_attrs.items()} if raw_attrs else {}
                )
                events.append(
                    {
                        "name": e.name,
                        "timestamp": str(e.timestamp),
                        "attributes": event_attrs,
                    }
                )

        start_ns = span.start_time or 0
        end_ns = span.end_time or 0
        start_time = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat()
        end_time = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc).isoformat()

        status = "OK"
        if hasattr(span, "status") and span.status:
            status = (
                span.status.status_code.name
                if hasattr(span.status.status_code, "name")
                else str(span.status.status_code)
            )

        results.append(
            {
                "span_id": span_id,
                "trace_id": trace_id,
                "parent_span_id": parent_id,
                "name": span.name,
                "start_time": start_time,
                "end_time": end_time,
                "status": status,
                "attributes": attrs,
                "events": events,
            }
        )
    return results
