"""OTel exporter helpers."""

from __future__ import annotations

from typing import Any


def create_otlp_exporter(
    endpoint: str = "http://localhost:4318/v1/traces",
    headers: dict[str, str] | None = None,
    protocol: str = "http",
) -> Any:
    """Create an OTLP exporter.

    Requires: pip install fastaiagent[otel-export]
    """
    try:
        if protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

        return OTLPSpanExporter(endpoint=endpoint, headers=headers or {})
    except ImportError:
        raise ImportError(
            "OTLP exporter requires additional dependencies. "
            "Install with: pip install fastaiagent[otel-export]"
        )
