"""Tracing context manager for creating OTel spans."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


@contextmanager
def trace_context(name: str, **attributes: Any) -> Generator[Any, None, None]:
    """Context manager that creates an OTel span.

    Example:
        with trace_context("my-operation") as span:
            # do work
            span.set_attribute("custom.key", "value")
    """
    from fastaiagent.trace.otel import get_tracer
    from fastaiagent.trace.span import set_span_attributes

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            set_span_attributes(span, **attributes)
        yield span
