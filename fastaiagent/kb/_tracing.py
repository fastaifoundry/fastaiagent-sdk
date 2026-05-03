"""OTel tracing helper for KB.search entrypoints.

Opens a ``retrieval.<kb_name>`` span with attributes matching our standard
observability namespace (``fastaiagent.runner.type = retrieval`` + a set of
``retrieval.*`` attributes). A ``list[SearchResult]`` passes through the
helper and its length + top doc ids are recorded on span close.

Query text + doc ids are payload-gated — they respect
``FASTAIAGENT_TRACE_PAYLOADS``. Structural attributes (backend,
search_type, top_k, latency, result_count) are always captured.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from fastaiagent.trace.span import trace_payloads_enabled

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def retrieval_span(
    *,
    kb_name: str,
    backend: str,
    search_type: str | None,
    query: str,
    top_k: int,
) -> Iterator[_SpanHandle]:
    """Open a retrieval span and yield a handle the caller closes with results."""
    from fastaiagent.trace.otel import get_tracer

    tracer = get_tracer("fastaiagent.kb")
    start = time.monotonic()
    handle = _SpanHandle()
    with tracer.start_as_current_span(f"retrieval.{kb_name}") as span:
        span.set_attribute("fastaiagent.runner.type", "retrieval")
        span.set_attribute("retrieval.kb_name", kb_name)
        span.set_attribute("retrieval.backend", backend)
        span.set_attribute("retrieval.top_k", top_k)
        if search_type:
            span.set_attribute("retrieval.search_type", search_type)
        if trace_payloads_enabled():
            span.set_attribute("retrieval.query", query)

        try:
            yield handle
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            span.set_attribute("retrieval.latency_ms", latency_ms)
            span.set_attribute("retrieval.result_count", handle.result_count)
            if trace_payloads_enabled() and handle.doc_ids:
                try:
                    span.set_attribute(
                        "retrieval.doc_ids",
                        json.dumps(handle.doc_ids[:20]),
                    )
                except (TypeError, ValueError):
                    logger.debug("Failed to serialize retrieval doc_ids for trace", exc_info=True)


class _SpanHandle:
    """Callers hand back the search results through this handle so the span
    helper can set ``retrieval.result_count`` / ``retrieval.doc_ids`` on close.
    """

    def __init__(self) -> None:
        self.result_count: int = 0
        self.doc_ids: list[str] = []

    def record(self, results: list[Any]) -> None:
        """Record the search results.

        ``retrieval.doc_ids`` prefers ``chunk.metadata['doc_id']`` (the
        user-level document identifier loaders typically set) and falls
        back to ``chunk.id`` (the canonical SDK chunk id).
        """
        self.result_count = len(results)
        ids: list[str] = []
        for r in results:
            chunk = getattr(r, "chunk", None)
            if chunk is None:
                continue
            metadata = getattr(chunk, "metadata", None) or {}
            identifier = (
                metadata.get("doc_id")
                if isinstance(metadata, dict)
                else None
            ) or getattr(chunk, "id", None) or getattr(chunk, "doc_id", None)
            if identifier is not None:
                ids.append(str(identifier))
        self.doc_ids = ids
