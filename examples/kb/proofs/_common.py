"""Shared setup for the knowledge-base proofs.

Imported before ``fastaiagent`` so every proof runs in a scratch directory of its
own: a fresh ``local.db`` for the traces, a scratch folder for the KB, no connection to a
control plane, and payloads recorded so the spans carry the system prompt.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

SCRATCH = Path(tempfile.mkdtemp(prefix="fastaiagent-kb-proof-"))
os.environ["FASTAIAGENT_LOCAL_DB"] = str(SCRATCH / "local.db")
os.environ.setdefault("FASTAIAGENT_TRACE_PAYLOADS", "1")
# A shell may point these at a control plane; the proofs must stay local.
os.environ.pop("FASTAIAGENT_API_KEY", None)
os.environ.pop("FASTAIAGENT_TARGET", None)


def heading(text: str) -> None:
    print(f"\n── {text} " + "─" * max(4, 72 - len(text)))


class SpanCollector:
    """Every span the SDK emits in this process, in order. Real OTel, no mocks."""

    def __init__(self) -> None:
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )

        from fastaiagent.trace.otel import get_tracer_provider

        collector = self

        class _Exporter(SpanExporter):
            def export(self, spans):  # type: ignore[override]
                for s in spans:
                    collector.spans.append((s.name, dict(s.attributes or {})))
                return SpanExportResult.SUCCESS

            def shutdown(self) -> None:  # pragma: no cover
                pass

        self.spans: list[tuple[str, dict[str, Any]]] = []
        get_tracer_provider().add_span_processor(SimpleSpanProcessor(_Exporter()))

    def llm(self) -> list[dict[str, Any]]:
        return [a for n, a in self.spans if n.startswith("llm.")]

    def reset(self) -> None:
        self.spans.clear()


KB_DIR = SCRATCH / "kb"


def embedder():
    """The local FastEmbed model (``fastaiagent[kb]``); exits with a message without it."""
    try:
        from fastaiagent.kb.embedding import FastEmbedEmbedder

        return FastEmbedEmbedder()
    except ImportError:
        raise SystemExit('This proof needs the local embedder: pip install "fastaiagent[kb]"')


class CountingEmbedder:
    """A real embedder that counts how many texts it was asked to embed."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.texts = 0
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        return self.inner.embed(texts)


def retrieval_spans(col: SpanCollector) -> list[dict[str, Any]]:
    return [a for n, a in col.spans if n.startswith("retrieval.")]
