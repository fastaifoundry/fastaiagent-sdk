"""Shared setup for the durability proofs.

Imported before ``fastaiagent`` so every proof runs in a scratch directory of its
own: a fresh ``local.db`` for the traces, no connection to a
control plane, and payloads recorded so the spans carry the system prompt.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

SCRATCH = Path(tempfile.mkdtemp(prefix="fastaiagent-durability-proof-"))
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


CKPT_DB = SCRATCH / "checkpoints.db"


def checkpointer():
    """A SQLite checkpointer on the scratch file (a child process gets the same path via env)."""
    from fastaiagent import SQLiteCheckpointer

    return SQLiteCheckpointer(db_path=str(os.environ.get("PROOF_CKPT_DB", CKPT_DB)))


def rows(cp, execution_id: str) -> list[str]:
    """One line per checkpoint, in the order they were written."""
    out = []
    for c in cp.list(execution_id):
        out.append(f"#{c.node_index} {c.node_id:<22} step={c.step_type or '-':<10} status={c.status}")
    return out
