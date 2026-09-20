"""Seed a real foreign-instrumentor span for the Playwright capture spec.

Emits ONE OpenInference-convention span through the *production* write path
(``LocalStorageProcessor`` with ``enable_otel_capture`` normalization on) into
the given SQLite DB — no mock, no network. The span lands normalized, so the
Local UI renders model / tokens / cost / IO content for it.

Usage:  python scripts/seed_foreign_otel.py <db_path>
Prints the trace_id to stdout (consumed by the screenshot harness).
"""

from __future__ import annotations

import sys

from opentelemetry.sdk.trace import TracerProvider

from fastaiagent.trace.storage import (
    LocalStorageProcessor,
    TraceStore,
    set_normalize_enabled,
)

PROMPT = "Summarize the quarterly earnings report."
COMPLETION = "Revenue grew 12 percent quarter over quarter."
MODEL = "gpt-4o-mini"


def seed(db_path: str) -> str:
    set_normalize_enabled(True)
    provider = TracerProvider()
    processor = LocalStorageProcessor(db_path=db_path)
    provider.add_span_processor(processor)
    # Scope name → framework slug "openai" (openinference.* → "openai").
    tracer = provider.get_tracer("openinference.instrumentation.openai")
    with tracer.start_as_current_span("ChatOpenAI") as span:
        span.set_attribute("llm.model_name", MODEL)
        span.set_attribute("llm.token_count.prompt", 1200)
        span.set_attribute("llm.token_count.completion", 350)
        span.set_attribute("input.value", PROMPT)
        span.set_attribute("output.value", COMPLETION)
        span.set_attribute("openinference.span.kind", "LLM")
    processor.shutdown()
    set_normalize_enabled(False)

    store = TraceStore(db_path=db_path)
    try:
        trace_id = store.list_traces()[0].trace_id
    finally:
        store.close()
    return trace_id


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/seed_foreign_otel.py <db_path>", file=sys.stderr)
        raise SystemExit(2)
    print(seed(sys.argv[1]))
