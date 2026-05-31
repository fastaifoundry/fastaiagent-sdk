"""End-to-end capture tests for foreign-instrumentor spans (no mocks).

Real OpenTelemetry spans carrying OpenInference-style attributes are emitted
through a real ``TracerProvider`` + ``LocalStorageProcessor`` into a real
SQLite DB. We then read the stored span (and the FTS index) back and assert:

* with normalization ON, foreign spans gain the canonical keys the UI reads;
* with normalization OFF (default), spans are stored byte-for-byte raw — the
  proof that the feature is non-breaking.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider

from fastaiagent.trace.storage import LocalStorageProcessor, TraceStore, set_normalize_enabled
from fastaiagent.ui.db import init_local_db


@pytest.fixture(autouse=True)
def _reset_flag():
    """Keep the global normalization flag isolated per test."""
    set_normalize_enabled(False)
    yield
    set_normalize_enabled(False)


def _emit_openinference_span(db_path: str) -> None:
    """Emit one real root span using OpenInference attribute conventions."""
    provider = TracerProvider()
    processor = LocalStorageProcessor(db_path=db_path)
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("openinference.instrumentation.openai")
    with tracer.start_as_current_span("ChatOpenAI") as span:
        span.set_attribute("llm.model_name", "gpt-4o-mini")
        span.set_attribute("llm.token_count.prompt", 11)
        span.set_attribute("llm.token_count.completion", 7)
        span.set_attribute("input.value", "What is 2+2?")
        span.set_attribute("output.value", "4")
        span.set_attribute("openinference.span.kind", "LLM")
    processor.shutdown()


def _only_span_attrs(db_path: str) -> dict:
    store = TraceStore(db_path=db_path)
    try:
        traces = store.list_traces()
        assert len(traces) == 1
        trace = store.get_trace(traces[0].trace_id)
        assert len(trace.spans) == 1
        return trace.spans[0].attributes
    finally:
        store.close()


class TestNormalizationOn:
    def test_foreign_span_renders_rich(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "local.db")
        set_normalize_enabled(True)
        _emit_openinference_span(db_path)

        attrs = _only_span_attrs(db_path)
        assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
        assert int(attrs["gen_ai.usage.input_tokens"]) == 11
        assert int(attrs["gen_ai.usage.output_tokens"]) == 7
        assert attrs["gen_ai.prompt"] == "What is 2+2?"
        assert attrs["gen_ai.completion"] == "4"
        assert attrs["fastaiagent.runner.type"] == "llm"
        assert attrs["fastaiagent.framework"] == "openai"
        # Originals preserved.
        assert attrs["llm.model_name"] == "gpt-4o-mini"

    def test_fts_index_is_populated(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "local.db")
        set_normalize_enabled(True)
        _emit_openinference_span(db_path)

        db = init_local_db(db_path)
        try:
            rows = db.fetchall("SELECT input_text, output_text FROM span_fts")
        finally:
            db.close()
        assert any(r["input_text"] == "What is 2+2?" for r in rows)
        assert any(r["output_text"] == "4" for r in rows)


class TestNormalizationOff:
    def test_default_off_stores_raw_unchanged(self, tmp_path: Path) -> None:
        """The non-breaking guarantee: with the flag off, nothing is added."""
        db_path = str(tmp_path / "local.db")
        # Flag is False via the autouse fixture — emit without enabling.
        _emit_openinference_span(db_path)

        attrs = _only_span_attrs(db_path)
        # Foreign keys are stored verbatim...
        assert attrs["llm.model_name"] == "gpt-4o-mini"
        assert attrs["input.value"] == "What is 2+2?"
        # ...and NO canonical keys are synthesized.
        assert "gen_ai.request.model" not in attrs
        assert "gen_ai.prompt" not in attrs
        assert "fastaiagent.framework" not in attrs
