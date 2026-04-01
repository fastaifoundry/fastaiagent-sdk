"""Tests for fastaiagent.trace module."""

from __future__ import annotations

import pytest

from fastaiagent.trace.otel import get_tracer, get_tracer_provider, reset
from fastaiagent.trace.span import set_fastai_attributes, set_genai_attributes
from fastaiagent.trace.storage import LocalStorageProcessor, TraceStore
from fastaiagent.trace.tracer import trace_context


@pytest.fixture(autouse=True)
def clean_tracer():
    """Reset tracer between tests."""
    reset()
    yield
    reset()


class TestOTelSetup:
    def test_get_tracer_provider(self):
        provider = get_tracer_provider()
        assert provider is not None

    def test_get_tracer(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_provider_is_singleton(self):
        p1 = get_tracer_provider()
        p2 = get_tracer_provider()
        assert p1 is p2

    def test_reset_clears_provider(self):
        p1 = get_tracer_provider()
        reset()
        p2 = get_tracer_provider()
        assert p1 is not p2


class TestTraceContext:
    def test_context_manager(self):
        with trace_context("test-op") as span:
            assert span is not None
            span.set_attribute("test.key", "value")

    def test_nested_spans(self):
        with trace_context("parent"):
            with trace_context("child") as child_span:
                assert child_span is not None


class TestLocalStorage:
    def test_processor_writes_spans(self, temp_dir):
        db_path = str(temp_dir / "traces.db")
        processor = LocalStorageProcessor(db_path=db_path)

        # Create a span using OTel
        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("key", "value")

        # Query the stored span
        store = TraceStore(db_path=db_path)
        traces = store.list_traces()
        assert len(traces) >= 1

        processor.shutdown()
        store.close()

    def test_trace_store_get_trace(self, temp_dir):
        db_path = str(temp_dir / "traces.db")
        processor = LocalStorageProcessor(db_path=db_path)

        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("parent-span") as parent:
            trace_id = format(parent.get_span_context().trace_id, "032x")
            with tracer.start_as_current_span("child-span") as child:
                child.set_attribute("test", "data")

        store = TraceStore(db_path=db_path)
        trace_data = store.get_trace(trace_id)
        assert trace_data.trace_id == trace_id
        assert len(trace_data.spans) == 2

        processor.shutdown()
        store.close()

    def test_trace_store_search(self, temp_dir):
        db_path = str(temp_dir / "traces.db")
        processor = LocalStorageProcessor(db_path=db_path)

        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("unique-searchable-name"):
            pass

        store = TraceStore(db_path=db_path)
        results = store.search("unique-searchable")
        assert len(results) >= 1
        assert "unique-searchable" in results[0].name

        processor.shutdown()
        store.close()

    def test_trace_store_export(self, temp_dir):
        db_path = str(temp_dir / "traces.db")
        processor = LocalStorageProcessor(db_path=db_path)

        from opentelemetry.sdk.trace import TracerProvider

        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("export-test") as span:
            trace_id = format(span.get_span_context().trace_id, "032x")

        store = TraceStore(db_path=db_path)
        exported = store.export(trace_id)
        assert "export-test" in exported
        assert trace_id in exported

        processor.shutdown()
        store.close()

    def test_trace_not_found(self, temp_dir):
        db_path = str(temp_dir / "traces.db")
        processor = LocalStorageProcessor(db_path=db_path)
        processor._get_db()  # init schema

        store = TraceStore(db_path=db_path)
        with pytest.raises(Exception, match="not found"):
            store.get_trace("nonexistent")

        processor.shutdown()
        store.close()


class TestSpanHelpers:
    def test_set_genai_attributes(self):
        from unittest.mock import MagicMock

        span = MagicMock()
        set_genai_attributes(
            span, system="openai", model="gpt-4o", input_tokens=100, output_tokens=50
        )
        calls = span.set_attribute.call_args_list
        attr_dict = {call[0][0]: call[0][1] for call in calls}
        assert attr_dict["gen_ai.system"] == "openai"
        assert attr_dict["gen_ai.request.model"] == "gpt-4o"
        assert attr_dict["gen_ai.usage.input_tokens"] == 100

    def test_set_fastai_attributes(self):
        from unittest.mock import MagicMock

        span = MagicMock()
        set_fastai_attributes(span, **{"agent.name": "test-agent", "tool.name": "search"})
        calls = span.set_attribute.call_args_list
        attr_dict = {call[0][0]: call[0][1] for call in calls}
        assert attr_dict["fastai.agent.name"] == "test-agent"
        assert attr_dict["fastai.tool.name"] == "search"
