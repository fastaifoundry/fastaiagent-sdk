"""Provider-ordering test for enable_otel_capture (adaptive attach).

The hard case: a third-party SDK ``TracerProvider`` wins the global slot
*before* fastaiagent initializes. ``enable_otel_capture()`` must then attach a
``LocalStorageProcessor`` to that foreign provider so its spans still reach the
local store. This exercises the real global OTel machinery (no mocks); it
brackets the global provider state with a reset so it does not leak to other
tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

import fastaiagent.trace.otel_capture as otel_capture
from fastaiagent._internal import config
from fastaiagent.trace import otel
from fastaiagent.trace.otel_capture import disable_otel_capture, enable_otel_capture
from fastaiagent.trace.storage import TraceStore


def _reset_global_provider() -> None:
    """Clear the OTel global tracer provider + its set-once guard."""
    from opentelemetry.util._once import Once

    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel.reset()
    otel_capture._enabled = False


@pytest.fixture
def isolated_global(tmp_path: Path):
    """Isolate global provider + default DB path for one ordering test.

    ``enable_otel_capture`` builds a ``LocalStorageProcessor()`` with no path, so
    it resolves the DB from the config singleton — point that at a tmp DB by
    mutating the singleton directly (there is no ``set_trace_db_path`` helper),
    and restore it afterwards.
    """
    cfg = config.get_config()
    saved_trace_db = cfg.trace_db_path
    saved_local_db = cfg.local_db_path
    db_path = str(tmp_path / "local.db")
    cfg.trace_db_path = db_path
    _reset_global_provider()
    try:
        yield db_path
    finally:
        disable_otel_capture()
        _reset_global_provider()
        cfg.trace_db_path = saved_trace_db
        cfg.local_db_path = saved_local_db


def test_foreign_provider_set_first_is_joined(isolated_global) -> None:
    db_path = isolated_global

    # 1) A foreign SDK provider wins the global slot FIRST.
    foreign = TracerProvider()
    otel_trace.set_tracer_provider(foreign)
    assert otel_trace.get_tracer_provider() is foreign

    # 2) Now fastaiagent opts in — it must adaptively join the foreign provider.
    enable_otel_capture()

    # 3) A span emitted through the (foreign) global provider must land in our
    #    store, richly normalized.
    tracer = otel_trace.get_tracer("openinference.instrumentation.langchain")
    with tracer.start_as_current_span("LangChainRun") as span:
        span.set_attribute("llm.model_name", "gpt-4o-mini")
        span.set_attribute("llm.token_count.prompt", 9)
        span.set_attribute("llm.token_count.completion", 4)
        span.set_attribute("input.value", "ping")
        span.set_attribute("output.value", "pong")
        span.set_attribute("openinference.span.kind", "CHAIN")

    foreign.force_flush()

    store = TraceStore(db_path=db_path)
    try:
        traces = store.list_traces()
        assert len(traces) == 1, "foreign span did not reach the local store"
        attrs = store.get_trace(traces[0].trace_id).spans[0].attributes
    finally:
        store.close()

    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.prompt"] == "ping"
    assert attrs["gen_ai.completion"] == "pong"
    assert attrs["fastaiagent.runner.type"] == "chain"
    assert attrs["fastaiagent.framework"] == "langchain"


def test_enable_is_idempotent(isolated_global) -> None:
    db_path = isolated_global
    foreign = TracerProvider()
    otel_trace.set_tracer_provider(foreign)

    enable_otel_capture()
    enable_otel_capture()  # second call must be a safe no-op

    # Exactly one LocalStorageProcessor should have been attached, so a single
    # emitted span is stored once (not duplicated).
    tracer = otel_trace.get_tracer("openinference.instrumentation.openai")
    with tracer.start_as_current_span("ChatOpenAI") as span:
        span.set_attribute("llm.model_name", "gpt-4o-mini")
        span.set_attribute("input.value", "hi")

    foreign.force_flush()

    store = TraceStore(db_path=db_path)
    try:
        trace = store.get_trace(store.list_traces()[0].trace_id)
        assert len(trace.spans) == 1
    finally:
        store.close()
