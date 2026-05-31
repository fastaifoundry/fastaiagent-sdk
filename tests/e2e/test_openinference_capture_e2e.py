"""Live drift check for ``enable_otel_capture()`` against a real instrumentor.

Installs nothing at runtime — but when the OpenInference OpenAI instrumentor and
an ``OPENAI_API_KEY`` are present, it makes a *real* OpenAI call with that
instrumentor active and ``enable_otel_capture()`` on, then asserts the captured
span in ``local.db`` carries the canonical ``gen_ai.*`` keys our UI reads. This
is the test that catches semantic-convention drift between the instrumentor's
emitted attributes and our normalizer's mapping table — something hand-crafted
attribute dicts cannot.

Gated (skipped) when the instrumentor or the API key is absent, so it is a clean
skip locally and only runs in the e2e quality-gate job.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.e2e

openai = pytest.importorskip("openai")
instrumentor_mod = pytest.importorskip(
    "openinference.instrumentation.openai",
    reason="openinference-instrumentation-openai not installed",
)

if not os.environ.get("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY required for live capture e2e", allow_module_level=True)


def test_live_openinference_span_is_normalized(tmp_path) -> None:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.util._once import Once

    from fastaiagent.trace import otel
    from fastaiagent.trace.otel_capture import disable_otel_capture, enable_otel_capture
    from fastaiagent.trace.storage import TraceStore

    db_path = str(tmp_path / "local.db")
    cfg = __import__("fastaiagent._internal.config", fromlist=["get_config"]).get_config()
    saved_db = cfg.trace_db_path
    cfg.trace_db_path = db_path

    # Fresh global provider so the instrumentor + our capture attach cleanly.
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel.reset()

    instrumentor = instrumentor_mod.OpenAIInstrumentor()
    provider = TracerProvider()
    otel_trace.set_tracer_provider(provider)
    instrumentor.instrument(tracer_provider=provider)
    enable_otel_capture()

    try:
        client = openai.OpenAI()
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=5,
        )
        provider.force_flush()

        store = TraceStore(db_path=db_path)
        try:
            traces = store.list_traces()
            assert traces, "no span captured from the live instrumentor"
            # Find the LLM span carrying a model (instrumentors may emit >1 span).
            llm_attrs = None
            for summ in traces:
                for sp in store.get_trace(summ.trace_id).spans:
                    if sp.attributes.get("gen_ai.request.model"):
                        llm_attrs = sp.attributes
                        break
                if llm_attrs:
                    break
            assert llm_attrs is not None, "no normalized gen_ai.request.model on any span"
            assert "gpt-4o-mini" in str(llm_attrs["gen_ai.request.model"])
            # Token counts normalized from the instrumentor's convention.
            assert int(llm_attrs.get("gen_ai.usage.input_tokens") or 0) > 0
            assert int(llm_attrs.get("gen_ai.usage.output_tokens") or 0) > 0
        finally:
            store.close()
    finally:
        disable_otel_capture()
        instrumentor.uninstrument()
        cfg.trace_db_path = saved_db
        otel_trace._TRACER_PROVIDER = None
        otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
        otel.reset()
