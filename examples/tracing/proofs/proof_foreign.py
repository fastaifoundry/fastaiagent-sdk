"""Proof for "5 · Between frameworks" on docs/tracing/trace-boundaries.md.

Offline: no API key, a throwaway local.db, and a plain OpenTelemetry tracer
emitting the attributes an OpenInference instrumentor emits (the real OTel SDK,
not a mock).

With enable_otel_capture(), a foreign span is stored in the same table as a
native one, and its OpenInference keys are mapped onto the canonical gen_ai.*
and runner.type keys at write time. Native spans are not touched.
"""

import os
import tempfile

os.environ["FASTAIAGENT_LOCAL_DB"] = os.path.join(tempfile.mkdtemp(), "local.db")

from opentelemetry import trace as otel  # noqa: E402

import fastaiagent as fa  # noqa: E402
from fastaiagent.testing import FunctionModel  # noqa: E402
from fastaiagent.trace import TraceStore, enable_otel_capture  # noqa: E402

enable_otel_capture()

# What openinference-instrumentation-openai puts on a chat-completion span.
tracer = otel.get_tracer_provider().get_tracer("openinference.instrumentation.openai")
with tracer.start_as_current_span("ChatCompletion") as span:
    span.set_attribute("openinference.span.kind", "LLM")
    span.set_attribute("llm.model_name", "gpt-4.1-mini")
    span.set_attribute("llm.token_count.prompt", 12)
    span.set_attribute("llm.token_count.completion", 8)
    span.set_attribute("input.value", "Where is order 1042?")
    span.set_attribute("output.value", "Order 1042 has shipped.")
foreign_trace = format(span.get_span_context().trace_id, "032x")

native = fa.Agent(name="support", llm=FunctionModel(lambda m: "Order 1042 has shipped."))
native_trace = native.run("Where is order 1042?").trace_id

store = TraceStore()
KEYS = (
    "gen_ai.request.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.response.content",
    "fastaiagent.runner.type",
    "fastaiagent.framework",
)
for label, tid in (("foreign (OpenInference)", foreign_trace), ("native", native_trace)):
    trace = store.get_trace(tid)
    print(f"{label}: {[s.name for s in trace.spans]}")
    llm = next(s for s in trace.spans if "llm" in s.name.lower() or s.name == "ChatCompletion")
    print(f"  the model-call span, {llm.name}:")
    for k in KEYS:
        print(f"    {k:<28} = {llm.attributes.get(k)!r}")
    print(f"    OpenInference keys kept: {'llm.model_name' in llm.attributes}")
summaries = store.list_traces()
print(f"list_traces() shows both: {sorted(t.name for t in summaries)}")
