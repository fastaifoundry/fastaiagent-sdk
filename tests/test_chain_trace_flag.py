"""``Chain.execute(trace=False)`` skips the chain root span.

Wires an in-memory OTel span exporter onto the singleton provider, runs
a chain twice — once with ``trace=True`` (default) and once with
``trace=False`` — and asserts the ``chain.<name>`` span is present in
the first run and absent in the second. Mirrors the
``Agent.run(trace=False)`` contract.
"""

from __future__ import annotations

from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from fastaiagent.agent import Agent
from fastaiagent.chain import Chain
from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.trace.otel import get_tracer_provider


class MockLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(provider="mock", model="mock")

    async def acomplete(self, messages, tools=None, **kwargs):
        return LLMResponse(content="ok", finish_reason="stop")


def _build_chain() -> Chain:
    chain = Chain("trace-flag-test", checkpoint_enabled=False)
    chain.add_node("only", agent=Agent(name="only", llm=MockLLMClient(), system_prompt="x"))
    return chain


def test_chain_trace_flag_gates_root_span():
    exporter = InMemorySpanExporter()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))

    chain = _build_chain()

    # trace=True (default) — span present.
    exporter.clear()
    chain.execute({"input": "hi"})
    chain_spans_traced = [
        s.name for s in exporter.get_finished_spans() if s.name.startswith("chain.")
    ]
    assert "chain.trace-flag-test" in chain_spans_traced

    # trace=False — no chain.* span.
    exporter.clear()
    chain.execute({"input": "hi"}, trace=False)
    chain_spans_untraced = [
        s.name for s in exporter.get_finished_spans() if s.name.startswith("chain.")
    ]
    assert chain_spans_untraced == []
