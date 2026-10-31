"""An eval case links to the trace its agent call started (1.96.0, O1).

``evaluate()`` read the case's trace id only from an ``AgentResult``. A plain
callable running another framework's agent (traced by that framework's own
OpenTelemetry instrumentation) returns a string, so every case had
``trace_id=None`` and the Local UI could not open the run behind a verdict.
"""

from __future__ import annotations

from pathlib import Path

from opentelemetry import trace as otel_trace

from fastaiagent.eval import Dataset, evaluate
from fastaiagent.trace import otel


def _foreign_agent(question: str) -> str:
    """Stands in for a framework run: its own tracer, its own root span."""
    tracer = otel_trace.get_tracer("openinference.instrumentation.someframework")
    with tracer.start_as_current_span("SomeFramework.run") as root:
        with tracer.start_as_current_span("ChatModel"):
            pass
        _foreign_agent.started.append(format(root.get_span_context().trace_id, "032x"))
    return "Paris" if "France" in question else "?"


_foreign_agent.started = []  # type: ignore[attr-defined]


def _cases() -> Dataset:
    return Dataset.from_list(
        [
            {"input": "Capital of France?", "expected": "Paris"},
            {"input": "Capital of Peru?", "expected": "Lima"},
        ]
    )


def test_a_plain_callable_case_links_to_the_trace_it_started(isolated_local_db: Path) -> None:
    otel.reset()
    otel.get_tracer_provider()  # the SDK's provider, as enable_otel_capture() leaves it
    _foreign_agent.started.clear()  # type: ignore[attr-defined]
    try:
        results = evaluate(
            agent_fn=_foreign_agent, dataset=_cases(), scorers=["exact_match"], persist=False
        )
    finally:
        otel.reset()

    linked = [c.trace_id for c in results.cases]
    assert all(linked), f"cases without a trace: {linked}"
    assert sorted(linked) == sorted(_foreign_agent.started)  # type: ignore[attr-defined]
    assert len(set(linked)) == 2, "each case keeps its own trace"


def test_an_agent_results_own_trace_id_still_wins(isolated_local_db: Path) -> None:
    from fastaiagent import Agent
    from fastaiagent.testing import FunctionModel

    otel.reset()
    agent = Agent(name="native", llm=FunctionModel(lambda m: "Paris"))
    try:
        results = evaluate(
            agent_fn=agent.run, dataset=_cases(), scorers=["exact_match"], persist=False
        )
    finally:
        otel.reset()

    for case in results.cases:
        assert case.trace_id, "an AgentResult's trace id was lost"
    assert len({c.trace_id for c in results.cases}) == 2
