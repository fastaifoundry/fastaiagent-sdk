"""The guardrail span wraps the check (1.89.0, B1).

The span was opened *after* the check, so its duration was about zero (the
plane records that as the execution time of every SDK guardrail) and a judge's
LLM call ran beside it, as a sibling, instead of inside it. Keys, values and
status are unchanged: ``tests/test_guardrail_span_wire_pin.py`` pins those.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition, GuardrailType


class _Collector(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans):  # type: ignore[override]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


@pytest.fixture
def collector(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    otel.reset()
    col = _Collector()
    otel.get_tracer_provider().add_span_processor(SimpleSpanProcessor(col))
    yield col
    otel.reset()
    reset_config()


def _slow(_text: str) -> bool:
    time.sleep(0.06)
    return True


@pytest.mark.parametrize("blocking", [True, False], ids=["blocking", "observer"])
def test_the_span_lasts_as_long_as_the_check(collector: Any, blocking: bool) -> None:
    rule = Guardrail(
        name="slow",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.output,
        fn=_slow,
        blocking=blocking,
    )
    asyncio.run(execute_guardrails([rule], "x", GuardrailPosition.output))
    (span,) = [s for s in collector.spans if s.name == "guardrail.slow"]
    duration_ms = (span.end_time - span.start_time) / 1e6
    assert duration_ms >= 50, f"the span lasted {duration_ms:.1f} ms for a 60 ms check"


@pytest.mark.parametrize("blocking", [True, False], ids=["blocking", "observer"])
def test_a_broken_tracer_never_breaks_the_check(
    monkeypatch: pytest.MonkeyPatch, blocking: bool
) -> None:
    """The span now opens before the check, so a tracer that fails to open one
    must not take the check (and the run) with it."""
    from fastaiagent.trace import otel as otel_mod

    class _BrokenTracer:
        def start_as_current_span(self, name: str, **_kwargs: Any) -> Any:
            raise RuntimeError("exporter is down")

    monkeypatch.setattr(otel_mod, "get_tracer", lambda *_a, **_k: _BrokenTracer())
    rule = Guardrail(
        name="ok",
        guardrail_type=GuardrailType.regex,
        position=GuardrailPosition.output,
        config={"pattern": "nope"},
        blocking=blocking,
    )
    outcome = asyncio.run(execute_guardrails([rule], "clean", GuardrailPosition.output))
    assert outcome.results[0].passed is True


def test_a_model_call_inside_the_check_is_a_child_of_its_guardrail_span(collector: Any) -> None:
    """What a judge does: call a model from inside the check."""
    from fastaiagent.llm import UserMessage
    from fastaiagent.testing import FunctionModel

    judge = FunctionModel(lambda _messages: "PASS")

    def polite(text: str) -> bool:
        return judge.complete([UserMessage(f"Is this polite? {text}")]).content == "PASS"

    rule = Guardrail(
        name="polite",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.output,
        fn=polite,
    )
    asyncio.run(execute_guardrails([rule], "Thanks for asking!", GuardrailPosition.output))
    guard = next(s for s in collector.spans if s.name == "guardrail.polite")
    llm = [s for s in collector.spans if s.name.startswith("llm.")]
    assert llm, "the judge made no traced call"
    assert all(s.parent is not None and s.parent.span_id == guard.context.span_id for s in llm), (
        "the judge's LLM call ran beside the guardrail span instead of inside it"
    )
