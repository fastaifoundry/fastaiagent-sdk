"""Unit tests for the OpenInference telemetry standardization (v1.47.0).

Covers the wire contract every runtime converges on:
* ``set_evaluation_attributes`` — the ``EVALUATOR`` / ``evaluation.*`` shape,
  including the 0..1 clamp-and-warn on a mis-scaled score.
* ``emit_guardrail`` / ``emit_evaluation`` — one-call emitters that use the
  *caller's* tracer and never touch the SDK's provider (one exporter per
  process).
* ``connect(export_traces=False)`` — connected for policy/scorers, but no
  platform span exporter and no claim on OTel's global provider.
* The public primitive surface (``run_guardrail`` & friends) importable from
  the top-level package.

Real OpenTelemetry spans through a real in-memory exporter; no mocks. The
global-provider assertions run in a fresh subprocess because OTel's global set
is first-wins and cannot be undone once another test has claimed it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from fastaiagent.trace.span import (
    emit_evaluation,
    emit_guardrail,
    set_evaluation_attributes,
)


class _Collector(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[tuple[str, dict, str]] = []

    def export(self, spans):  # type: ignore[override]
        for s in spans:
            self.spans.append((s.name, dict(s.attributes), s.status.status_code.name))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover
        pass


@pytest.fixture()
def foreign_tracer() -> tuple[object, _Collector]:
    """A tracer on a provider the SDK knows nothing about.

    Stands in for a LangChain/CrewAI runtime's own provider+exporter: nothing
    here is registered globally, and the SDK's provider is never involved.
    """
    col = _Collector()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(col))
    return provider.get_tracer("foreign.runtime"), col


# --------------------------------------------------------------------------- #
# Task 2 — set_evaluation_attributes
# --------------------------------------------------------------------------- #
def test_evaluation_attributes_full_shape(foreign_tracer) -> None:
    tracer, col = foreign_tracer
    with tracer.start_as_current_span("scored") as span:
        set_evaluation_attributes(
            span,
            name="correctness",
            score=0.75,
            label="pass",
            explanation="matched the expected answer",
            annotator_kind="LLM",
        )

    (_, attrs, _), = col.spans
    assert attrs["openinference.span.kind"] == "EVALUATOR"
    assert attrs["evaluation.name"] == "correctness"
    assert attrs["evaluation.score"] == 0.75
    assert isinstance(attrs["evaluation.score"], float)
    assert attrs["evaluation.label"] == "pass"
    assert attrs["evaluation.explanation"] == "matched the expected answer"
    assert attrs["evaluation.annotator_kind"] == "LLM"


def test_evaluation_attributes_optional_fields_omitted(foreign_tracer) -> None:
    tracer, col = foreign_tracer
    with tracer.start_as_current_span("scored") as span:
        set_evaluation_attributes(span, name="brevity", score=1)

    (_, attrs, _), = col.spans
    assert attrs["evaluation.score"] == 1.0
    assert "evaluation.label" not in attrs
    assert "evaluation.explanation" not in attrs
    # annotator_kind always lands — it has a default, unlike label/explanation.
    assert attrs["evaluation.annotator_kind"] == "LLM"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(4.0, 1.0), (5.0, 1.0), (-0.5, 0.0)],
)
def test_evaluation_score_clamped_and_warned(
    foreign_tracer, caplog: pytest.LogCaptureFixture, raw: float, expected: float
) -> None:
    """A raw 1..5 judge score is the mistake this guard exists to surface."""
    tracer, col = foreign_tracer
    with caplog.at_level(logging.WARNING, logger="fastaiagent.trace.span"):
        with tracer.start_as_current_span("scored") as span:
            set_evaluation_attributes(span, name="helpfulness", score=raw)

    (_, attrs, _), = col.spans
    assert attrs["evaluation.score"] == expected
    assert "outside 0..1" in caplog.text
    assert "helpfulness" in caplog.text


def test_evaluation_score_in_range_does_not_warn(
    foreign_tracer, caplog: pytest.LogCaptureFixture
) -> None:
    tracer, _ = foreign_tracer
    with caplog.at_level(logging.WARNING, logger="fastaiagent.trace.span"):
        with tracer.start_as_current_span("scored") as span:
            set_evaluation_attributes(span, name="helpfulness", score=1.0)
            set_evaluation_attributes(span, name="helpfulness", score=0.0)
    assert caplog.text == ""


# --------------------------------------------------------------------------- #
# Task 3 — one-call emitters on the caller's tracer
# --------------------------------------------------------------------------- #
def test_emit_guardrail_pass_and_block_on_foreign_tracer(foreign_tracer) -> None:
    tracer, col = foreign_tracer
    checks_pass = json.dumps([{"name": "no_pii", "result": "pass"}])
    checks_block = json.dumps([{"name": "no_pii", "result": "block"}])

    emit_guardrail(tracer, name="no_pii", position="output", passed=True, checks=checks_pass)
    emit_guardrail(
        tracer,
        name="no_pii",
        position="output",
        passed=False,
        checks=checks_block,
        message="SSN detected",
    )

    assert [n for n, _, _ in col.spans] == ["guardrail.no_pii", "guardrail.no_pii"]
    (_, ok_attrs, ok_status), (_, bad_attrs, bad_status) = col.spans

    # Identical wire shape to the SDK runtime's own guardrail span.
    assert ok_attrs["openinference.span.kind"] == "GUARDRAIL"
    assert ok_attrs["span_type"] == "guardrail"
    assert ok_attrs["fastaiagent.guardrail.name"] == "no_pii"
    assert ok_attrs["fastaiagent.guardrail.position"] == "output"
    assert ok_attrs["fastaiagent.guardrail.passed"] is True
    assert ok_attrs["fastaiagent.guardrail.errored"] is False
    assert ok_status == "OK"

    assert bad_attrs["fastaiagent.guardrail.passed"] is False
    assert bad_status == "ERROR"


def test_emit_guardrail_errored_pass_keeps_ok_status(foreign_tracer) -> None:
    """A fail-open (errored + on_error='allow') is an OK span, still flagged."""
    tracer, col = foreign_tracer
    emit_guardrail(
        tracer,
        name="judge",
        position="output",
        passed=True,
        checks=json.dumps([{"name": "judge", "result": "error"}]),
        errored=True,
    )
    (_, attrs, status), = col.spans
    assert status == "OK"
    assert attrs["fastaiagent.guardrail.errored"] is True
    assert json.loads(attrs["fastaiagent.guardrail.checks"])[0]["result"] == "error"


def test_emit_evaluation_on_foreign_tracer(foreign_tracer) -> None:
    tracer, col = foreign_tracer
    emit_evaluation(
        tracer, name="correctness", score=0.5, label="partial", annotator_kind="CODE"
    )
    (name, attrs, _), = col.spans
    assert name == "evaluation.correctness"
    assert attrs["openinference.span.kind"] == "EVALUATOR"
    assert attrs["evaluation.score"] == 0.5
    assert attrs["evaluation.label"] == "partial"
    assert attrs["evaluation.annotator_kind"] == "CODE"


def test_emitters_never_touch_the_sdk_provider(foreign_tracer, monkeypatch) -> None:
    """The whole point of the borrow surface: no second exporter appears."""
    import fastaiagent.trace.otel as otel

    def _boom() -> None:
        raise AssertionError("emitters must not reach for the SDK tracer provider")

    monkeypatch.setattr(otel, "get_tracer_provider", _boom)
    tracer, col = foreign_tracer
    emit_guardrail(tracer, name="g", position="input", passed=True, checks="[]")
    emit_evaluation(tracer, name="e", score=0.9)
    assert len(col.spans) == 2


def test_emitters_swallow_tracing_failures() -> None:
    """Observability must never break the caller's control flow."""

    class _BrokenTracer:
        def start_as_current_span(self, name):  # noqa: ANN001
            raise RuntimeError("exporter is down")

    emit_guardrail(_BrokenTracer(), name="g", position="input", passed=True, checks="[]")
    emit_evaluation(_BrokenTracer(), name="e", score=0.5)


def test_sdk_runtime_and_borrowed_primitive_produce_the_same_span(foreign_tracer) -> None:
    """Cell 3 vs cell 1: same guardrail, two runtimes, one wire shape."""
    from fastaiagent.guardrail import GuardrailPosition, execute_guardrails, no_pii, run_guardrail

    rail = no_pii(position=GuardrailPosition.output)

    # Cell 1 — foreign runtime: compute with the primitive, emit on own tracer.
    tracer, col = foreign_tracer
    result = asyncio.run(run_guardrail(rail, "all clean here"))
    emit_guardrail(
        tracer,
        name=rail.name,
        position=rail.position.value,
        passed=result.passed,
        checks=json.dumps([{"name": rail.name, "result": "pass"}]),
    )
    (_, borrowed, _), = col.spans

    # Cell 3 — SDK runtime: compute+emit coupled, SDK tracer.
    sdk_col = _Collector()
    from fastaiagent.trace.otel import get_tracer_provider

    get_tracer_provider().add_span_processor(SimpleSpanProcessor(sdk_col))
    asyncio.run(execute_guardrails([rail], "all clean here", GuardrailPosition.output))
    native = next(a for n, a, _ in sdk_col.spans if n.startswith("guardrail."))

    shared = ("openinference.span.kind", "span_type", "fastaiagent.guardrail.name",
              "fastaiagent.guardrail.position", "fastaiagent.guardrail.passed",
              "fastaiagent.guardrail.errored")
    assert {k: borrowed[k] for k in shared} == {k: native[k] for k in shared}


# --------------------------------------------------------------------------- #
# Task 3 — public primitive surface
# --------------------------------------------------------------------------- #
def test_primitives_are_public_api() -> None:
    import fastaiagent as fa

    for name in (
        "run_guardrail",
        "guardrail_from_policy_rule",
        "plane_guardrails_for_agent",
        "set_guardrail_attributes",
        "set_evaluation_attributes",
        "emit_guardrail",
        "emit_evaluation",
    ):
        assert name in fa.__all__, f"{name} missing from fastaiagent.__all__"
        assert getattr(fa, name) is not None


def test_run_guardrail_is_compute_only(foreign_tracer) -> None:
    """The primitive returns a verdict and emits nothing — compute ≠ emit."""
    from fastaiagent.guardrail import GuardrailPosition, no_pii, run_guardrail

    _, col = foreign_tracer
    sdk_col = _Collector()
    from fastaiagent.trace.otel import get_tracer_provider

    get_tracer_provider().add_span_processor(SimpleSpanProcessor(sdk_col))

    rail = no_pii(position=GuardrailPosition.output)
    result = asyncio.run(run_guardrail(rail, "My SSN is 123-45-6789."))

    assert result.passed is False
    assert col.spans == []
    assert [n for n, _, _ in sdk_col.spans if n.startswith("guardrail.")] == []


# --------------------------------------------------------------------------- #
# Task 4 — connect(export_traces=False)
# --------------------------------------------------------------------------- #
# Port 1 is reserved/unbound: connect() takes its unreachable-plane path, which
# is a real code path (it exists so traces queue while the plane is down), not a
# stub. Everything under test here is local to the process.
_UNREACHABLE = "http://127.0.0.1:1"


def _processor_names(provider) -> list[str]:  # noqa: ANN001
    return [type(p).__name__ for p in provider._active_span_processor._span_processors]


def test_connect_without_trace_export_registers_no_platform_exporter() -> None:
    import fastaiagent as fa
    from fastaiagent.client import _connection
    from fastaiagent.trace.otel import get_tracer_provider

    try:
        fa.connect("fa_test_key", target=_UNREACHABLE, auto_register=False, export_traces=False)
        assert _connection._platform_processor is None
        provider = get_tracer_provider()
        assert "PlatformSpanExporter" not in " ".join(_processor_names(provider))
        # Still connected for everything else — policy, scorers, prompts.
        assert _connection.is_connected
    finally:
        fa.disconnect()


def test_connect_default_still_registers_the_exporter() -> None:
    """The default must preserve 1.46.0 behavior exactly."""
    import fastaiagent as fa
    from fastaiagent.client import _connection

    try:
        fa.connect("fa_test_key", target=_UNREACHABLE, auto_register=False)
        assert _connection._platform_processor is not None
    finally:
        fa.disconnect()


_GLOBAL_PROVIDER_PROBE = """
import sys
from opentelemetry import trace as otel_trace
import fastaiagent as fa
from fastaiagent.trace.otel import get_tracer_provider

fa.connect("fa_test_key", target="http://127.0.0.1:1", auto_register=False,
           export_traces={export_traces})
sdk_provider = get_tracer_provider()
sys.stdout.write("CLAIMED" if otel_trace.get_tracer_provider() is sdk_provider else "FREE")
"""


@pytest.mark.parametrize(
    ("export_traces", "expected"), [("False", "FREE"), ("True", "CLAIMED")]
)
def test_global_provider_claim_follows_the_flag(export_traces: str, expected: str) -> None:
    """Fresh interpreter each time — OTel's global set is first-wins."""
    proc = subprocess.run(
        [sys.executable, "-c", _GLOBAL_PROVIDER_PROBE.format(export_traces=export_traces)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith(expected), proc.stdout


# --------------------------------------------------------------------------- #
# Span-kind classification — the Local UI reads fastaiagent.runner.type
# --------------------------------------------------------------------------- #
def test_openinference_kinds_classify_for_the_local_ui() -> None:
    """Both kinds this release emits must normalize to a runner type.

    Without EVALUATOR in the map, an inline eval-score span falls back to being
    classified as an agent span and shows up in the Local UI's trace list as a
    stray agent run. The map must also stay in step with the control plane's,
    so a span is classified identically locally and remotely.
    """
    from fastaiagent.trace.normalize import _SPAN_KIND_MAP, normalize_attributes

    assert _SPAN_KIND_MAP["GUARDRAIL"] == "guardrail"
    assert _SPAN_KIND_MAP["EVALUATOR"] == "evaluator"

    guard = normalize_attributes({"openinference.span.kind": "GUARDRAIL"})
    assert guard["fastaiagent.runner.type"] == "guardrail"

    ev = normalize_attributes(
        {"openinference.span.kind": "EVALUATOR", "evaluation.name": "helpfulness",
         "evaluation.score": 0.8}
    )
    assert ev["fastaiagent.runner.type"] == "evaluator"
    # Passthrough: the normalizer never drops the payload it classified on.
    assert ev["evaluation.score"] == 0.8
    assert ev["evaluation.name"] == "helpfulness"
