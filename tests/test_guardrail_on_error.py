"""Tests for the guardrail ``on_error`` policy and the ``errored`` outcome.

No mocks and no live LLM: a Python callable that raises is a perfectly real
user guardrail whose check failed, so it drives the *actual* production error
path (run_guardrail's choke point → executor → on_error policy) with a genuine
exception. That is exactly the fail-open/fail-closed scenario, exercised
deterministically.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

import fastaiagent as fa
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail.builtins import (
    allowed_topics,
    banned_topics,
    grounded,
    openai_moderation,
)
from fastaiagent.guardrail.executor import execute_guardrails
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailPosition
from fastaiagent.trace.otel import get_tracer_provider
from fastaiagent.ui.events import _outcome


def _boom(_text: str) -> bool:
    raise RuntimeError("detector down")


# --------------------------------------------------------------------------- #
# GuardrailResult.errored + on_error at the single choke point
# --------------------------------------------------------------------------- #
def test_errors_block_by_default() -> None:
    g = Guardrail(name="boom", fn=_boom)  # default on_error="block"
    res = g.execute("hello")
    assert res.passed is False
    assert res.errored is True
    assert res.metadata.get("on_error") == "block"
    assert "detector down" in (res.message or "")


def test_errors_fail_open_when_allowed() -> None:
    g = Guardrail(name="boom", fn=_boom, on_error="allow")
    res = g.execute("hello")
    assert res.passed is True
    assert res.errored is True
    assert res.metadata.get("on_error") == "allow"


def test_healthy_guardrail_not_marked_errored() -> None:
    res = fa.no_pii().execute("nothing to see here")
    assert res.passed is True
    assert res.errored is False


# --------------------------------------------------------------------------- #
# Executor enforcement
# --------------------------------------------------------------------------- #
def test_executor_raises_on_errored_block() -> None:
    g = Guardrail(name="boom", fn=_boom, blocking=True, on_error="block")
    with pytest.raises(GuardrailBlockedError) as exc:
        asyncio.run(execute_guardrails([g], "x", GuardrailPosition.output))
    assert exc.value.guardrail_name == "boom"


def test_executor_does_not_raise_on_errored_allow() -> None:
    g = Guardrail(name="boom", fn=_boom, blocking=True, on_error="allow")
    results = asyncio.run(execute_guardrails([g], "x", GuardrailPosition.output))
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].errored is True


# --------------------------------------------------------------------------- #
# Serialization round-trips the policy
# --------------------------------------------------------------------------- #
def test_on_error_serialization_roundtrip() -> None:
    g = Guardrail(name="x", fn=_boom, on_error="allow")
    restored = Guardrail.from_dict(g.to_dict())
    assert restored.on_error == "allow"
    # Backward compat: a dict without on_error defaults to "block".
    legacy = Guardrail.from_dict({"name": "legacy"})
    assert legacy.on_error == "block"


# --------------------------------------------------------------------------- #
# Built-in factory defaults preserve today's behavior
# --------------------------------------------------------------------------- #
def test_factory_on_error_defaults() -> None:
    # Previously fail-open detectors keep "allow".
    assert fa.toxicity_check().on_error == "allow"
    assert fa.no_prompt_injection().on_error == "allow"
    assert banned_topics(["politics"]).on_error == "allow"
    # Previously fail-closed detectors keep "block".
    assert grounded("ref").on_error == "block"
    assert openai_moderation().on_error == "block"
    # allowed_topics is a whitelist — an unclassifiable output was blocked.
    assert allowed_topics(["support"]).on_error == "block"


def test_responsible_ai_override_applies_to_llm_rails() -> None:
    rails = {g.name: g for g in fa.responsible_ai(toxicity=True, on_error="block")}
    assert rails["toxicity_check"].on_error == "block"
    assert rails["no_prompt_injection"].on_error == "block"


# --------------------------------------------------------------------------- #
# Observability: outcome + trace span
# --------------------------------------------------------------------------- #
def test_outcome_reports_errored() -> None:
    g = Guardrail(name="boom", fn=_boom, on_error="allow")
    assert _outcome(g, g.execute("x")) == "errored"


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
def collector() -> _Collector:
    col = _Collector()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(col))
    return col


def test_span_marks_errored_check(collector: _Collector) -> None:
    async def _run() -> None:
        # Fail open → run continues, span still emitted.
        await execute_guardrails(
            [Guardrail(name="boom", fn=_boom, on_error="allow", position=GuardrailPosition.output)],
            "x",
            GuardrailPosition.output,
        )

    asyncio.run(_run())
    guard = [(a, st) for n, a, st in collector.spans if n.startswith("guardrail.")]
    assert len(guard) == 1
    attrs, status = guard[0]
    assert attrs["fastaiagent.guardrail.errored"] is True
    assert json.loads(attrs["fastaiagent.guardrail.checks"])[0]["result"] == "error"
    # Degraded pass keeps an OK status; the errored attribute carries the signal.
    assert status == "OK"
