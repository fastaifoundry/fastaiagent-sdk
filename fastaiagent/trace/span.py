"""Span helpers and GenAI semantic convention mappings."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def trace_payloads_enabled() -> bool:
    """Whether to capture payload-bearing trace attributes (prompts, messages, responses).

    Defaults to True. Set ``FASTAIAGENT_TRACE_PAYLOADS=0`` to disable when payloads
    may contain PII or sensitive data. Structural metadata (provider, model, tool
    schemas, guardrail config) is always captured regardless of this flag — only
    free-text content is gated.
    """
    return os.environ.get("FASTAIAGENT_TRACE_PAYLOADS", "1") != "0"


# GenAI semantic conventions (OTel standard)
GENAI_ATTRIBUTES = {
    "gen_ai.system": str,
    "gen_ai.request.model": str,
    "gen_ai.request.temperature": float,
    "gen_ai.request.max_tokens": int,
    "gen_ai.usage.input_tokens": int,
    "gen_ai.usage.output_tokens": int,
    "gen_ai.response.finish_reasons": list,
}

# FastAIAgent custom attributes (namespaced as ``fastaiagent.*`` — the short
# form ``fastai.*`` is reserved by fast.ai, a different company, so we never
# emit it).
FASTAIAGENT_ATTRIBUTES = {
    "fastaiagent.agent.name": str,
    "fastaiagent.chain.name": str,
    "fastaiagent.chain.node_id": str,
    "fastaiagent.chain.iteration": int,
    "fastaiagent.tool.name": str,
    "fastaiagent.checkpoint.id": str,
    "fastaiagent.guardrail.name": str,
    "fastaiagent.guardrail.passed": bool,
    "fastaiagent.guardrail.position": str,
    # True when the check itself failed to run (e.g. a model-judged detector's
    # LLM call raised) and ``passed`` reflects the guardrail's ``on_error``
    # policy rather than a real verdict. Lets a fail-open be told apart from a
    # genuine pass in traces.
    "fastaiagent.guardrail.errored": bool,
    # JSON string: ``[{"name": <guardrail>, "result": "pass"|"block"|"error"}, …]``.
    # The plane maps this onto ``output.checks`` and the console renders a
    # per-span CHECKS row (green pass / red block / amber error). Set via
    # :func:`set_guardrail_attributes`.
    "fastaiagent.guardrail.checks": str,
    "fastaiagent.prompt.name": str,
    "fastaiagent.prompt.version": int,
    "fastaiagent.prompt.slug": str,
    # Optional deployment environment the registry prompt was resolved for
    # (e.g. ``"production"``). Stamped on ``llm.*`` spans alongside slug/version
    # so the plane's Prompt Analytics can attribute the call.
    "fastaiagent.prompt.environment": str,
    "fastaiagent.cost.total_usd": float,
    "fastaiagent.thread.id": str,
    # Universal-harness attributes — set on the root span of every run so
    # the Local UI can badge / filter / group by source framework.
    "fastaiagent.framework": str,
    "fastaiagent.framework.version": str,
    "fastaiagent.external_agent.name": str,
    # Template marker — set on the root span by flagship example templates
    # (deep-research-agent, customer-support-agent, …) so the UI can render
    # a per-template badge and trace lists can be filtered by template kind
    # without parsing span names. Use :func:`set_template_kind` to set it.
    "fastaiagent.template.kind": str,
    # Deep Research template attributes (set by examples/deep-research-agent).
    # JSON-serialized payloads — the local UI / replay tooling can deserialize
    # them to reconstruct brief, plan, and findings without re-parsing prose.
    "fastaiagent.research.topic": str,
    "fastaiagent.research.brief": str,
    "fastaiagent.research.plan": str,
    "fastaiagent.research.subtopic": str,
    "fastaiagent.research.findings": str,
    "fastaiagent.research.report.chars": int,
    "fastaiagent.research.report.citations": int,
}

# Back-compat alias — remove in 0.9. Existing callers that imported the old
# name keep working; new code should reach for ``FASTAIAGENT_ATTRIBUTES``.
FASTAI_ATTRIBUTES = FASTAIAGENT_ATTRIBUTES


def set_span_attributes(span: Any, **kwargs: Any) -> None:
    """Set attributes on a span, filtering out None values."""
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, list):
                span.set_attribute(key, str(value))
            else:
                span.set_attribute(key, value)


def set_genai_attributes(
    span: Any,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    finish_reasons: list[str] | None = None,
    request_messages: str | None = None,
    request_tools: str | None = None,
    response_content: str | None = None,
    response_tool_calls: str | None = None,
    finish_reason: str | None = None,
) -> None:
    """Set GenAI semantic convention attributes on a span.

    Payload-bearing fields (request_messages, request_tools, response_content,
    response_tool_calls) are pre-serialized JSON strings provided by the caller
    and are gated by ``trace_payloads_enabled()``.
    """
    attrs: dict[str, Any] = {}
    if system is not None:
        attrs["gen_ai.system"] = system
    if model is not None:
        attrs["gen_ai.request.model"] = model
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = temperature
    if max_tokens is not None:
        attrs["gen_ai.request.max_tokens"] = max_tokens
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    if finish_reasons is not None:
        attrs["gen_ai.response.finish_reasons"] = str(finish_reasons)
    if finish_reason is not None:
        attrs["gen_ai.response.finish_reason"] = finish_reason
    if trace_payloads_enabled():
        if request_messages is not None:
            attrs["gen_ai.request.messages"] = request_messages
        if request_tools is not None:
            attrs["gen_ai.request.tools"] = request_tools
        if response_content is not None:
            attrs["gen_ai.response.content"] = response_content
        if response_tool_calls is not None:
            attrs["gen_ai.response.tool_calls"] = response_tool_calls
    set_span_attributes(span, **attrs)


def set_fastaiagent_attributes(span: Any, **kwargs: Any) -> None:
    """Set FastAIAgent custom attributes on a span.

    Keys should be without the ``fastaiagent.`` prefix — it's added
    automatically. We use ``fastaiagent.*`` rather than ``fastai.*`` so we
    don't squat on fast.ai's namespace.
    """
    prefixed = {f"fastaiagent.{k}": v for k, v in kwargs.items() if v is not None}
    set_span_attributes(span, **prefixed)


# Back-compat alias — remove in 0.9.
set_fastai_attributes = set_fastaiagent_attributes


def set_guardrail_attributes(
    span: Any,
    name: str,
    position: str,
    passed: bool,
    checks: str,
    errored: bool = False,
) -> None:
    """Stamp guardrail-outcome attributes on a per-guardrail child span.

    Mirrors :func:`set_genai_attributes` for the guardrail contract. The span is
    classified with the **OpenInference** standard kind
    (``openinference.span.kind="GUARDRAIL"``) so any consumer of that ecosystem
    recognizes it. OpenInference standardizes the *kind*, not the outcome
    fields — so the ``fastaiagent.guardrail.*`` namespace is our documented
    convention riding **under** the standard kind, and it is what the plane maps
    onto ``output.checks``.

    The legacy ``span_type="guardrail"`` marker is still written alongside it
    (dual-write) for plane deployments that predate the OpenInference reader; it
    is not removed until that old-format reader is retired.

    Emitted on **pass and block** so the console shows green passes too — the
    caller is responsible for the OTel span *status* (``ERROR`` on block), or
    can use :func:`emit_guardrail` which handles the whole span lifecycle.

    ``checks`` is a pre-serialized JSON string, conventionally
    ``[{"name": name, "result": "pass"|"block"|"error"}]``. ``errored`` marks a
    check that could not run (its ``passed`` reflects the ``on_error`` policy).
    """
    # Both classifiers are plain (unprefixed) attributes carried inside the open
    # OTel envelope — no wire-protocol bump for either.
    span.set_attribute("span_type", "guardrail")  # legacy — old-plane back-compat
    span.set_attribute("openinference.span.kind", "GUARDRAIL")  # the standard kind
    set_fastaiagent_attributes(
        span,
        **{
            "guardrail.name": name,
            "guardrail.position": position,
            "guardrail.passed": passed,
            "guardrail.errored": errored,
            "guardrail.checks": checks,
        },
    )


def set_evaluation_attributes(
    span: Any,
    *,
    name: str,
    score: float,
    label: str | None = None,
    explanation: str | None = None,
    annotator_kind: str = "LLM",
) -> None:
    """Stamp OpenInference ``EVALUATOR`` attributes for an inline eval score.

    This is the canonical shape for a **per-trace** score: one span carrying
    ``openinference.span.kind="EVALUATOR"`` plus the ``evaluation.*`` namespace
    (the convention Arize/Phoenix and eval2otel emit). The plane reads
    ``evaluation.score`` off any span in a trace into a ``TraceScore``.

    ``score`` **must be on a 0..1 scale**. A raw judge score on another scale
    has to be normalized by the caller (a 1..5 judge → ``score / 5``). An
    out-of-range value is clamped into [0, 1] and warned about rather than
    emitted as-is, so a scale mistake surfaces at the emitter instead of landing
    silently clamped on the plane.

    ``annotator_kind`` follows the OpenInference vocabulary — ``"LLM"`` for a
    model-judged score, ``"CODE"`` for a deterministic one, ``"HUMAN"`` for an
    annotation.

    This is a *stamper*: the SDK does not score inline anywhere in ``agent.run``
    today. It exists for runtimes that compute a score themselves — see
    :func:`emit_evaluation` for the one-call version. Dataset/batch/CI scoring
    stays on ``EvalResults.publish``; the two granularities do not overlap.
    """
    value = float(score)
    if not 0.0 <= value <= 1.0:
        clamped = min(1.0, max(0.0, value))
        logger.warning(
            "evaluation.score=%s for %r is outside 0..1 and was clamped to %s. "
            "Scores must be normalized before emitting (a 1..5 judge score is score/5).",
            value,
            name,
            clamped,
        )
        value = clamped

    span.set_attribute("openinference.span.kind", "EVALUATOR")
    span.set_attribute("evaluation.name", name)
    span.set_attribute("evaluation.score", value)
    if label is not None:
        span.set_attribute("evaluation.label", label)
    if explanation is not None:
        span.set_attribute("evaluation.explanation", explanation)
    span.set_attribute("evaluation.annotator_kind", annotator_kind)


def emit_guardrail(
    tracer: Any,
    *,
    name: str,
    position: str,
    passed: bool,
    checks: str,
    errored: bool = False,
    message: str | None = None,
) -> None:
    """Open, stamp and close one guardrail-outcome child span in a single call.

    The convenience form of :func:`set_guardrail_attributes` for a runtime that
    computed a verdict itself — e.g. a LangChain/CrewAI app that ran
    :func:`fastaiagent.run_guardrail` (compute) and now needs to report the
    outcome (emit).

    **The tracer is yours.** Pass the tracer of the exporter your process
    already owns; this function never reaches for the SDK's tracer provider, so
    a borrowing runtime keeps exactly one exporter. The span nests under
    whatever span is currently active, joining the caller's trace.

    Best-effort: a tracing failure is swallowed, never propagated into the
    caller's control flow.
    """
    try:
        from opentelemetry.trace import Status, StatusCode

        with tracer.start_as_current_span(f"guardrail.{name}") as span:
            set_guardrail_attributes(
                span,
                name=name,
                position=position,
                passed=passed,
                checks=checks,
                errored=errored,
            )
            if passed:
                # A degraded pass (errored + on_error="allow") keeps an OK
                # status — the run continued — but the errored attribute and
                # the "error" check result keep it distinguishable.
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(
                    Status(StatusCode.ERROR, message or f"Blocked by guardrail: {name}")
                )
    except Exception:  # pragma: no cover - observability must never break a run
        logger.debug("Failed to emit guardrail span for %r", name, exc_info=True)


def emit_evaluation(
    tracer: Any,
    *,
    name: str,
    score: float,
    label: str | None = None,
    explanation: str | None = None,
    annotator_kind: str = "LLM",
) -> None:
    """Open, stamp and close one ``EVALUATOR`` span in a single call.

    The convenience form of :func:`set_evaluation_attributes` — e.g. a foreign
    runtime that scored a turn with ``Scorer.from_platform(...).score(...)`` and
    wants that score attached to the trace it just produced.

    ``score`` must be on a 0..1 scale (see :func:`set_evaluation_attributes`).
    As with :func:`emit_guardrail`, the tracer is the caller's — one exporter
    per process — and failures are swallowed.
    """
    try:
        with tracer.start_as_current_span(f"evaluation.{name}") as span:
            set_evaluation_attributes(
                span,
                name=name,
                score=score,
                label=label,
                explanation=explanation,
                annotator_kind=annotator_kind,
            )
    except Exception:  # pragma: no cover - observability must never break a run
        logger.debug("Failed to emit evaluation span for %r", name, exc_info=True)


def set_metadata_attributes(span: Any, metadata: dict[str, Any] | None) -> None:
    """Stamp developer-supplied run metadata as ``fastaiagent.meta.*`` attributes.

    MLflow-style tags: the developer attaches arbitrary key/values to a run
    (``agent.run(metadata={...})``) and they land on the run's root span as
    namespaced attributes — queryable in the trace store, and rideable on the
    open OTel envelope with no wire change. Keys are stringified; primitive
    values pass through, everything else is JSON-serialized. ``None`` values are
    skipped. The developer owns the keys, the values, and any PII implications.
    """
    if not metadata:
        return
    import json

    for key, value in metadata.items():
        if value is None:
            continue
        attr_key = f"fastaiagent.meta.{key}"
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(attr_key, value)
        else:
            try:
                span.set_attribute(attr_key, json.dumps(value, default=str))
            except Exception:
                span.set_attribute(attr_key, str(value))


def set_template_kind(span: Any, kind: str) -> None:
    """Stamp a template-kind marker on the root span of a flagship template run.

    Use this in template entrypoints (e.g. ``examples/deep-research-agent``)
    to make the trace identifiable as belonging to a known template — the
    UI can then render a per-template badge and filter trace lists by kind
    without parsing span names.

    ``kind`` is a short, kebab-case identifier (``"deep-research"``,
    ``"customer-support"``, ``"meeting-notes"``). Conventionally matches the
    template's directory name under ``examples/``.

    Example::

        from fastaiagent.trace import trace_context
        from fastaiagent.trace.span import set_template_kind

        with trace_context("deep_research.session") as span:
            set_template_kind(span, "deep-research")
            ...
    """
    if not kind:
        return
    span.set_attribute("fastaiagent.template.kind", kind)
