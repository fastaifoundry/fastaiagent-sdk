"""Span helpers and GenAI semantic convention mappings."""

from __future__ import annotations

import os
from typing import Any


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
    "fastaiagent.prompt.name": str,
    "fastaiagent.prompt.version": int,
    "fastaiagent.prompt.slug": str,
    "fastaiagent.cost.total_usd": float,
    "fastaiagent.thread.id": str,
    # Universal-harness attributes — set on the root span of every run so
    # the Local UI can badge / filter / group by source framework.
    "fastaiagent.framework": str,
    "fastaiagent.framework.version": str,
    "fastaiagent.external_agent.name": str,
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
