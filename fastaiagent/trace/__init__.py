"""OTel-native tracing with local storage and Agent Replay."""

from fastaiagent.trace.otel import add_exporter, get_tracer, reset
from fastaiagent.trace.otel_capture import disable_otel_capture, enable_otel_capture
from fastaiagent.trace.redaction import (
    SENSITIVE_ATTR_KEYS,
    RedactionPolicy,
    get_redaction_policy,
    set_redaction_policy,
)
from fastaiagent.trace.replay import Replay
from fastaiagent.trace.span import (
    emit_evaluation,
    emit_guardrail,
    set_evaluation_attributes,
    set_guardrail_attributes,
)
from fastaiagent.trace.storage import TraceData, TraceStore, TraceSummary
from fastaiagent.trace.tracer import trace_context

__all__ = [
    "trace_context",
    "get_tracer",
    "add_exporter",
    "reset",
    # OpenInference emitters — stamp attributes on a span you own, or emit a
    # whole child span on your own tracer (see fastaiagent.emit_guardrail).
    "set_guardrail_attributes",
    "set_evaluation_attributes",
    "emit_guardrail",
    "emit_evaluation",
    "enable_otel_capture",
    "disable_otel_capture",
    "TraceStore",
    "TraceData",
    "TraceSummary",
    "Replay",
    "RedactionPolicy",
    "set_redaction_policy",
    "get_redaction_policy",
    "SENSITIVE_ATTR_KEYS",
]
