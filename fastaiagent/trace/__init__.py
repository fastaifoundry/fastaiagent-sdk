"""OTel-native tracing with local storage and Agent Replay."""

from fastaiagent.trace.otel import add_exporter, get_tracer, reset
from fastaiagent.trace.redaction import (
    SENSITIVE_ATTR_KEYS,
    RedactionPolicy,
    get_redaction_policy,
    set_redaction_policy,
)
from fastaiagent.trace.replay import Replay
from fastaiagent.trace.storage import TraceData, TraceStore, TraceSummary
from fastaiagent.trace.tracer import trace_context

__all__ = [
    "trace_context",
    "get_tracer",
    "add_exporter",
    "reset",
    "TraceStore",
    "TraceData",
    "TraceSummary",
    "Replay",
    "RedactionPolicy",
    "set_redaction_policy",
    "get_redaction_policy",
    "SENSITIVE_ATTR_KEYS",
]
