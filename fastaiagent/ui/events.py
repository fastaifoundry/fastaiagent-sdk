"""Guardrail event persistence.

The UI is read-only over REST (see ``project_ui_refresh_based`` memory): there
is no in-process pub/sub or WebSocket between the agent runtime and the UI.
What this module does do is append a row to the ``guardrail_events`` table in
``local.db`` each time a Guardrail runs, so the Guardrails page can render it
on refresh. Gated on ``SDKConfig.ui_enabled`` — non-UI users pay nothing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import init_local_db

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail, GuardrailResult


def log_guardrail_event(
    guardrail: Guardrail,
    result: GuardrailResult,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    agent_name: str | None = None,
    db_path: str | None = None,
) -> None:
    """Persist a guardrail execution to ``local.db`` for the UI to read.

    Gated on :attr:`SDKConfig.ui_enabled`; no-op when the UI isn't in use.
    """
    config = get_config()
    if not config.ui_enabled:
        return

    trace_id, span_id, agent_name = _fill_span_context(trace_id, span_id, agent_name)

    outcome = _outcome(guardrail, result)
    timestamp = datetime.now(tz=timezone.utc).isoformat()

    resolved_db = db_path or config.local_db_path
    helper: SQLiteHelper | None = None
    try:
        helper = init_local_db(resolved_db)
        import json

        from fastaiagent._internal.project import safe_get_project_id

        helper.execute(
            """INSERT INTO guardrail_events
               (event_id, trace_id, span_id, guardrail_name, guardrail_type,
                position, outcome, score, message, agent_name, timestamp, metadata,
                project_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                trace_id,
                span_id,
                guardrail.name,
                guardrail.guardrail_type.value,
                guardrail.position.value,
                outcome,
                result.score,
                result.message,
                agent_name,
                timestamp,
                json.dumps(result.metadata or {}),
                safe_get_project_id(),
            ),
        )
    finally:
        if helper is not None:
            helper.close()


def _outcome(guardrail: Guardrail, result: GuardrailResult) -> str:
    if result.passed:
        return "passed"
    return "blocked" if guardrail.blocking else "warned"


def _fill_span_context(
    trace_id: str | None, span_id: str | None, agent_name: str | None
) -> tuple[str | None, str | None, str | None]:
    """Best-effort pull of the current OTel span context + agent name."""
    if trace_id and span_id and agent_name:
        return trace_id, span_id, agent_name
    try:
        from opentelemetry import trace as otel_trace
    except ImportError:
        return trace_id, span_id, agent_name

    span = otel_trace.get_current_span()
    ctx = getattr(span, "get_span_context", lambda: None)()
    if ctx is not None and getattr(ctx, "trace_id", None):
        if not trace_id:
            trace_id = format(ctx.trace_id, "032x")
        if not span_id:
            span_id = format(ctx.span_id, "016x")
    if not agent_name:
        raw_attrs = getattr(span, "attributes", None)
        if raw_attrs:
            try:
                attrs_dict = dict(raw_attrs)
            except (TypeError, AttributeError):
                attrs_dict = {}
            from fastaiagent.ui.attrs import attr

            resolved = attr(attrs_dict, "agent.name")
            agent_name = str(resolved) if resolved is not None else None
    return trace_id, span_id, agent_name
