"""Guardrail event persistence: the local record of every guardrail check.

The UI is read-only over REST (see ``project_ui_refresh_based`` memory): there
is no in-process pub/sub or WebSocket between the agent runtime and the UI.
This module appends rows to the ``guardrail_events`` table in ``local.db``,
which the Guardrails page reads on refresh.

**When rows are written (1.89.0).** Whenever tracing is on
(``FASTAIAGENT_TRACE_ENABLED``, default on): the event log is local capture,
like spans, not a UI feature. It used to be gated on ``FASTAIAGENT_UI_ENABLED``
in the *agent's* process, so the Guardrail Events page was empty by default.
``UI_ENABLED`` no longer affects it. A bare ``Guardrail.execute()`` writes a
row only inside a recording span (see :func:`bare_call_recording`).

**What a row says.** ``outcome`` keeps its vocabulary (plus ``reasked``) and
``effect`` says what the check did to the run; see
:func:`fastaiagent.guardrail.actions.effect_of`. The text the rule judged is
stored once per distinct text in ``guardrail_payloads``. Nothing here is
exported: these tables never leave the machine.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import init_local_db

if TYPE_CHECKING:
    from fastaiagent.guardrail.guardrail import Guardrail, GuardrailFiring, GuardrailResult

_log = logging.getLogger(__name__)

#: Every value written to ``guardrail_events.outcome``. Readers that filter on
#: outcome (curation, the UI) must use these, never a literal: ``eval curate
#: --filter guardrail`` matched ``'fail'``, a value nothing has ever written, and
#: returned nothing from 1.22.0 to 1.88.0. ``reasked`` is new in 1.89.0: a
#: re-ask used to be recorded as ``blocked``.
GUARDRAIL_OUTCOMES: tuple[str, ...] = (
    "passed",
    "blocked",
    "warned",
    "filtered",
    "errored",
    "reasked",
)

#: The outcomes that mean "the rule fired": it judged the payload and did
#: something about it. ``errored`` (could not run) is an infrastructure signal,
#: not a quality one, and is deliberately excluded.
FIRED_OUTCOMES: tuple[str, ...] = ("blocked", "warned", "filtered", "reasked")


# ── when to record ──────────────────────────────────────────────────────────


def recording_enabled() -> bool:
    """Rows follow local capture: written whenever tracing is on."""
    try:
        from fastaiagent.trace.otel import tracing_enabled

        return tracing_enabled()
    except Exception:  # pragma: no cover - tracing always imports
        return False


def bare_call_recording() -> bool:
    """Should a bare ``Guardrail.execute()`` write a row?

    Only when tracing is on **and** a span is recording, so a check called from
    a web handler or a notebook outside any run never creates ``local.db``.
    """
    if not recording_enabled():
        return False
    try:
        from opentelemetry import trace as otel_trace

        return bool(otel_trace.get_current_span().is_recording())
    except Exception:  # pragma: no cover
        return False


# ── the writer ──────────────────────────────────────────────────────────────

_writers: dict[str, SQLiteHelper] = {}
_writers_lock = threading.Lock()


def _writer(db_path: str) -> SQLiteHelper:
    """One helper per database per process, migrated once.

    Each row used to open a connection and take the migration lock. Fine while
    rows were opt-in; not once every traced check writes one.
    """
    with _writers_lock:
        helper = _writers.get(db_path)
        if helper is not None and not os.path.exists(db_path):
            # The file was deleted (a purge-and-recreate, a test's tmp dir): a
            # cached connection would keep writing to the unlinked inode.
            try:
                helper.close()
            except Exception:  # pragma: no cover
                pass
            helper = None
        if helper is None:
            helper = init_local_db(db_path)
            _writers[db_path] = helper
        return helper


def _reset_writers() -> None:
    """Close the cached writers. For tests, and after the DB path changes."""
    with _writers_lock:
        for helper in _writers.values():
            try:
                helper.close()
            except Exception:  # pragma: no cover
                pass
        _writers.clear()


def _resolve_db(db_path: str | None) -> str:
    # A replay written to another store keeps its guardrail rows beside its
    # spans (1.89.0, the split-DB sink).
    from fastaiagent.trace.provenance import current_provenance

    prov = current_provenance()
    scoped_db = prov.trace_db if prov is not None and prov.trace_db else None
    return str(db_path or scoped_db or get_config().local_db_path)


# ── what a row holds ────────────────────────────────────────────────────────


def _capture_redact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Mask a guardrail event's metadata with the installed capture-mode policy.

    ``trace.storage`` runs span attributes through ``_capture_redact`` on the way
    into ``local.db``; guardrail events skipped it. Best-effort by construction —
    an event row must never be lost because a policy raised.
    """
    try:
        from fastaiagent.trace.redaction import _walk_and_redact, get_redaction_policy

        policy = get_redaction_policy()
        if policy is None or policy.mode not in ("capture", "both") or not policy._compiled:
            return metadata
        return {k: _walk_and_redact(v, policy) for k, v in metadata.items()}
    except Exception:  # pragma: no cover - redaction must not break the write
        return metadata


def _capture_redact_text(text: str | None) -> str | None:
    """The same capture-mode policy, for one string (judged text, message)."""
    if text is None:
        return None
    redacted = _capture_redact_metadata({"t": text}).get("t", text)
    return redacted if isinstance(redacted, str) else str(redacted)


def judged_text(data: Any) -> str:
    """The text a rule saw: the string itself, or the JSON of a structured payload."""
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, default=str)
    except Exception:  # pragma: no cover
        return str(data)


def rule_snapshot(guardrail: Guardrail) -> dict[str, Any]:
    """The rule as it ran, safe to store: ``config.llm`` keeps provider and model only."""
    try:
        snap = guardrail.to_dict()
    except Exception:  # pragma: no cover - a broken to_dict must not cost the row
        snap = {"name": getattr(guardrail, "name", "?")}
    config = dict(snap.get("config") or {})
    llm = config.get("llm")
    if isinstance(llm, dict):
        config["llm"] = {k: llm[k] for k in ("provider", "model") if k in llm}
    elif llm is not None:
        config["llm"] = {
            "provider": getattr(llm, "provider", None),
            "model": getattr(llm, "model", None),
        }
    snap["config"] = config
    snap["origin"] = getattr(guardrail, "origin", "local")
    return snap


def event_outcome(guardrail: Guardrail, result: GuardrailResult, effect: str) -> str:
    """The ``outcome`` column for this check, given its ``effect``."""
    if result.errored:
        return "errored"
    if result.passed:
        return "passed"
    if effect in ("observed", "warned"):
        return "warned"
    if effect == "rewritten":
        return "filtered"
    if effect == "reasked":
        return "reasked"
    return "blocked"


def _outcome(guardrail: Guardrail, result: GuardrailResult, *, allow_reask: bool = False) -> str:
    """Compatibility wrapper (pre-1.89.0 name): the outcome from the verdict alone."""
    from fastaiagent.guardrail.actions import effect_of

    return event_outcome(guardrail, result, effect_of(guardrail, result, allow_reask=allow_reask))


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


@dataclass
class GuardrailEventRecord:
    """One recorded check, handed back on ``GuardrailOutcome.records``."""

    event_id: str
    guardrail_name: str
    position: str
    effect: str
    outcome: str
    firing: GuardrailFiring | None = None
    result: GuardrailResult | None = None


@dataclass
class EventBatch:
    """The rows of one position pass, written in one transaction.

    Context (trace, span, agent) is captured when the batch opens, so every row
    of the pass points at the span that ran it, not at a child span.
    """

    agent_name: str | None = None
    attempt: int | None = None
    trace_id: str | None = None
    span_id: str | None = None
    db_path: str | None = None
    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _rows: list[tuple[Any, ...]] = field(default_factory=list)
    _payloads: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trace_id, self.span_id, self.agent_name = _fill_span_context(
            self.trace_id, self.span_id, self.agent_name
        )

    def add(
        self,
        guardrail: Guardrail,
        result: GuardrailResult,
        *,
        data: Any,
        effect: str,
        guardrail_span_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        final_reason: str | None = None,
        outcome: str | None = None,
    ) -> str:
        """Queue one row; returns its ``event_id``. Never raises."""
        event_id = uuid.uuid4().hex
        try:
            metadata = dict(result.metadata or {})
            if result.rewrote_payload():
                # The detail page diffs these two. Kept so an older UI can read
                # a newer database.
                metadata.setdefault("before", judged_text(data))
                metadata.setdefault("after", str(result.modified_data))
            if extra_metadata:
                metadata.update(extra_metadata)
            metadata = _capture_redact_metadata(metadata)

            judged_ref: str | None = None
            if data is not None:
                text = _capture_redact_text(judged_text(data)) or ""
                judged_ref = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self._payloads.setdefault(judged_ref, text)

            from fastaiagent._internal.project import safe_get_project_id

            self._rows.append(
                (
                    event_id,
                    self.trace_id,
                    self.span_id,
                    guardrail.name,
                    guardrail.guardrail_type.value,
                    guardrail.position.value,
                    outcome or event_outcome(guardrail, result, effect),
                    result.score,
                    _capture_redact_text(result.message),
                    self.agent_name,
                    datetime.now(tz=timezone.utc).isoformat(),
                    # ``default=str``: a ``code`` guardrail can return anything in
                    # ``metadata``; one unserializable value must not cost the row.
                    json.dumps(metadata, default=str),
                    safe_get_project_id(),
                    result.action,
                    result.action_taken,
                    guardrail.severity,
                    1 if guardrail.floor else 0,
                    effect,
                    json.dumps(rule_snapshot(guardrail), default=str),
                    int(result.execution_time_ms or 0),
                    judged_ref,
                    guardrail_span_id,
                    self.batch_id,
                    self.attempt,
                    final_reason,
                )
            )
        except Exception:
            _log.debug("Could not queue guardrail event for %r", guardrail.name, exc_info=True)
        return event_id

    def flush(self) -> None:
        """Write the queued rows. Best-effort: never raises."""
        if not self._rows:
            return
        rows, payloads = self._rows, self._payloads
        self._rows, self._payloads = [], {}
        try:
            helper = _writer(_resolve_db(self.db_path))
            now = datetime.now(tz=timezone.utc).isoformat()
            with helper.exclusive():
                for payload_id, text in payloads.items():
                    helper.execute(
                        "INSERT OR IGNORE INTO guardrail_payloads (payload_id, text, created_at) "
                        "VALUES (?, ?, ?)",
                        (payload_id, text, now),
                    )
                for row in rows:
                    helper.execute(_INSERT, row)
        except Exception:
            # A locked, read-only or full local.db must never fail a run.
            _log.debug("Failed to write %d guardrail event(s)", len(rows), exc_info=True)


_INSERT = """INSERT INTO guardrail_events
   (event_id, trace_id, span_id, guardrail_name, guardrail_type,
    position, outcome, score, message, agent_name, timestamp, metadata,
    project_id, action, action_taken, severity, floor,
    effect, rule, latency_ms, judged_ref, guardrail_span_id, batch_id, attempt,
    final_reason)
   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def delete_guardrail_events(
    db: SQLiteHelper,
    trace_ids: list[str],
    *,
    orphans_before: str | None = None,
    everything: bool = False,
) -> None:
    """Delete the guardrail rows of these traces, then their unshared payloads.

    Call it **before** deleting the spans or with the trace ids collected
    first: ``fastaiagent traces purge --older-than-days`` deleted the spans and
    then selected guardrail rows through a subquery over the spans it had just
    removed, so it removed none and left judged payloads behind (1.89.0).
    ``orphans_before`` also removes rows with no trace older than that ISO
    timestamp; ``everything`` empties both tables.
    """
    try:
        if everything:
            db.execute("DELETE FROM guardrail_events")
        else:
            ids = list(dict.fromkeys(t for t in trace_ids if t))
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                marks = ",".join("?" * len(chunk))
                db.execute(
                    f"DELETE FROM guardrail_events WHERE trace_id IN ({marks})", tuple(chunk)
                )
            if orphans_before is not None:
                db.execute(
                    "DELETE FROM guardrail_events WHERE trace_id IS NULL AND timestamp < ?",
                    (orphans_before,),
                )
    except Exception:
        _log.debug("Could not delete guardrail events", exc_info=True)
    gc_guardrail_payloads(db)


def gc_guardrail_payloads(db: SQLiteHelper) -> None:
    """Remove judged payloads no event refers to any more. Never raises."""
    try:
        db.execute(
            "DELETE FROM guardrail_payloads WHERE payload_id NOT IN "
            "(SELECT judged_ref FROM guardrail_events WHERE judged_ref IS NOT NULL)"
        )
    except Exception:
        # A database older than v25 has no payload table.
        _log.debug("Could not collect guardrail payloads", exc_info=True)


def finalize_guardrail_events(
    event_ids: list[str],
    *,
    effect: str,
    outcome: str,
    final_reason: str | None,
    db_path: str | None = None,
) -> None:
    """Rewrite rows already written, when the caller refused what they did.

    A row is written as soon as its check runs, so a crash keeps the evidence;
    if the caller then refuses a rewrite or exhausts a re-ask, the row is
    updated in place rather than duplicated. Best-effort.
    """
    if not event_ids or not recording_enabled():
        return
    try:
        helper = _writer(_resolve_db(db_path))
        placeholders = ",".join("?" * len(event_ids))
        helper.execute(
            f"UPDATE guardrail_events SET effect = ?, outcome = ?, final_reason = ? "
            f"WHERE event_id IN ({placeholders})",
            (effect, outcome, final_reason, *event_ids),
        )
    except Exception:
        _log.debug("Failed to finalize guardrail events", exc_info=True)


def log_guardrail_event(
    guardrail: Guardrail,
    result: GuardrailResult,
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    agent_name: str | None = None,
    db_path: str | None = None,
    data: Any = None,
    effect: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    final_reason: str | None = None,
    attempt: int | None = None,
) -> str | None:
    """Persist one guardrail check to ``local.db``. Returns the event id.

    Written whenever tracing is on (1.89.0); ``FASTAIAGENT_UI_ENABLED`` no
    longer gates it. ``data`` is the payload the rule judged.
    """
    if not recording_enabled():
        return None
    if effect is None:
        from fastaiagent.guardrail.actions import effect_of

        effect = effect_of(guardrail, result)
    batch = EventBatch(
        agent_name=agent_name,
        attempt=attempt,
        trace_id=trace_id,
        span_id=span_id,
        db_path=db_path,
    )
    event_id = batch.add(
        guardrail,
        result,
        data=data,
        effect=effect,
        extra_metadata=extra_metadata,
        final_reason=final_reason,
    )
    batch.flush()
    return event_id
