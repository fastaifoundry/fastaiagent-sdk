"""Curate eval datasets from captured agent traces.

Selects captured traces from the local SQLite DB by a filter (favorites, noted,
guardrail-fired, failed, or all) and turns every ``agent.<name>`` span — whether
it is a trace root (a plain agent run) or nested inside a chain / supervisor /
swarm run — into an eval dataset item via the ``agent.input`` / ``agent.output``
span attributes. One agent invocation → one case.

The captured agent output is only a *gold* answer for known-good traces. For the
failure filters (``guardrail`` / ``failed``) the captured output is the *bad*
answer to fix, so those cases are emitted with ``expected_output=""`` and
``needs_review=True`` (the bad output is preserved as ``actual_output``). The
``mark_output_as_expected`` argument overrides this per run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AGENT_SPAN_PREFIX = "agent."

# Filters that treat the captured agent output as the gold answer by default.
_GOOD_FILTERS = {"all", "favorites", "noted"}
_FAILURE_FILTERS = {"guardrail", "failed"}
_VALID_FILTERS = _GOOD_FILTERS | _FAILURE_FILTERS
_VALID_DEDUP = {"none", "input"}
# How aggressively to drop infrastructure-errored runs from the gold set:
#   "agent" — drop only when the agent produced no usable output (the reliable,
#             agent-attributable signal); keep a run where a tool errored but the
#             agent recovered and still produced a clean answer.
#   "trace" — additionally drop any run whose trace carries an error-status span.
_VALID_INFRA_MODES = {"agent", "trace"}
# Span statuses that indicate a clean (non-error) span.
_OK_STATUSES = {"OK", "UNSET"}


class CuratedDataset(list):
    """A list of curated eval items that also reports curation **coverage**.

    Behaves exactly like ``list`` — a drop-in anywhere a dataset list is used —
    but additionally reports how many captured runs were dropped as
    infrastructure-errored. That lets a caller tell "20 clean cases" from
    "20 clean cases, 200 dropped as errored": the latter signals an unhealthy
    agent or a trace-capture problem worth investigating, independent of any
    optimization run built on top of it.
    """

    def __init__(
        self,
        items: Any = (),
        *,
        emitted: int = 0,
        infra_excluded: int = 0,
        needs_review: int = 0,
        traces: int = 0,
    ) -> None:
        super().__init__(items)
        self.emitted = emitted
        self.infra_excluded = infra_excluded
        self.needs_review = needs_review
        self.traces = traces

    def coverage_summary(self) -> str:
        return (
            f"{self.emitted} case(s) from {self.traces} trace(s); "
            f"{self.infra_excluded} dropped as infra-errored, "
            f"{self.needs_review} need review."
        )


def curate_from_traces(
    *,
    filter: str = "all",
    agent: str | None = None,
    since_hours: float | None = None,
    limit: int | None = 200,
    trace_ids: list[str] | None = None,
    mark_output_as_expected: bool | None = None,
    db_path: str | Path | None = None,
    dedup_by: str = "none",
    exclude_infra_errors: str = "agent",
) -> CuratedDataset:
    """Build eval dataset item dicts from captured agent traces.

    Args:
        filter: ``all`` | ``favorites`` | ``noted`` | ``guardrail`` | ``failed``.
        agent: only keep ``agent.<name>`` spans for this agent name.
        since_hours: only traces whose agent span started within the last N hours.
        limit: cap on number of traces read (most recent first); ``None`` for all.
        trace_ids: explicit trace ids to curate (overrides ``filter`` selection).
        mark_output_as_expected: ``True`` → captured output becomes
            ``expected_output``; ``False`` → ``needs_review`` with empty expected.
            ``None`` (default) auto-picks by filter (good filters → True).
        db_path: local.db path; defaults to ``get_config().local_db_path``.
        dedup_by: ``none`` (one case per agent span) or ``input`` (drop dup inputs).
        exclude_infra_errors: how to drop infrastructure-errored runs from the
            gold set (only applies when the captured output is treated as gold).
            ``agent`` (default) drops a run only when the agent produced no usable
            output — the reliable, agent-attributable signal — keeping a run where
            a tool errored but the agent recovered with a clean answer. ``trace``
            additionally drops any run whose trace carries an error-status span.
            The ``guardrail`` / ``failed`` filters are unaffected (they
            intentionally surface bad runs as ``needs_review``).

    Returns:
        A list of item dicts ready for :class:`~fastaiagent.eval.Dataset`.
    """
    if filter not in _VALID_FILTERS:
        raise ValueError(f"Unknown filter '{filter}'. Choose from {sorted(_VALID_FILTERS)}.")
    if dedup_by not in _VALID_DEDUP:
        raise ValueError(f"dedup_by must be one of {sorted(_VALID_DEDUP)}.")
    if exclude_infra_errors not in _VALID_INFRA_MODES:
        raise ValueError(
            f"exclude_infra_errors must be one of {sorted(_VALID_INFRA_MODES)}."
        )

    from fastaiagent._internal.config import get_config
    from fastaiagent.trace import TraceStore
    from fastaiagent.ui.db import init_local_db

    resolved = str(db_path) if db_path is not None else get_config().local_db_path
    if mark_output_as_expected is None:
        mark_output_as_expected = filter in _GOOD_FILTERS

    # init_local_db ensures the full schema exists (idempotent) so the filter
    # tables are queryable even on a DB the tracer created with only `spans`.
    db = init_local_db(resolved)
    try:
        if trace_ids is not None:
            selected = list(dict.fromkeys(trace_ids))  # de-dup, preserve order
            truncated = 0
        else:
            selected, truncated = _select_trace_ids(
                db, filter=filter, agent=agent, since_hours=since_hours, limit=limit
            )
        notes = _notes_map(db, selected) if filter == "noted" else {}
        reasons = _guardrail_reasons(db, selected) if filter == "guardrail" else {}
    finally:
        db.close()

    store = TraceStore(db_path=resolved)
    items: list[dict[str, Any]] = []
    seen_inputs: set[str] = set()
    n_traces = 0
    n_spans = 0
    n_skipped = 0
    n_infra_excluded = 0

    for tid in selected:
        try:
            trace = store.get_trace(tid)
        except Exception:
            logger.debug("curate: could not load trace %s", tid, exc_info=True)
            n_skipped += 1
            continue
        agent_spans = [s for s in trace.spans if (s.name or "").startswith(_AGENT_SPAN_PREFIX)]
        if not agent_spans:
            n_skipped += 1
            continue
        n_traces += 1
        # Trace-level infra signal (used only by exclude_infra_errors="trace").
        # The agent root span isn't always stamped ERROR on a downstream failure,
        # so "any span carries an error status" is the reliable trace-level cue.
        trace_has_error = any((s.status or "OK") not in _OK_STATUSES for s in trace.spans)
        for span in agent_spans:
            n_spans += 1
            attrs = span.attributes or {}
            name = attrs.get("agent.name") or (span.name or "")[len(_AGENT_SPAN_PREFIX) :]
            name = name or "agent"
            if agent is not None and name != agent:
                continue
            input_val = attrs.get("agent.input")
            if input_val is None or str(input_val).strip() == "":
                n_skipped += 1
                continue
            output_text = str(attrs.get("agent.output", "") or "")
            # Part 1 — don't curate gold from an infrastructure-errored run. Keyed
            # on agent-output presence (the reliable signal); span status only
            # corroborates the drop reason. Skipped on the failure filters, which
            # deliberately surface bad runs for review.
            if mark_output_as_expected:
                no_output = not output_text.strip()
                drop = no_output or (exclude_infra_errors == "trace" and trace_has_error)
                if drop:
                    n_infra_excluded += 1
                    reason = "no agent output" if no_output else "trace had an error-status span"
                    if no_output and trace_has_error:
                        reason += " (error status on the run)"
                    logger.debug(
                        "curate: dropped infra-errored run trace=%s span=%s — %s",
                        tid,
                        span.span_id,
                        reason,
                    )
                    continue
            item = _build_item(
                input_text=str(input_val),
                output_text=output_text,
                trace_id=tid,
                span_id=span.span_id,
                agent_name=name,
                filter=filter,
                mark_output_as_expected=mark_output_as_expected,
                media_count=_as_int(attrs.get("fastaiagent.input.media_count")),
                note=notes.get(tid),
                guardrail_reason=reasons.get(tid),
            )
            if dedup_by == "input":
                if item["input"] in seen_inputs:
                    continue
                seen_inputs.add(item["input"])
            items.append(item)

    needs_review = sum(1 for it in items if it.get("needs_review"))
    if truncated:
        logger.warning(
            "curate: capped at limit=%s; %s more matching trace(s) not read.",
            limit,
            truncated,
        )
    logger.info(
        "curate(filter=%s): %s trace(s) -> %s agent span(s) -> %s case(s) "
        "(%s need review, %s skipped, %s dropped as infra-errored)",
        filter,
        n_traces,
        n_spans,
        len(items),
        needs_review,
        n_skipped,
        n_infra_excluded,
    )
    if n_infra_excluded:
        logger.warning(
            "curate(filter=%s): dropped %s run(s) as infrastructure-errored "
            "(no usable agent output%s). Emitted %s clean case(s) — check "
            "`.infra_excluded` on the result; a high drop rate signals an "
            "unhealthy agent or a trace-capture issue.",
            filter,
            n_infra_excluded,
            "/error status" if exclude_infra_errors == "trace" else "",
            len(items),
        )
    return CuratedDataset(
        items,
        emitted=len(items),
        infra_excluded=n_infra_excluded,
        needs_review=needs_review,
        traces=n_traces,
    )


def _select_trace_ids(
    db: Any,
    *,
    filter: str,
    agent: str | None,
    since_hours: float | None,
    limit: int | None,
) -> tuple[list[str], int]:
    """Return (trace_ids ordered most-recent-first, truncated_count)."""
    where = ["s.name LIKE 'agent.%'"]
    params: list[Any] = []

    if since_hours is not None:
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=since_hours)).isoformat()
        where.append("s.start_time >= ?")
        params.append(cutoff)

    if agent is not None:
        # Narrow to traces that contain a span for this agent (extraction also filters).
        where.append("EXISTS (SELECT 1 FROM spans a WHERE a.trace_id = s.trace_id AND a.name = ?)")
        params.append(f"agent.{agent}")

    if filter == "favorites":
        where.append("s.trace_id IN (SELECT trace_id FROM trace_favorites)")
    elif filter == "noted":
        where.append("s.trace_id IN (SELECT trace_id FROM trace_notes)")
    elif filter == "guardrail":
        where.append(
            "s.trace_id IN (SELECT DISTINCT trace_id FROM guardrail_events WHERE outcome = 'fail')"
        )
    elif filter == "failed":
        # Best-effort: a trace is "failed" if any span carries an error status.
        # (The agent root span does not always get ERROR set on exception, so the
        # `guardrail` filter is the more reliable failure signal — see docs.)
        where.append(
            "s.trace_id IN (SELECT DISTINCT trace_id FROM spans "
            "WHERE status NOT IN ('OK', 'UNSET'))"
        )

    sql = (
        "SELECT s.trace_id AS trace_id, MAX(s.start_time) AS ts FROM spans s "
        f"WHERE {' AND '.join(where)} GROUP BY s.trace_id ORDER BY ts DESC"
    )
    rows = db.fetchall(sql, tuple(params))
    ids = [r["trace_id"] for r in rows]
    truncated = 0
    if limit is not None and len(ids) > limit:
        truncated = len(ids) - limit
        ids = ids[:limit]
    return ids, truncated


def _notes_map(db: Any, trace_ids: list[str]) -> dict[str, str]:
    if not trace_ids:
        return {}
    wanted = set(trace_ids)
    rows = db.fetchall("SELECT trace_id, note FROM trace_notes")
    return {r["trace_id"]: r["note"] for r in rows if r["trace_id"] in wanted and r["note"]}


def _guardrail_reasons(db: Any, trace_ids: list[str]) -> dict[str, str]:
    if not trace_ids:
        return {}
    wanted = set(trace_ids)
    rows = db.fetchall(
        "SELECT trace_id, guardrail_name, message FROM guardrail_events "
        "WHERE outcome = 'fail' ORDER BY timestamp"
    )
    out: dict[str, str] = {}
    for r in rows:
        tid = r["trace_id"]
        if tid not in wanted or tid in out:
            continue
        gname = r["guardrail_name"] or "guardrail"
        msg = (r["message"] or "").strip()
        out[tid] = f"guardrail '{gname}' fired" + (f": {msg}" if msg else "")
    return out


def _build_item(
    *,
    input_text: str,
    output_text: str,
    trace_id: str,
    span_id: str,
    agent_name: str,
    filter: str,
    mark_output_as_expected: bool,
    media_count: int,
    note: str | None,
    guardrail_reason: str | None,
) -> dict[str, Any]:
    needs_review = False
    reasons: list[str] = []

    if not mark_output_as_expected:
        needs_review = True
        if guardrail_reason:
            reasons.append(guardrail_reason)
        elif filter == "failed":
            reasons.append("trace marked failed (error status)")
        else:
            reasons.append("needs a human-supplied expected output")
    if not output_text.strip():
        needs_review = True
        reasons.append("captured output was empty")
    if media_count > 0:
        needs_review = True
        reasons.append("multimodal input: media not captured (text summary only)")

    item: dict[str, Any] = {
        "input": input_text,
        "expected_output": "" if needs_review else output_text,
        "trace_id": trace_id,
        "source_trace_id": trace_id,
        "span_id": span_id,
        "agent_name": agent_name,
        "source": f"curated:{filter}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if needs_review:
        item["needs_review"] = True
        item["actual_output"] = output_text
        item["reason"] = "; ".join(reasons)
    if note:
        item["note"] = note
    return item


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
