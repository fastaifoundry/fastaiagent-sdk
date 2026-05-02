"""Self-contained, human-readable JSON export of a trace.

Used by:

* ``GET /api/traces/{trace_id}/export?include_attachments=…&include_checkpoint_state=…``
* ``fastaiagent export-trace --trace-id <id> --output <path>``

The two paths build the same payload via :func:`build_export_payload`,
so the JSON schema is single-sourced. Attachments default to
metadata-only — set ``include_attachments=True`` to base64-embed the
bytes.

Spec reference: ``claude_files/sprint1-ui.md`` (Feature 5).
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent._version import __version__
from fastaiagent.ui.attrs import attr, trace_cost_usd
from fastaiagent.ui.pricing import compute_cost_usd

EXPORT_VERSION = "1.0"


def build_export_payload(
    db: SQLiteHelper,
    trace_id: str,
    *,
    include_attachments: bool = False,
    include_checkpoint_state: bool = False,
) -> dict[str, Any]:
    """Read every span / checkpoint / attachment for ``trace_id`` and
    return a JSON-safe dict matching the spec.

    Raises :class:`KeyError` if the trace does not exist.
    """
    span_rows = db.fetchall(
        """SELECT span_id, trace_id, parent_span_id, name, start_time,
                  end_time, status, attributes, events
           FROM spans WHERE trace_id = ? ORDER BY start_time ASC""",
        (trace_id,),
    )
    if not span_rows:
        raise KeyError(trace_id)

    spans: list[dict[str, Any]] = []
    root: dict[str, Any] | None = None
    total_tokens = 0
    total_cost = 0.0
    execution_id: str | None = None
    runner_type = "agent"
    runner_name: str | None = None
    durable = False

    for row in span_rows:
        try:
            attrs = json.loads(row.get("attributes") or "{}") or {}
        except json.JSONDecodeError:
            attrs = {}
        try:
            events = json.loads(row.get("events") or "[]")
        except json.JSONDecodeError:
            events = []

        # Per-span cost: reported wins, else compute from tokens.
        reported = trace_cost_usd(attrs)
        if reported is not None:
            cost = reported
        else:
            cost = (
                compute_cost_usd(
                    attrs.get("gen_ai.request.model"),
                    attrs.get("gen_ai.usage.input_tokens"),
                    attrs.get("gen_ai.usage.output_tokens"),
                )
                or 0.0
            )

        in_t = attrs.get("gen_ai.usage.input_tokens")
        out_t = attrs.get("gen_ai.usage.output_tokens")
        tokens = (
            {"input": int(in_t or 0), "output": int(out_t or 0)}
            if in_t is not None or out_t is not None
            else None
        )

        duration_ms = _duration_ms(row.get("start_time"), row.get("end_time"))
        spans.append(
            {
                "span_id": row["span_id"],
                "name": row.get("name") or "",
                "parent_span_id": row.get("parent_span_id"),
                "status": row.get("status") or "OK",
                "started_at": row.get("start_time"),
                "duration_ms": duration_ms,
                "input": _select_input(attrs),
                "output": _select_output(attrs),
                "attributes": _strip_io(attrs),
                "events": events,
                "model": attrs.get("gen_ai.request.model"),
                "tokens": tokens,
                "cost": round(cost, 6) if cost else None,
            }
        )

        if not row.get("parent_span_id"):
            root = row
            runner_type = str(attr(attrs, "runner.type") or "agent")
            runner_name = (
                attr(attrs, "agent.name")
                or attr(attrs, "chain.name")
                or attr(attrs, "swarm.name")
                or attr(attrs, "supervisor.name")
            )
            execution_id = (
                attr(attrs, "chain.execution_id")
                or attr(attrs, "agent.execution_id")
                or None
            )
            durable = execution_id is not None

        if cost:
            total_cost += cost
        if tokens:
            total_tokens += tokens.get("input", 0) + tokens.get("output", 0)

    if root is None:
        root = span_rows[0]

    trace_payload = {
        "trace_id": trace_id,
        "name": root.get("name") or "",
        "status": root.get("status") or "OK",
        "started_at": root.get("start_time"),
        "duration_ms": _duration_ms(root.get("start_time"), root.get("end_time")),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6) if total_cost else 0.0,
        "workflow_type": runner_type,
        "runner": runner_name,
        "execution_id": execution_id,
        "durable": durable,
    }

    # Checkpoints (metadata always, full state only when asked).
    checkpoints: list[dict[str, Any]] = []
    if execution_id:
        cp_rows = db.fetchall(
            """SELECT checkpoint_id, parent_checkpoint_id, chain_name,
                      execution_id, node_id, node_index, status,
                      state_snapshot, node_input, node_output,
                      interrupt_reason, interrupt_context, agent_path, created_at
               FROM checkpoints WHERE execution_id = ?
               ORDER BY node_index ASC, created_at ASC""",
            (execution_id,),
        )
        for cp in cp_rows:
            payload = {
                "checkpoint_id": cp["checkpoint_id"],
                "node_id": cp.get("node_id"),
                "step": cp.get("node_index"),
                "status": cp.get("status"),
                "interrupt_reason": cp.get("interrupt_reason"),
                "agent_path": cp.get("agent_path"),
                "created_at": cp.get("created_at"),
            }
            if include_checkpoint_state:
                payload["state_snapshot"] = _maybe_load(cp.get("state_snapshot"))
                payload["node_input"] = _maybe_load(cp.get("node_input"))
                payload["node_output"] = _maybe_load(cp.get("node_output"))
                payload["interrupt_context"] = _maybe_load(
                    cp.get("interrupt_context")
                )
            checkpoints.append(payload)

    # Multimodal attachment metadata (always) + bytes (opt-in).
    attachments: list[dict[str, Any]] = []
    att_rows = db.fetchall(
        """SELECT attachment_id, span_id, media_type, size_bytes,
                  thumbnail, full_data, metadata_json, created_at
           FROM trace_attachments WHERE trace_id = ?
           ORDER BY created_at ASC""",
        (trace_id,),
    )
    for att in att_rows:
        payload = {
            "attachment_id": att["attachment_id"],
            "span_id": att.get("span_id"),
            "media_type": att.get("media_type"),
            "size_bytes": att.get("size_bytes"),
            "metadata": _maybe_load(att.get("metadata_json")),
            "created_at": att.get("created_at"),
        }
        if include_attachments:
            blob = att.get("full_data") or att.get("thumbnail") or b""
            payload["included"] = True
            payload["attachment_data"] = base64.b64encode(blob).decode("ascii")
        else:
            payload["included"] = False
            payload["note"] = (
                "Attachment data excluded from export. Use "
                "include_attachments=True to embed base64 bytes."
            )
        attachments.append(payload)

    return {
        "export_version": EXPORT_VERSION,
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "sdk_version": __version__,
        "trace": trace_payload,
        "spans": spans,
        "checkpoints": checkpoints,
        "multimodal_attachments": attachments,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INPUT_KEYS = {
    "gen_ai.request.messages",
    "gen_ai.request.prompt",
    "agent.input",
    "chain.input",
    "swarm.input",
    "supervisor.input",
    "tool.input",
    "tool.args",
    "retrieval.query",
    "input",
}

_OUTPUT_KEYS = {
    "gen_ai.response.content",
    "gen_ai.response.tool_calls",
    "agent.output",
    "chain.output",
    "swarm.output",
    "supervisor.output",
    "tool.output",
    "tool.result",
    "retrieval.doc_ids",
    "retrieval.result_count",
    "output",
}


def _select_input(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: _maybe_load(v) for k, v in attrs.items() if k in _INPUT_KEYS}


def _select_output(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: _maybe_load(v) for k, v in attrs.items() if k in _OUTPUT_KEYS}


def _strip_io(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _maybe_load(v)
        for k, v in attrs.items()
        if k not in _INPUT_KEYS and k not in _OUTPUT_KEYS
    }


def _maybe_load(value: Any) -> Any:
    """Decode JSON-stringified attribute values (the SDK stores arrays/dicts
    as JSON strings to keep OTel happy)."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _duration_ms(start: Any, end: Any) -> int | None:
    if not start or not end:
        return None
    try:
        a = datetime.fromisoformat(str(start))
        b = datetime.fromisoformat(str(end))
    except (TypeError, ValueError):
        return None
    return int((b - a).total_seconds() * 1000)


# ---------------------------------------------------------------------------
# CLI helper — used by ``fastaiagent export-trace``
# ---------------------------------------------------------------------------


def export_trace_to_file(
    db_path: str | Path,
    trace_id: str,
    output: str | Path,
    *,
    include_attachments: bool = False,
    include_checkpoint_state: bool = False,
) -> Path:
    """Build the export payload and write it to ``output`` as JSON.

    Returns the absolute :class:`Path` written.
    """
    out_path = Path(output).expanduser().resolve()
    db = SQLiteHelper(str(db_path))
    try:
        payload = build_export_payload(
            db,
            trace_id,
            include_attachments=include_attachments,
            include_checkpoint_state=include_checkpoint_state,
        )
    finally:
        db.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path
