"""Remove credentials from spans already stored in local.db (1.89.0).

1.89.0 stops new credential leaks at the source (S1, P9). Rows written by an
older SDK can still hold one in two places:

* the root span's ``agent.guardrails`` attribute, when a guardrail was given
  ``llm={"api_key": ...}``;
* an exception event, when a provider echoed a rejected key back.

:func:`scrub_stored_spans` rewrites those rows in place: guardrail ``llm``
configs pass the shareable-field allowlist, exception text is scrubbed, and
any other string attribute has the credentials currently in the environment
replaced verbatim. Backs ``fastaiagent traces scrub-secrets``.

It cannot recall a span that was already sent to the plane. The report counts
those rows separately, because any key in them has to be rotated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastaiagent._internal.scrub import (
    allowlist_llm_config,
    known_credential_values,
    scrub_event_dicts,
    scrub_secrets,
)


@dataclass
class StoredScrubReport:
    """What :func:`scrub_stored_spans` found, and changed unless it was a dry run."""

    spans_scanned: int = 0
    spans_changed: int = 0
    traces_changed: int = 0
    already_sent: int = 0
    """Changed spans marked ``synced=1``: already sent to the plane (or given up
    on), so a copy may exist outside this machine."""
    dry_run: bool = False


def _clean_guardrails(value: Any) -> Any:
    """``agent.guardrails`` (a JSON list of guardrail dicts) with each ``llm``
    config allowlisted. Anything that does not parse is returned unchanged."""
    if not isinstance(value, str):
        return value
    try:
        rules = json.loads(value)
    except ValueError:
        return value
    if not isinstance(rules, list):
        return value
    changed = False
    for rule in rules:
        config = rule.get("config") if isinstance(rule, dict) else None
        if isinstance(config, dict) and isinstance(config.get("llm"), dict):
            clean = allowlist_llm_config(
                config["llm"], f"Guardrail {rule.get('name')!r} (stored)", level=logging.DEBUG
            )
            if clean != config["llm"]:
                config["llm"] = clean
                changed = True
    return json.dumps(rules) if changed else value


def _clean_attributes(attrs: dict[str, Any], secrets: list[str]) -> dict[str, Any]:
    out = dict(attrs)
    if "agent.guardrails" in out:
        out["agent.guardrails"] = _clean_guardrails(out["agent.guardrails"])
    for key, value in out.items():
        if isinstance(value, str):
            # Known credentials only: these are the user's payloads, and a
            # pattern match could rewrite content they meant to keep.
            out[key] = scrub_secrets(value, secrets, patterns=False)
    return out


def scrub_stored_spans(db_path: str | Path, *, dry_run: bool = False) -> StoredScrubReport:
    """Remove credentials from every span in ``db_path``. See the module docstring."""
    from fastaiagent._internal.storage import SQLiteHelper

    report = StoredScrubReport(dry_run=dry_run)
    secrets = known_credential_values()
    changed_traces: set[str] = set()
    with SQLiteHelper(db_path) as db:
        columns = {r["name"] for r in db.fetchall("PRAGMA table_info(spans)")}
        if not columns:
            return report
        synced_col = ", synced" if "synced" in columns else ""
        rows = db.fetchall(f"SELECT span_id, trace_id, attributes, events{synced_col} FROM spans")
        for row in rows:
            report.spans_scanned += 1
            try:
                attrs = json.loads(row["attributes"] or "{}")
                events = json.loads(row["events"] or "[]")
            except ValueError:
                continue
            new_attrs = _clean_attributes(attrs, secrets) if isinstance(attrs, dict) else attrs
            new_events = scrub_event_dicts(events) if isinstance(events, list) else events
            if new_attrs == attrs and new_events == events:
                continue
            report.spans_changed += 1
            changed_traces.add(row["trace_id"])
            if synced_col and row["synced"]:
                report.already_sent += 1
            if not dry_run:
                db.execute(
                    "UPDATE spans SET attributes = ?, events = ? WHERE span_id = ?",
                    (json.dumps(new_attrs), json.dumps(new_events), row["span_id"]),
                )
    report.traces_changed = len(changed_traces)
    return report


__all__ = ["StoredScrubReport", "scrub_stored_spans"]
