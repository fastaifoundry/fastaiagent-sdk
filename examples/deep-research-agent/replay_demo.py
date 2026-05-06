"""
Replay demo — re-render a past research run from local.db.

Every ``run_deep_research`` call persists structured spans under the
``fastaiagent.research.*`` namespace. This script reads the most recent
``deep_research.session`` span, pulls the brief / plan / findings payloads
back out of the span attributes, and prints a structured reconstruction.

Why bother: it's the smallest possible illustration that traces in this
SDK are not just an audit log — they're a queryable store. PR B's
``fastaiagent learn`` CLI builds on the same primitive.

Usage:
    python replay_demo.py                # re-render the most recent run
    python replay_demo.py --trace-id <hex>
"""

from __future__ import annotations

import argparse
import json

from spans import (
    ATTR_BRIEF,
    ATTR_FINDINGS,
    ATTR_PLAN,
    ATTR_SUBTOPIC,
    ATTR_TOPIC,
)

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper


def _open_db() -> SQLiteHelper:
    return SQLiteHelper(get_config().local_db_path)


def _latest_session_trace_id(db: SQLiteHelper) -> str | None:
    rows = db.fetchall(
        "SELECT trace_id, start_time FROM spans "
        "WHERE name = 'deep_research.session' "
        "ORDER BY start_time DESC LIMIT 1"
    )
    if not rows:
        return None
    return rows[0]["trace_id"]


def _spans_for_trace(db: SQLiteHelper, trace_id: str) -> list[dict]:
    return db.fetchall(
        "SELECT name, attributes FROM spans WHERE trace_id = ? ORDER BY start_time ASC",
        (trace_id,),
    )


def replay(trace_id: str | None = None) -> int:
    db = _open_db()
    tid = trace_id or _latest_session_trace_id(db)
    if not tid:
        print("No deep_research.session traces found in local.db.")
        print("Run `python agent.py --topic '<your topic>'` first.")
        return 1

    print(f"Replaying trace: {tid}\n")
    rows = _spans_for_trace(db, tid)
    print(f"  {len(rows)} spans in this trace\n")

    for row in rows:
        attrs = json.loads(row["attributes"]) if row["attributes"] else {}
        name = row["name"]
        if name == "deep_research.session":
            print(f"[session] topic: {attrs.get(ATTR_TOPIC, '?')}")
            plan_raw = attrs.get(ATTR_PLAN)
            if plan_raw:
                plan = json.loads(plan_raw)
                print(f"          plan: {len(plan.get('subtopics', []))} subtopics")
        elif name == "deep_research.scope":
            brief_raw = attrs.get(ATTR_BRIEF)
            if brief_raw:
                brief = json.loads(brief_raw)
                print(f"[scope]   {brief.get('summary', '')[:200]}")
        elif name == "deep_research.research":
            sub = attrs.get(ATTR_SUBTOPIC, "?")
            findings_raw = attrs.get(ATTR_FINDINGS)
            if findings_raw:
                findings = json.loads(findings_raw)
                print(
                    f"[research]  • {sub}: "
                    f"{len(findings.get('citations', []))} citations"
                )
        elif name == "deep_research.write":
            chars = attrs.get("fastaiagent.research.report.chars", "?")
            cites = attrs.get("fastaiagent.research.report.citations", "?")
            print(f"[write]   {chars} chars, {cites} citations")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a past deep-research run")
    parser.add_argument("--trace-id", help="Specific trace to replay (default: latest)")
    args = parser.parse_args()
    raise SystemExit(replay(args.trace_id))


if __name__ == "__main__":
    main()
