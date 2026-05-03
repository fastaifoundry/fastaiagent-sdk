"""Example 51 — Guardrail Event Detail demo.

Seeds three guardrail events (blocked / filtered / warned) plus the
spans that triggered them directly into the unified local.db. Each row
is shaped so all three detail-page panels render with real content:
PII categories on the blocked event, a before/after rewrite on the
filtered event, and a passing-by-policy note on the warned event.

We hand-write the rows (rather than running a real agent) so the demo
is reproducible without an API key — the data shape mirrors exactly
what the agent runtime emits via ``log_guardrail_event``.

Prereqs:
    pip install 'fastaiagent[ui]'

Run:
    python examples/51_guardrail_events.py
    fastaiagent ui --no-auth
    # Open http://127.0.0.1:7842/guardrails
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastaiagent._internal.config import get_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.ui.db import init_local_db


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _insert_span(
    db: SQLiteHelper,
    *,
    span_id: str,
    trace_id: str,
    name: str,
    attributes: dict,
    when: datetime,
    project_id: str,
) -> None:
    db.execute(
        """INSERT OR REPLACE INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events, project_id)
           VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]', ?)""",
        (
            span_id,
            trace_id,
            name,
            _iso(when),
            _iso(when + timedelta(milliseconds=120)),
            json.dumps(attributes),
            project_id,
        ),
    )


def _insert_event(
    db: SQLiteHelper,
    *,
    event_id: str,
    trace_id: str,
    span_id: str,
    guardrail_name: str,
    guardrail_type: str,
    position: str,
    outcome: str,
    score: float,
    message: str,
    agent_name: str,
    metadata: dict,
    when: datetime,
    project_id: str,
) -> None:
    db.execute(
        """INSERT OR REPLACE INTO guardrail_events
           (event_id, trace_id, span_id, guardrail_name, guardrail_type,
            position, outcome, score, message, agent_name, timestamp,
            metadata, project_id, false_positive, false_positive_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
        (
            event_id,
            trace_id,
            span_id,
            guardrail_name,
            guardrail_type,
            position,
            outcome,
            score,
            message,
            agent_name,
            _iso(when),
            json.dumps(metadata),
            project_id,
        ),
    )


def main() -> int:
    config = get_config()
    db_path = Path(config.local_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(db_path).close()

    project_id = ""  # use the default project so stock fastaiagent ui sees it
    now = datetime.now(tz=timezone.utc)

    with SQLiteHelper(db_path) as db:
        # ── Trace 1 — blocked + filtered events on the same agent.support-bot run.
        t1 = "trace-demo-blocked-" + uuid.uuid4().hex[:8]
        s_root = "span-" + uuid.uuid4().hex[:8]
        _insert_span(
            db,
            span_id=s_root,
            trace_id=t1,
            name="agent.support-bot",
            when=now - timedelta(seconds=5),
            project_id=project_id,
            attributes={
                "agent.name": "support-bot",
                "agent.input": "What's my balance?",
                "agent.output": (
                    "Your balance is $42.10. Email me at "
                    "alice@example.com for follow-up."
                ),
            },
        )
        s_llm = "span-" + uuid.uuid4().hex[:8]
        _insert_span(
            db,
            span_id=s_llm,
            trace_id=t1,
            name="llm.openai.gpt-4o-mini",
            when=now - timedelta(seconds=4),
            project_id=project_id,
            attributes={
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.request.messages": "[user] What's my balance?",
                "gen_ai.response.content": (
                    "Your balance is $42.10. Email me at alice@example.com..."
                ),
            },
        )
        # Blocked: PII detected.
        ev_blocked = "ev-" + uuid.uuid4().hex[:10]
        _insert_event(
            db,
            event_id=ev_blocked,
            trace_id=t1,
            span_id=s_root,
            guardrail_name="no_pii",
            guardrail_type="regex",
            position="output",
            outcome="blocked",
            score=0.0,
            message="PII detected (email)",
            agent_name="support-bot",
            metadata={
                "pii_types": ["email"],
                "match": "alice@example.com",
            },
            when=now - timedelta(seconds=3),
            project_id=project_id,
        )
        # Filtered sibling: same span, different rule rewrote the content.
        ev_filtered = "ev-" + uuid.uuid4().hex[:10]
        _insert_event(
            db,
            event_id=ev_filtered,
            trace_id=t1,
            span_id=s_root,
            guardrail_name="email_redactor",
            guardrail_type="regex",
            position="output",
            outcome="filtered",
            score=1.0,
            message="Redacted 1 email address",
            agent_name="support-bot",
            metadata={
                "before": "alice@example.com",
                "after": "[REDACTED]",
            },
            when=now - timedelta(seconds=3),
            project_id=project_id,
        )

        # ── Trace 2 — warned event on a different run / different rule.
        t2 = "trace-demo-warned-" + uuid.uuid4().hex[:8]
        s2 = "span-" + uuid.uuid4().hex[:8]
        _insert_span(
            db,
            span_id=s2,
            trace_id=t2,
            name="agent.support-bot",
            when=now - timedelta(minutes=2),
            project_id=project_id,
            attributes={
                "agent.name": "support-bot",
                "agent.input": "I'm really frustrated.",
                "agent.output": (
                    "I hear you — let me help. Could you tell me what went wrong?"
                ),
            },
        )
        ev_warned = "ev-" + uuid.uuid4().hex[:10]
        _insert_event(
            db,
            event_id=ev_warned,
            trace_id=t2,
            span_id=s2,
            guardrail_name="toxicity_check",
            guardrail_type="classifier",
            position="output",
            outcome="warned",
            score=0.32,
            message="Below threshold — passed with note",
            agent_name="support-bot",
            metadata={
                "threshold": 0.5,
                "categories": ["frustration"],
            },
            when=now - timedelta(minutes=2),
            project_id=project_id,
        )

    print("Seeded 3 guardrail events into", db_path)
    print()
    print("Open the Local UI:")
    print("  fastaiagent ui --no-auth")
    print()
    print("Direct links once it's running on port 7842:")
    print("  http://127.0.0.1:7842/guardrails")
    print(f"  http://127.0.0.1:7842/guardrail-events/{ev_blocked}")
    print(f"  http://127.0.0.1:7842/guardrail-events/{ev_filtered}")
    print(f"  http://127.0.0.1:7842/guardrail-events/{ev_warned}")
    print()
    print(
        "Tip: open the blocked event, click 'Mark as false positive',"
        " refresh — the flag persists."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
