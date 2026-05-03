"""Seed Sprint 2 fixtures into a snapshot DB.

Adds:
  * two prompts in the registry (Playground spec)
  * three guardrail events (blocked / filtered / warned) plus their
    triggering spans (Guardrail Event Detail spec)

Designed to overlay on top of ``seed_ui_snapshot.py``. Used by
``scripts/capture-sprint2-screenshots.sh``. Never run in production.
The DB path comes from ``argv[1]``.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _seed_guardrail_events(
    db: SQLiteHelper, project_id: str
) -> dict[str, str]:
    """Insert 3 guardrail events plus the spans that triggered them.

    Returns ``{"blocked": id, "filtered": id, "warned": id}`` so callers
    can build deep-link URLs for the demo / capture script.
    """
    now = datetime.now(tz=timezone.utc)

    # Trace 1 — blocked + filtered on the same agent.support-bot run.
    t1 = "trace-demo-blocked"
    s_root = "span-demo-blocked-root"
    db.execute(
        """INSERT OR REPLACE INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events, project_id)
           VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]', ?)""",
        (
            s_root,
            t1,
            "agent.support-bot",
            _iso(now - timedelta(seconds=5)),
            _iso(now - timedelta(seconds=4, milliseconds=880)),
            json.dumps(
                {
                    "agent.name": "support-bot",
                    "agent.input": "What's my balance?",
                    "agent.output": (
                        "Your balance is $42.10. Email me at "
                        "alice@example.com for follow-up."
                    ),
                }
            ),
            project_id,
        ),
    )
    db.execute(
        """INSERT OR REPLACE INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events, project_id)
           VALUES (?, ?, NULL, 'llm.openai.gpt-4o-mini', ?, ?, 'OK', ?,
                   '[]', ?)""",
        (
            "span-demo-blocked-llm",
            t1,
            _iso(now - timedelta(seconds=4)),
            _iso(now - timedelta(seconds=3, milliseconds=900)),
            json.dumps(
                {
                    "gen_ai.request.model": "gpt-4o-mini",
                    "gen_ai.request.messages": "[user] What's my balance?",
                    "gen_ai.response.content": (
                        "Your balance is $42.10. Email me at "
                        "alice@example.com..."
                    ),
                }
            ),
            project_id,
        ),
    )

    blocked_id = "ev-demo-blocked-" + uuid.uuid4().hex[:6]
    filtered_id = "ev-demo-filtered-" + uuid.uuid4().hex[:6]

    db.execute(
        """INSERT OR REPLACE INTO guardrail_events
           (event_id, trace_id, span_id, guardrail_name, guardrail_type,
            position, outcome, score, message, agent_name, timestamp,
            metadata, project_id, false_positive, false_positive_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
        (
            blocked_id,
            t1,
            s_root,
            "no_pii",
            "regex",
            "output",
            "blocked",
            0.0,
            "PII detected (email)",
            "support-bot",
            _iso(now - timedelta(seconds=3)),
            json.dumps(
                {
                    "pii_types": ["email"],
                    "match": "alice@example.com",
                }
            ),
            project_id,
        ),
    )
    db.execute(
        """INSERT OR REPLACE INTO guardrail_events
           (event_id, trace_id, span_id, guardrail_name, guardrail_type,
            position, outcome, score, message, agent_name, timestamp,
            metadata, project_id, false_positive, false_positive_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
        (
            filtered_id,
            t1,
            s_root,
            "email_redactor",
            "regex",
            "output",
            "filtered",
            1.0,
            "Redacted 1 email address",
            "support-bot",
            _iso(now - timedelta(seconds=3)),
            json.dumps(
                {
                    "before": "alice@example.com",
                    "after": "[REDACTED]",
                }
            ),
            project_id,
        ),
    )

    # Trace 2 — warned event on a different run.
    t2 = "trace-demo-warned"
    s2 = "span-demo-warned-root"
    db.execute(
        """INSERT OR REPLACE INTO spans
           (span_id, trace_id, parent_span_id, name, start_time, end_time,
            status, attributes, events, project_id)
           VALUES (?, ?, NULL, ?, ?, ?, 'OK', ?, '[]', ?)""",
        (
            s2,
            t2,
            "agent.support-bot",
            _iso(now - timedelta(minutes=2)),
            _iso(now - timedelta(minutes=2) + timedelta(milliseconds=300)),
            json.dumps(
                {
                    "agent.name": "support-bot",
                    "agent.input": "I'm really frustrated.",
                    "agent.output": (
                        "I hear you — let me help. Could you tell me "
                        "what went wrong?"
                    ),
                }
            ),
            project_id,
        ),
    )

    warned_id = "ev-demo-warned-" + uuid.uuid4().hex[:6]
    db.execute(
        """INSERT OR REPLACE INTO guardrail_events
           (event_id, trace_id, span_id, guardrail_name, guardrail_type,
            position, outcome, score, message, agent_name, timestamp,
            metadata, project_id, false_positive, false_positive_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
        (
            warned_id,
            t2,
            s2,
            "toxicity_check",
            "classifier",
            "output",
            "warned",
            0.32,
            "Below threshold — passed with note",
            "support-bot",
            _iso(now - timedelta(minutes=2)),
            json.dumps(
                {
                    "threshold": 0.5,
                    "categories": ["frustration"],
                }
            ),
            project_id,
        ),
    )

    return {
        "blocked": blocked_id,
        "filtered": filtered_id,
        "warned": warned_id,
    }


def seed(db_path: Path, project_id: str = "sprint2-demo") -> None:
    now = datetime.now(tz=timezone.utc).isoformat()

    prompts = [
        {
            "slug": "support-greeting",
            "template": (
                "You are a friendly support agent for {{company}}.\n"
                "A customer named {{customer_name}} asks about {{topic}}.\n"
                "Reply in 2-3 sentences, polite and concrete."
            ),
            "variables": ["company", "customer_name", "topic"],
            "metadata": {"purpose": "Customer support opener"},
        },
        {
            "slug": "image-describe",
            "template": (
                "Describe what you see in the attached image. "
                "Pay particular attention to {{focus}}."
            ),
            "variables": ["focus"],
            "metadata": {"purpose": "Vision-model image describer"},
        },
    ]

    with SQLiteHelper(db_path) as db:
        for p in prompts:
            db.execute(
                """INSERT OR REPLACE INTO prompts
                   (slug, latest_version, created_at, updated_at, project_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (p["slug"], "1", now, now, project_id),
            )
            db.execute(
                """INSERT OR REPLACE INTO prompt_versions
                   (slug, version, template, variables, fragments, metadata,
                    created_at, created_by, project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    p["slug"],
                    "1",
                    p["template"],
                    json.dumps(p["variables"]),
                    json.dumps([]),
                    json.dumps(p["metadata"]),
                    now,
                    "code",
                    project_id,
                ),
            )

        guardrail_ids = _seed_guardrail_events(db, project_id)

    print(
        f"✓ seeded {len(prompts)} prompts into {db_path} "
        f"(project_id={project_id})"
    )
    print(f"✓ seeded 3 guardrail events: {guardrail_ids}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--project-id", default="sprint2-demo")
    args = parser.parse_args()
    seed(args.db_path, project_id=args.project_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
