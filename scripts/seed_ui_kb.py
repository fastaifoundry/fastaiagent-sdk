"""Seed a LocalKB for docs screenshots.

Creates two short markdown files, ingests them into ``<kb_root>/docs-demo``,
and (optionally) writes one retrieval span into the given ``local.db`` so the
Lineage tab has content to render.

Used by ``scripts/capture-ui-screenshots.sh`` — never run in production.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastaiagent._internal.storage import SQLiteHelper  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402

_DOCS = {
    "refund-policy.md": (
        "# Refund policy\n\n"
        "Refunds are processed within 7 business days after we receive the "
        "return. Customers must include the original packing slip. Items "
        "marked final-sale are non-refundable.\n"
    ),
    "shipping-policy.md": (
        "# Shipping policy\n\n"
        "Standard shipping is free on orders over $50. Expedited shipping is "
        "available at checkout for an extra charge. Orders placed before 3pm "
        "local time ship the same day.\n"
    ),
    "return-window.md": (
        "# Return window\n\n"
        "You have 30 days from the delivery date to initiate a return. "
        "Start a return from the order history page in your account.\n"
    ),
}


def seed(kb_root: Path, db_path: Path | None) -> None:
    from fastaiagent.kb.local import LocalKB

    kb_root.mkdir(parents=True, exist_ok=True)
    source_dir = kb_root.parent / "_kb-source"
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _DOCS.items():
        (source_dir / name).write_text(content)

    kb = LocalKB(
        name="support-kb",
        path=str(kb_root),
        chunk_size=240,
        chunk_overlap=30,
    )
    for name in _DOCS:
        kb.add(str(source_dir / name))

    if db_path is not None:
        _seed_retrieval_spans(db_path)


def _seed_retrieval_spans(db_path: Path) -> None:
    init_local_db(db_path).close()
    now = datetime.now(tz=timezone.utc)
    with SQLiteHelper(db_path) as db:
        for i in range(3):
            trace_id = f"kb-lineage-{i:04d}"
            root_span = uuid.uuid4().hex
            child_span = uuid.uuid4().hex
            start = (now - __import__("datetime").timedelta(minutes=5 * (i + 1))).isoformat()
            end = now.isoformat()
            db.execute(
                """INSERT OR REPLACE INTO spans (span_id, trace_id, parent_span_id, name,
                                       start_time, end_time, status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
                (
                    root_span,
                    trace_id,
                    None,
                    "agent.support-bot",
                    start,
                    end,
                    json.dumps({"agent.name": "support-bot"}),
                ),
            )
            db.execute(
                """INSERT OR REPLACE INTO spans (span_id, trace_id, parent_span_id, name,
                                       start_time, end_time, status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
                (
                    child_span,
                    trace_id,
                    root_span,
                    "retrieval.support-kb",
                    start,
                    end,
                    json.dumps(
                        {
                            "agent.name": "support-bot",
                            "retrieval.backend": "local",
                            "retrieval.top_k": 3,
                        }
                    ),
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kb_root",
        type=Path,
        help="Directory that will hold <kb_root>/support-kb/kb.sqlite",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Optional local.db to seed retrieval spans into",
    )
    args = parser.parse_args()
    seed(args.kb_root, args.db)
    print(f"✓ seeded LocalKB under {args.kb_root}/support-kb/")


if __name__ == "__main__":
    main()
