"""Seed Sprint 3 fixtures into a snapshot DB.

Three sets of data, all real rows the UI reads through the standard
endpoints:

  1. Two comparable traces (terse vs verbose system prompts) so the
     trace comparison view (Feature 1) has data to diff.
  2. Two datasets — one plain-text, one multimodal — already present
     under ``<db_dir>/datasets/`` so the editor lists them on first
     load (Feature 2).
  3. Twenty assorted traces with varying agents, costs, durations,
     and prompt/response text so the richer filter bar (Feature 3) has
     bite for FTS5 + cost/duration ranges.

Designed to overlay on top of ``seed_ui_snapshot.py``. Used by
``scripts/capture-sprint3-screenshots.sh``. Never run in production.
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
from fastaiagent.ui.db import init_local_db  # noqa: E402


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _seed_compare_pair(db: SQLiteHelper, project_id: str) -> tuple[str, str]:
    """Two traces with overlapping spans + one extra in B + an output diff."""
    base = datetime.now(tz=timezone.utc) - timedelta(days=3)
    trace_a = "trace-compare-terse"
    trace_b = "trace-compare-verbose"

    def _add(
        trace_id: str,
        idx: int,
        name: str,
        attrs: dict,
        duration_ms: int,
        parent_id: str | None,
    ) -> str:
        span_id = f"{trace_id}-s{idx}"
        start = base + timedelta(seconds=idx * 0.5) + (timedelta(days=3) if "verbose" in trace_id else timedelta())
        end = start + timedelta(milliseconds=duration_ms)
        db.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events, project_id)
               VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]', ?)""",
            (
                span_id,
                trace_id,
                parent_id,
                name,
                _iso(start),
                _iso(end),
                json.dumps(attrs),
                project_id,
            ),
        )
        return span_id

    # Trace A — three spans, terse system prompt.
    root_a = _add(
        trace_a,
        0,
        "agent.support",
        {
            "agent.name": "support",
            "fastaiagent.cost.total_usd": 0.012,
            "gen_ai.usage.input_tokens": 80,
            "gen_ai.usage.output_tokens": 40,
        },
        100,
        None,
    )
    _add(
        trace_a,
        1,
        "retrieval.support-kb",
        {"kb.name": "support", "kb.results": 3},
        80,
        root_a,
    )
    _add(
        trace_a,
        2,
        "llm.openai.gpt-4o-mini",
        {
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.prompt": "What is your refund policy? (system: be terse)",
            "gen_ai.response.text": "Refunds within 14 days.",
            "gen_ai.usage.input_tokens": 80,
            "gen_ai.usage.output_tokens": 40,
        },
        2400,
        root_a,
    )

    # Trace B — same three plus an extra tool span at the end. LLM is
    # 600ms slower with a longer response (verbose system prompt).
    root_b = _add(
        trace_b,
        0,
        "agent.support",
        {
            "agent.name": "support",
            "fastaiagent.cost.total_usd": 0.04,
            "gen_ai.usage.input_tokens": 95,
            "gen_ai.usage.output_tokens": 175,
        },
        100,
        None,
    )
    _add(
        trace_b,
        1,
        "retrieval.support-kb",
        {"kb.name": "support", "kb.results": 3},
        85,
        root_b,
    )
    _add(
        trace_b,
        2,
        "llm.openai.gpt-4o-mini",
        {
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.prompt": "What is your refund policy? (system: be verbose)",
            "gen_ai.response.text": (
                "Refunds are processed within 14 business days. To start a "
                "return, click 'Request refund' in your order page; you'll "
                "get an email confirmation once the refund is queued."
            ),
            "gen_ai.usage.input_tokens": 95,
            "gen_ai.usage.output_tokens": 175,
        },
        3000,
        root_b,
    )
    _add(
        trace_b,
        3,
        "tool.format_response",
        {"tool.name": "format_response", "tool.input": "raw", "tool.output": "formatted"},
        45,
        root_b,
    )

    return trace_a, trace_b


# --- Datasets — JSONL on disk (matches what the editor reads).
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63fcffff3f0300055e02fea33cc7ab0000000049454e44ae426082"
)


def _seed_datasets(db_path: Path) -> None:
    """Drop two datasets under ``<db_dir>/datasets/`` so the editor's
    list is populated on first load."""
    base = db_path.parent / "datasets"
    base.mkdir(parents=True, exist_ok=True)
    images = base / "images" / "vision-smoke"
    images.mkdir(parents=True, exist_ok=True)

    text = base / "echo-strict.jsonl"
    with text.open("w", encoding="utf-8") as f:
        for word in ("ready", "yes", "no", "ok", "stop"):
            f.write(
                json.dumps(
                    {
                        "input": f"Reply with exactly the word '{word}'.",
                        "expected_output": word,
                        "tags": ["exact_match"],
                        "metadata": {},
                    }
                )
                + "\n"
            )

    img1 = images / "tile-a.png"
    img1.write_bytes(PNG_BYTES)
    vision = base / "vision-smoke.jsonl"
    with vision.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "input": [
                        {"type": "text", "text": "What is shown?"},
                        {"type": "image", "path": "images/vision-smoke/tile-a.png"},
                    ],
                    "expected_output": "blank",
                    "tags": ["vision"],
                    "metadata": {"source": "demo"},
                }
            )
            + "\n"
        )


def _seed_filter_traces(db: SQLiteHelper, project_id: str) -> None:
    """20 assorted traces so the filter bar has room to filter."""
    topics = [
        ("refund", "What is your refund policy?", "Refunds processed within 14 days."),
        ("refund", "How do I request a refund?", "Click 'Request refund' in your order page."),
        ("shipping", "When does shipping arrive?", "Standard: 5-7 business days."),
        ("shipping", "Express shipping cost?", "Express adds $9.99."),
        ("billing", "How do I update my card?", "Account → Billing → Update payment method."),
        ("billing", "Why was I double-charged?", "Pending auth — clears in 48h."),
    ]
    base = datetime.now(tz=timezone.utc) - timedelta(hours=8)
    for i in range(20):
        topic, prompt, response = topics[i % len(topics)]
        agent = "support" if i % 2 == 0 else "billing-agent"
        cost = round(0.001 + (i * 0.011), 4)
        duration_ms = 200 + (i * 137)
        start = base + timedelta(minutes=i * 7)
        end = start + timedelta(milliseconds=duration_ms)
        trace_id = f"trace-filter-{i:02d}"
        root_id = f"{trace_id}-root"
        db.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events, project_id)
               VALUES (?, ?, NULL, ?, ?, ?, ?, ?, '[]', ?)""",
            (
                root_id,
                trace_id,
                f"agent.{agent}",
                _iso(start),
                _iso(end),
                "OK" if i % 7 != 0 else "ERROR",
                json.dumps(
                    {
                        "agent.name": agent,
                        "fastaiagent.cost.total_usd": cost,
                        "gen_ai.usage.input_tokens": 30 + i,
                        "gen_ai.usage.output_tokens": 60 + (i * 3),
                        "fastaiagent.thread.id": f"thread-{topic}",
                    }
                ),
                project_id,
            ),
        )
        db.execute(
            """INSERT OR REPLACE INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events, project_id)
               VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]', ?)""",
            (
                f"{trace_id}-llm",
                trace_id,
                root_id,
                "llm.openai.gpt-4o-mini",
                _iso(start),
                _iso(end),
                json.dumps(
                    {
                        "gen_ai.request.model": "gpt-4o-mini",
                        "gen_ai.prompt": prompt,
                        "gen_ai.response.text": response,
                    }
                ),
                project_id,
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", help="path to local.db")
    parser.add_argument("--project-id", default="sprint3-demo")
    args = parser.parse_args()

    db_path = Path(args.db)
    db = init_local_db(db_path)
    try:
        a, b = _seed_compare_pair(db, args.project_id)
        _seed_filter_traces(db, args.project_id)
    finally:
        db.close()
    _seed_datasets(db_path)

    print(f"compare_a={a}")
    print(f"compare_b={b}")
    print("datasets seeded: echo-strict, vision-smoke")
    print("filter traces seeded: 20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
