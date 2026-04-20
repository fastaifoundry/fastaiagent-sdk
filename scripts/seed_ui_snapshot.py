"""Seed a local.db with representative data for docs screenshots.

Used by ``scripts/capture-ui-screenshots.sh`` — never run in production.
Writes spans (one healthy trace + one failing trace), prompt versions, an
eval run with cases, and guardrail events so every UI surface has content.
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


def seed(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_local_db(db_path).close()

    now = datetime.now(tz=timezone.utc)
    with SQLiteHelper(db_path) as db:
        _seed_traces(db, now)
        _seed_analytics_spread(db, now)
        _seed_prompts(db, now)
        _seed_evals(db, now)
        _seed_guardrails(db, now)


def _seed_analytics_spread(db: SQLiteHelper, now: datetime) -> None:
    """Fill the last 7 days with varied traces so /analytics has a real chart."""
    import random

    rng = random.Random(42)
    agents = [
        ("support-bot", 1200, 2800, 0.004),
        ("recommender", 400, 1200, 0.0015),
    ]
    for hour_offset in range(7 * 24):
        for _ in range(2):
            agent, low, high, base_cost = rng.choice(agents)
            start = now - timedelta(hours=hour_offset, minutes=rng.randint(0, 59))
            dur = rng.randint(low, high) + rng.randint(-200, 200)
            end = start + timedelta(milliseconds=max(50, dur))
            errored = rng.random() < 0.08
            tokens_in = rng.randint(80, 400)
            tokens_out = rng.randint(40, 200)
            thread_id = (
                f"session-{rng.randint(1, 30)}" if rng.random() < 0.3 else None
            )
            attrs = {
                "agent.name": agent,
                "fastaiagent.cost.total_usd": round(base_cost * rng.uniform(0.7, 1.6), 6),
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": tokens_in,
                "gen_ai.usage.output_tokens": tokens_out,
            }
            if thread_id:
                attrs["fastaiagent.thread.id"] = thread_id
            span_id = f"s-a-{hour_offset}-{rng.randint(0, 1_000_000):x}"
            trace_id = f"auto-{hour_offset}-{rng.randint(0, 1_000_000):x}"
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]')""",
                (
                    span_id,
                    trace_id,
                    None,
                    f"agent.{agent}",
                    start.isoformat(),
                    end.isoformat(),
                    "ERROR" if errored else "OK",
                    json.dumps(attrs),
                ),
            )
    # Demo session that the /threads screenshot targets.
    for i in range(4):
        start = now - timedelta(hours=2, minutes=30 - i * 6)
        end = start + timedelta(milliseconds=800 + i * 120)
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]')""",
            (
                f"s-demo-{i}",
                f"demo-thread-trace-{i}",
                None,
                "agent.support-bot",
                start.isoformat(),
                end.isoformat(),
                "OK",
                json.dumps(
                    {
                        "agent.name": "support-bot",
                        "fastaiagent.thread.id": "session-demo",
                        "fastaiagent.cost.total_usd": 0.002 + i * 0.001,
                        "gen_ai.request.model": "gpt-4o-mini",
                        "gen_ai.usage.input_tokens": 120 + i * 30,
                        "gen_ai.usage.output_tokens": 80 + i * 20,
                    }
                ),
            ),
        )


def _seed_traces(db: SQLiteHelper, now: datetime) -> None:
    # Healthy trace with a small span tree.
    trace_a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa1111"
    root_a = "s_root_aaaaaaaa"
    llm_a = "s_llm_aaaaaaaaa"
    tool_a = "s_tool_aaaaaaaa"

    base = now - timedelta(minutes=3)
    spans = [
        (
            root_a,
            trace_a,
            None,
            "agent.support-bot",
            base,
            base + timedelta(milliseconds=1800),
            "OK",
            {
                "agent.name": "support-bot",
                "fastaiagent.cost.total_usd": 0.0042,
                "gen_ai.usage.input_tokens": 180,
                "gen_ai.usage.output_tokens": 90,
                "fastaiagent.prompt.name": "ui-demo.support",
                "fastaiagent.prompt.version": "1",
                "agent.input": "My order hasn't shipped — can you check?",
                "agent.output": "I can see order #4271 shipped yesterday via FedEx. Tracking: 8421099.",
            },
        ),
        (
            llm_a,
            trace_a,
            root_a,
            "llm.chat",
            base + timedelta(milliseconds=120),
            base + timedelta(milliseconds=1400),
            "OK",
            {
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.request.temperature": 0.2,
                "gen_ai.usage.input_tokens": 180,
                "gen_ai.usage.output_tokens": 90,
                "gen_ai.response.content": "I can see order #4271 shipped yesterday.",
            },
        ),
        (
            tool_a,
            trace_a,
            root_a,
            "tool.lookup_order",
            base + timedelta(milliseconds=1400),
            base + timedelta(milliseconds=1750),
            "OK",
            {
                "tool.input": {"order_id": "4271"},
                "tool.output": {"status": "shipped", "carrier": "FedEx"},
            },
        ),
    ]

    # Failing trace.
    trace_b = "bbbbbbbbbbbbbbbbbbbbbbbbbbbb2222"
    root_b = "s_root_bbbbbbbb"
    llm_b = "s_llm_bbbbbbbbb"
    base_b = now - timedelta(minutes=12)
    spans.extend(
        [
            (
                root_b,
                trace_b,
                None,
                "agent.flaky",
                base_b,
                base_b + timedelta(milliseconds=5300),
                "ERROR",
                {
                    "agent.name": "flaky",
                    "fastaiagent.cost.total_usd": 0.0031,
                    "gen_ai.usage.input_tokens": 220,
                    "gen_ai.usage.output_tokens": 40,
                    "agent.input": "Give me the PII for customer 42",
                    "agent.output": "Request blocked by guardrail no_pii.",
                },
            ),
            (
                llm_b,
                trace_b,
                root_b,
                "llm.chat",
                base_b + timedelta(milliseconds=100),
                base_b + timedelta(milliseconds=4900),
                "ERROR",
                {
                    "gen_ai.request.model": "gpt-4o-mini",
                    "gen_ai.response.content": "SSN 111-22-3333",
                },
            ),
        ]
    )

    for sid, tid, pid, name, start, end, status, attrs in spans:
        db.execute(
            """INSERT INTO spans
               (span_id, trace_id, parent_span_id, name, start_time, end_time,
                status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                sid,
                tid,
                pid,
                name,
                _iso(start),
                _iso(end),
                status,
                json.dumps(attrs),
                "[]",
            ),
        )


def _seed_prompts(db: SQLiteHelper, now: datetime) -> None:
    iso = _iso(now)
    db.execute(
        """INSERT INTO prompts (slug, latest_version, created_at, updated_at)
           VALUES (?, ?, ?, ?)""",
        ("ui-demo.support", "2", iso, iso),
    )
    for version, template, created in (
        (
            "1",
            "You are a concise support assistant. Always verify orders before replying.",
            _iso(now - timedelta(days=3)),
        ),
        (
            "2",
            (
                "You are a concise support assistant. Always verify orders before replying.\n"
                "Cite the tracking number when giving shipping information."
            ),
            iso,
        ),
    ):
        db.execute(
            """INSERT INTO prompt_versions
               (slug, version, template, variables, fragments, metadata,
                created_at, created_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "ui-demo.support",
                version,
                template,
                "[]",
                "[]",
                "{}",
                created,
                "code",
            ),
        )


def _seed_evals(db: SQLiteHelper, now: datetime) -> None:
    run_id = uuid.uuid4().hex
    started = now - timedelta(hours=2)
    db.execute(
        """INSERT INTO eval_runs
           (run_id, run_name, dataset_name, agent_name, agent_version,
            scorers, started_at, finished_at, pass_count, fail_count,
            pass_rate, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            "support-smoke",
            "support-cases.jsonl",
            "support-bot",
            "v12",
            json.dumps(["exact_match"]),
            _iso(started),
            _iso(started + timedelta(seconds=18)),
            2,
            1,
            0.6667,
            "{}",
        ),
    )
    cases = [
        (
            "What is order 4271's status?",
            "shipped",
            "shipped",
            True,
        ),
        (
            "Can you refund my order 4271?",
            "yes, initiating",
            "no, refund window expired",
            False,
        ),
        (
            "Where is my package?",
            "tracking 8421099",
            "tracking 8421099",
            True,
        ),
    ]
    for ordinal, (inp, expected, actual, passed) in enumerate(cases):
        db.execute(
            """INSERT INTO eval_cases
               (case_id, run_id, ordinal, input, expected_output,
                actual_output, trace_id, per_scorer)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                run_id,
                ordinal,
                json.dumps(inp),
                json.dumps(expected),
                json.dumps(actual),
                None,
                json.dumps(
                    {
                        "exact_match": {
                            "passed": passed,
                            "score": 1.0 if passed else 0.0,
                            "reason": None,
                        }
                    }
                ),
            ),
        )


def _seed_guardrails(db: SQLiteHelper, now: datetime) -> None:
    events = [
        (
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaa1111",
            "s_llm_aaaaaaaaa",
            "no_pii",
            "regex",
            "output",
            "passed",
            1.0,
            "no PII detected",
            "support-bot",
            now - timedelta(minutes=3),
        ),
        (
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbb2222",
            "s_llm_bbbbbbbbb",
            "no_pii",
            "regex",
            "output",
            "blocked",
            0.0,
            "SSN pattern matched in response",
            "flaky",
            now - timedelta(minutes=12),
        ),
        (
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbb2222",
            "s_llm_bbbbbbbbb",
            "toxicity",
            "classifier",
            "output",
            "warned",
            0.42,
            "borderline toxicity score",
            "flaky",
            now - timedelta(minutes=12, seconds=20),
        ),
    ]
    for trace_id, span_id, name, g_type, pos, outcome, score, msg, agent, when in events:
        db.execute(
            """INSERT INTO guardrail_events
               (event_id, trace_id, span_id, guardrail_name, guardrail_type,
                position, outcome, score, message, agent_name, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                trace_id,
                span_id,
                name,
                g_type,
                pos,
                outcome,
                score,
                msg,
                agent,
                _iso(when),
                "{}",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path, help="Target local.db path (WILL BE OVERWRITTEN)")
    args = parser.parse_args()
    seed(args.db)
    print(f"seeded {args.db}")
