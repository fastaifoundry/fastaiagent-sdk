"""Example 54 — Richer Trace Filtering demo (Sprint 3).

Seeds 20 spans directly into ``./.fastaiagent/local.db`` so the Local
UI's Traces page has data to filter against. No LLM call — the example
is filesystem + DB seeding only, so it's safe to re-run and free.

After running:

    fastaiagent ui --no-auth
    open http://127.0.0.1:7842/traces

Then try these pre-filtered URLs to see each Sprint 3 filter in
action — the script prints them all.

Prereqs:
    pip install 'fastaiagent[ui]'

Run:
    python examples/54_trace_filters.py
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone

from fastaiagent.ui.db import init_local_db


TOPICS = [
    ("refund", "What is your refund policy?", "Refunds processed within 14 days."),
    ("refund", "How do I request a refund?", "Click 'Request refund' in your order page."),
    ("refund", "Refund timeline?", "Two business weeks after approval."),
    ("shipping", "When does shipping arrive?", "Standard shipping: 5-7 business days."),
    ("shipping", "Express shipping cost?", "Express adds $9.99 to your order total."),
    ("shipping", "International shipping?", "We ship to 30+ countries; rates at checkout."),
    ("billing", "How do I update my card?", "Account → Billing → Update payment method."),
    ("billing", "Why was I double-charged?", "Likely a pending auth — should clear in 48h."),
    ("returns", "Return policy?", "Returns accepted within 30 days, original packaging."),
    ("returns", "Free returns?", "Yes for orders over $50."),
]


def main() -> int:
    db = init_local_db()
    try:
        base_time = datetime.now(tz=timezone.utc) - timedelta(hours=8)
        # Two passes through TOPICS = 20 traces total. Vary cost across
        # a spread so min_cost / max_cost filters have something to bite
        # on. Vary duration so duration filters work too. Alternate
        # agent name so the agent filter works.
        for i in range(20):
            topic, prompt_text, response_text = TOPICS[i % len(TOPICS)]
            agent_name = "support" if i % 2 == 0 else "billing-agent"
            cost_usd = round(0.001 + (i * 0.011), 4)  # 0.001 .. 0.210
            duration_ms = 200 + (i * 137)  # 200 .. 2803 ms
            input_tokens = 30 + i
            output_tokens = 60 + (i * 3)
            start = base_time + timedelta(minutes=i * 7)
            end = start + timedelta(milliseconds=duration_ms)

            trace_id = uuid.uuid4().hex
            root_id = uuid.uuid4().hex
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events, project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    root_id,
                    trace_id,
                    None,
                    f"agent.{agent_name}",
                    start.isoformat(),
                    end.isoformat(),
                    "OK" if i % 7 != 0 else "ERROR",
                    json.dumps(
                        {
                            "agent.name": agent_name,
                            "fastaiagent.cost.total_usd": cost_usd,
                            "gen_ai.usage.input_tokens": input_tokens,
                            "gen_ai.usage.output_tokens": output_tokens,
                            "fastaiagent.thread.id": f"t-{topic}",
                        }
                    ),
                    "[]",
                    "",
                ),
            )

            child_id = uuid.uuid4().hex
            db.execute(
                """INSERT INTO spans
                   (span_id, trace_id, parent_span_id, name, start_time, end_time,
                    status, attributes, events, project_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    child_id,
                    trace_id,
                    root_id,
                    "llm.openai.gpt-4o-mini",
                    start.isoformat(),
                    end.isoformat(),
                    "OK",
                    json.dumps(
                        {
                            "gen_ai.request.model": "gpt-4o-mini",
                            "gen_ai.prompt": prompt_text,
                            "gen_ai.response.text": response_text,
                            "gen_ai.usage.input_tokens": input_tokens,
                            "gen_ai.usage.output_tokens": output_tokens,
                        }
                    ),
                    "[]",
                    "",
                ),
            )
    finally:
        db.close()

    print("Seeded 20 traces (40 spans) into ./.fastaiagent/local.db")
    print()
    print("Open the UI:")
    print("  fastaiagent ui --no-auth")
    print()
    print("Try these URLs to see each Sprint 3 filter:")
    print()
    print("  Full-text search (FTS5)")
    print("    http://127.0.0.1:7842/traces?q=refund")
    print("    http://127.0.0.1:7842/traces?q=shipping%20cost")
    print()
    print("  Agent + status")
    print("    http://127.0.0.1:7842/traces?agent=support&status=OK")
    print()
    print("  Cost range (under $0.05)")
    print("    http://127.0.0.1:7842/traces?max_cost=0.05")
    print()
    print("  Duration range (slower than 1.5s)")
    print("    http://127.0.0.1:7842/traces?min_duration_ms=1500")
    print()
    print("  Combined (errors with high cost)")
    print("    http://127.0.0.1:7842/traces?status=ERROR&min_cost=0.05")
    print()
    print("Save any of these as a preset via the bookmark icon in the bar,")
    print("then use the dropdown to one-click reapply later.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
