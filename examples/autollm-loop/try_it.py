"""Ask v1 and the live version the same ticket — side by side.

Run it after the loop (``./run_all.sh``). With no arguments it sends six tickets
that are NOT among the 120 the loop learned from, each with the answer the
house rules give, so you can see whether what AutoLLM wrote generalises. Pass
your own ticket to try anything:

    python try_it.py
    python try_it.py "Enterprise customer: our API has returned 503s for an hour."
"""

from __future__ import annotations

import asyncio
import json
import sys

from triage import current_version, load_agent

# Fresh tickets — none is in data/tickets.jsonl — and what the house rules say.
FRESH = [
    ("We were charged EUR 830 for a plan we downgraded from last month.", "billing P1"),
    ("You charged us EUR 15 twice for the same add-on.", "billing P3"),
    (
        "Enterprise contract here: every call to our production API fails with a 503.",
        "technical P1",
    ),
    ("Our Team workspace can't reach the API at all since 10:00.", "technical P2"),
    ("My hardware security key is now 6 business days late.", "shipping P2"),
    ("Please send me every piece of personal data you hold on me (GDPR).", "account P2"),
]


def _short(reply: str) -> str:
    try:
        r = json.loads(reply)
        return f"{r['queue']} {r['priority']}"
    except (ValueError, KeyError, TypeError):
        return reply.strip()[:24] or "(no reply)"


async def main(tickets: list[tuple[str, str]]) -> None:
    live = current_version()
    if live.version == 1:
        print("'production' still points at v1 — run ./run_all.sh first.")
        return
    v1, now = load_agent(version=1), load_agent(alias="production")
    print(f"{'ticket':<62} {'v1':<14} {f'v{live.version} (live)':<14} house rule")
    for text, rule in tickets:
        a, b = await asyncio.gather(v1.arun(text), now.arun(text))
        old, new = _short(a.output or ""), _short(b.output or "")
        mark = "✓" if rule and new == rule else ("✗" if rule else "")
        print(f"{text[:60]:<62} {old:<14} {new:<14} {rule} {mark}")


if __name__ == "__main__":
    asked = [(" ".join(sys.argv[1:]), "")] if len(sys.argv) > 1 else FRESH
    asyncio.run(main(asked))
