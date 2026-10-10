"""Step 2 — production traffic: 120 tickets through whatever 'production' points at.

Each ticket is a traced run, and each model call is stamped with the registry
prompt behind it (``fastaiagent.prompt.name`` / ``.version``). Run this again
after step 7 and the same page shows the traffic moving to the new version.

UI: Traces (one per ticket) · Prompts → ticket-triage → lineage.
"""

from __future__ import annotations

import asyncio

from triage import AGENT, current_version, load_agent, tickets


async def _serve() -> None:
    agent = load_agent(alias="production")
    live = current_version()
    rows = tickets()
    gate = asyncio.Semaphore(8)

    async def handle(row: dict) -> tuple[str, str]:
        async with gate:
            result = await agent.arun(row["ticket"], metadata={"ticket_id": row["id"]})
            return row["id"], (result.output or "").strip()

    print(f"serving {len(rows)} tickets with {AGENT!r} on prompt v{live.version}\n")
    for ticket_id, reply in await asyncio.gather(*(handle(r) for r in rows)):
        print(f"  {ticket_id}  {reply}")
    print(f"\n{len(rows)} traced runs, each stamped with prompt v{live.version}.")


if __name__ == "__main__":
    asyncio.run(_serve())
