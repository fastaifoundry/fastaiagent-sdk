"""Example 83: a registered-runner job's trace flows SDK -> Enterprise (Task A).

Mirrors what ``fastaiagent runner`` now does natively: connect to the platform
(which wires the trace exporter), execute a ``live_playground`` command through the
runner's own ``execute_command``, flush, and confirm the trace landed on the
platform — linked by the ``trace_id`` the runner reports.

The platform routes the trace by your **API key** (one runner == one tenant); the
local trace store is just a buffer. The self-verify step needs the ``trace:read``
scope on the key (it degrades gracefully without it — check the console instead).

Usage:
    export OPENAI_API_KEY=sk-...
    export FASTAIAGENT_API_KEY=fa_k_...          # runner:register (+ trace:read to self-verify)
    export FASTAIAGENT_TARGET=http://localhost:20001
    python examples/83_runner_trace_to_platform.py

Expected output (snapshot — real run against a local plane on :20001):
    ============================================================
      Runner trace -> platform (Task A)
    ============================================================
      Connected: True  domain=8ccd14b5-…  project=ab4d5161-…
      Executing live_playground 'demo-1' (real gpt-4o-mini)…
      status=completed  output='OK'  trace_id=19c1f4083a0409d075d8d18d7a7c3871
      Flushed exporter -> POST /public/v1/traces/ingest

      Verifying on the platform (GET /public/v1/traces/{id})…
      ✓ trace 19c1f408…  source=sdk  status=completed  spans=[agent.demo, llm.openai.gpt-4o-mini]
    ============================================================
      DONE — open the trace in the Enterprise console
    ============================================================
"""

from __future__ import annotations

import asyncio
import os

import httpx

import fastaiagent as fa
from fastaiagent import Agent, LLMClient
from fastaiagent.client import _connection
from fastaiagent.runner.execute import execute_command


def main() -> int:
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:20001")
    if not api_key:
        print("Skipping: FASTAIAGENT_API_KEY not set")
        return 1
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        return 1

    print("=" * 60)
    print("  Runner trace -> platform (Task A)")
    print("=" * 60)

    # Exactly what `fastaiagent runner` does on startup: connect with the key,
    # which wires the PlatformSpanExporter so the jobs it runs push their traces.
    fa.connect(api_key=api_key, target=target)
    print(
        f"  Connected: {fa.is_connected}  domain={_connection.domain_id}"
        f"  project={_connection.project_id}"
    )

    agent = Agent(
        name="demo",
        system_prompt="Reply with exactly the word OK and nothing else.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )
    cmd = {
        "command_id": "demo-1",
        "type": "live_playground",
        "tenant": _connection.domain_id,
        "payload": {"agent": agent.to_dict(), "input": "say ok"},
    }
    print("  Executing live_playground 'demo-1' (real gpt-4o-mini)…")

    async def _run():
        # Own asyncio task (the daemon's per-job invariant), then flush the
        # exporter the way the daemon's _run_command does.
        res = await asyncio.create_task(execute_command(cmd))
        proc = _connection._platform_processor
        if proc is not None:
            await asyncio.to_thread(proc.force_flush, 10000)
        return res

    res = asyncio.run(_run())
    print(f"  status={res.status}  output={res.result!r}  trace_id={res.trace_id}")
    print("  Flushed exporter -> POST /public/v1/traces/ingest")

    # Confirm it landed on the platform (needs the trace:read scope).
    print("\n  Verifying on the platform (GET /public/v1/traces/{id})…")
    resp = httpx.get(f"{target}/public/v1/traces/{res.trace_id}", headers={"X-API-Key": api_key})
    if resp.status_code == 200:
        d = resp.json()
        spans = ", ".join(s["name"] for s in d.get("spans", []))
        print(
            f"  ✓ trace {res.trace_id}  source={d['source']}"
            f"  status={d['status']}  spans=[{spans}]"
        )
    else:
        print(f"  (verify skipped: HTTP {resp.status_code} — key may lack the trace:read scope)")

    fa.disconnect()
    print("=" * 60)
    print("  DONE — open the trace in the Enterprise console")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
