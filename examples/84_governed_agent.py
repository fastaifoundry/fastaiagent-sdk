"""Example 84: a connected agent honors a managed approval policy (Task C).

A platform admin configures an **approval policy** (a tool-name pattern). When a
connected agent is about to call a matching tool, the SDK asks the platform
(``POST /policy/decide``); on ``require_approval`` it registers a pending run and
**pauses** (a real checkpoint). A human approves on the console; the agent
**resumes** and finishes.

* ``connect()`` caches the policy (``GET /policy``).
* Enroll the agent by setting ``agent_id`` to its **platform agent UUID** — that's
  what ``/policy/decide`` matches on (and the plane validates it).
* Give the agent a ``checkpointer`` so it can pause/resume.
* ``arun()`` **blocks by default** until the console decides, then resumes. Pass
  ``wait_for_approval=False`` to get the paused result and drive resume yourself.

Usage:
    export OPENAI_API_KEY=sk-...
    export FASTAIAGENT_API_KEY=fa_k_...      # scopes: policy:read policy:decide run:write run:read
    export FASTAIAGENT_TARGET=http://localhost:20001
    export FASTAIAGENT_AGENT_ID=<platform agent uuid>
    python examples/84_governed_agent.py

Expected output (snapshot — real run against a local plane on :20001):
    connected. cached approval_policies: 1
    === arun(wait_for_approval=False) ===
      paused for approval: policy_approval_required
      pending run status: pending  ->  (console approves) ->  approved
      resumed: completed | output: 'I have successfully transferred $500 to Bob.'
"""

from __future__ import annotations

import asyncio
import os

import fastaiagent as fa
from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.chain.interrupt import Resume
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.client import _connection


def transfer_funds(amount: int, to: str) -> str:
    # A "high-stakes" tool — running in your boundary with your own creds.
    return f"Transferred ${amount} to {to}."


def main() -> int:
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:20001")
    agent_id = os.environ.get("FASTAIAGENT_AGENT_ID", "")
    if not (api_key and os.environ.get("OPENAI_API_KEY") and agent_id):
        print("Skipping: set OPENAI_API_KEY, FASTAIAGENT_API_KEY and FASTAIAGENT_AGENT_ID")
        return 1

    fa.connect(api_key=api_key, target=target)
    n = len((_connection.policy_cache or {}).get("approval_policies", []))
    print(f"connected. cached approval_policies: {n}")

    agent = Agent(
        name="banker",
        agent_id=agent_id,  # enroll in managed governance
        system_prompt=(
            "You are a banking assistant. To move money, call transfer_funds(amount, to). "
            "After the tool returns, confirm to the user in one sentence."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
        checkpointer=SQLiteCheckpointer("governed_agent.db"),
    )

    async def run() -> None:
        # Non-blocking: see the pause, then resume after the console approves.
        print("=== arun(wait_for_approval=False) ===")
        res = await agent.arun(
            "Transfer $500 to Bob.", wait_for_approval=False, execution_id="ex-84"
        )
        print("  paused for approval:", (res.pending_interrupt or {}).get("reason"))
        print("  -> approve this run in the console (POST /api/v1/pending-runs/{id}/approve)")
        # Once approved, resume. (The blocking default — plain agent.arun(...) —
        # waits for the console and resumes for you.)
        final = await agent.aresume("ex-84", resume_value=Resume(approved=True))
        print("  resumed:", final.status, "| output:", repr(final.output))

    asyncio.run(run())
    fa.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
