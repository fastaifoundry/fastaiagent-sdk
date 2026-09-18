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

The precondition this example cannot run without: a matching **approval
policy** on the domain. With none, ``/policy/decide`` returns "allow", the
agent never pauses, and there is nothing to resume. Create one (domain-admin
JWT, not an API key)::

    POST /api/v1/approval-policies?domain_id=<domain uuid>
    {"name": "example-84", "tool_pattern": "transfer_funds",
     "agent_id": "<the same agent uuid>", "condition_type": "always"}

Expected output (snapshot — real run against a local plane on :20001):
    connected. cached approval_policies: 1
    === arun(wait_for_approval=False) ===
      paused for approval: policy_approval_required
      -> approve this run in the console (POST /api/v1/pending-runs/{id}/approve)
      resumed: completed | output: 'The $500 has been successfully transferred to Bob.'
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
    policies = (_connection.policy_cache or {}).get("approval_policies", [])
    print(f"connected. cached approval_policies: {len(policies)}")

    # Check the precondition before spending an LLM call on it. With no
    # approval policy on the domain, ``/policy/decide`` answers "allow": the
    # agent runs straight through, never pauses, and the resume below has
    # nothing to claim.
    if not policies:
        print(
            "Skipping: this domain has no approval policy, so nothing will pause.\n"
            "  Create one with a domain-admin JWT (an API key cannot):\n"
            f'    POST {target}/api/v1/approval-policies?domain_id=<domain uuid>\n'
            '    {"name": "example-84", "tool_pattern": "transfer_funds",\n'
            f'     "agent_id": "{agent_id}", "condition_type": "always"}}'
        )
        fa.disconnect()
        return 1

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
        # Assert the pause actually happened before trying to resume it. A
        # policy can exist and still not match — wrong agent_id, a
        # ``tool_pattern`` that does not cover ``transfer_funds``, an inactive
        # policy, or a ``condition_type`` whose condition did not fire.
        #
        # This check is not decoration. Before 1.65.0, ``aresume`` on a run
        # that never paused returned a plausible ``completed`` result, so this
        # example printed a clean success line with no approval having
        # happened anywhere — a governance demo that looked like it worked
        # while the gate was inert. 1.65.0 made that raise ``AlreadyResumed``
        # instead; what it raises on is *this* example's own missing
        # precondition, so check it here rather than catching the exception.
        if res.status != "paused":
            print(f"  NOT paused (status={res.status!r}) — the policy did not match this call.")
            print("  Check the policy's agent_id, tool_pattern and is_active, then re-run.")
            return
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
