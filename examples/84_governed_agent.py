"""Example 84: a connected agent honors a managed approval policy (Task C).

A platform admin configures an **approval policy** (a tool-name pattern). When a
connected agent is about to call a matching tool, the SDK asks the platform
(``POST /policy/decide``); on ``require_approval`` it registers a pending run and
**pauses** (a real checkpoint). **Your application is the approver**: ``arun()``
returns the pause, you ask your user, and you resume with their answer. The
plane records the pause and the outcome; it never decides one (since 1.74.0).

* ``connect()`` caches the policy (``GET /policy``).
* Enroll the agent by setting ``agent_id`` to its **platform agent UUID** — that's
  what ``/policy/decide`` matches on (and the plane validates it).
* Give the agent a ``checkpointer`` so it can pause/resume.
* ``arun()`` returns ``status="paused"``; ``pending_interrupt["context"]`` holds the
  ``tool`` and its ``tool_input``. Resume with
  ``aresume(run_id, resume_value=Resume(approved=..., metadata={"resolver": ...}))``.
  A rejection never runs the tool — the model is told it was refused.

This example plays the application's user twice: it **rejects** a $500 transfer to
Bob, then **approves** a $40 transfer to Eve.

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
    === Transfer $500 to Bob. ===
      paused: transfer_funds {'amount': 500, 'to': 'Bob'}
      the user says: no
      resumed: completed | output: "I'm unable to process the transfer to Bob at this time due …"
      transfer_funds ran: False
    === Transfer $40 to Eve. ===
      paused: transfer_funds {'amount': 40, 'to': 'Eve'}
      the user says: yes
      resumed: completed | output: '$40 has been successfully transferred to Eve.'
      transfer_funds ran: True
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import fastaiagent as fa
from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.chain.interrupt import Resume
from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
from fastaiagent.client import _connection

RAN: list[dict[str, Any]] = []


def transfer_funds(amount: int, to: str) -> str:
    # A "high-stakes" tool — running in your boundary with your own creds.
    RAN.append({"amount": amount, "to": to})
    return f"Transferred ${amount} to {to}."


def ask_the_user(tool: str, args: dict[str, Any]) -> bool:
    """Your approval surface — a UI prompt, a chat reply, a ticket. Scripted here:
    anything over $100 is refused."""
    return float(args.get("amount", 0)) <= 100


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
            f"    POST {target}/api/v1/approval-policies?domain_id=<domain uuid>\n"
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

    async def handle(request: str) -> None:
        print(f"=== {request} ===")
        run_id = f"ex-84-{uuid.uuid4().hex[:8]}"
        before = len(RAN)
        res = await agent.arun(request, execution_id=run_id)
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
        if res.status != "paused" or res.pending_interrupt is None:
            print(f"  NOT paused (status={res.status!r}) — the policy did not match this call.")
            print("  Check the policy's agent_id, tool_pattern and is_active, then re-run.")
            return
        ctx = res.pending_interrupt["context"]
        print("  paused:", ctx["tool"], ctx["tool_input"])
        approved = ask_the_user(ctx["tool"], ctx["tool_input"])
        print("  the user says:", "yes" if approved else "no")
        # The resolver is recorded on the plane's ledger as who decided.
        final = await agent.aresume(
            run_id,
            resume_value=Resume(approved=approved, metadata={"resolver": "example-84-user"}),
        )
        print("  resumed:", final.status, "| output:", repr(final.output))
        print("  transfer_funds ran:", len(RAN) > before)

    async def run() -> None:
        await handle("Transfer $500 to Bob.")
        await handle("Transfer $40 to Eve.")

    asyncio.run(run())
    fa.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
