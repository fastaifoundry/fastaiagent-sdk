"""Example 117: a Supervisor and a Swarm end to end — what 1.99.0 changed.

Demonstrates:
  1. Knowledge bases that tell the agent what they hold (``description=``)
  2. ``Supervisor(memory=...)``: the team holds a conversation across turns
  3. The Supervisor's result: the whole team's tokens and cost, and every
     worker's tool calls after the ``delegate_to_<role>`` call that ran them
  4. A Swarm whose handoffs carry what earlier agents found, so the next agent
     does not search again
  5. A streamed Swarm run is one trace
  6. Connected (optional): set FASTAIAGENT_API_KEY and each member registers as
     itself, is gated by the plane's approval policies on every run, and its
     spans carry its ``agent.id``

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/117_teams_end_to_end.py
    # optional: export FASTAIAGENT_API_KEY=... FASTAIAGENT_TARGET=http://localhost:20001
"""

from __future__ import annotations

import asyncio
import os

import fastaiagent as fa
from fastaiagent import Agent, LLMClient, Memory, Supervisor, Swarm, Worker, tool
from fastaiagent.kb import LocalKB
from fastaiagent.llm.stream import HandoffEvent

MODEL = os.environ.get("EXAMPLE_MODEL", "gpt-5-mini")


def knowledge_bases() -> tuple[LocalKB, LocalKB]:
    """Two KBs. ``description=`` is what lets an agent holding both choose."""
    policy = LocalKB(
        name="refund-policy",
        persist=False,
        search_type="keyword",  # no embedding calls in this example
        description="refund windows, cancellation fees and refund codes",
    )
    policy.add(
        "Annual plans cancelled within 30 days get a full refund minus a processing "
        "fee of $17.40, filed under refund code RFX-9."
    )
    tech = LocalKB(
        name="router-help",
        persist=False,
        search_type="keyword",
        description="router error codes and their fixes",
    )
    tech.add(
        "Router error E-311 is a firmware mismatch: hold reset for 15 seconds, then "
        "re-pair in the app. If that fails, escalate under code TCH-77."
    )
    return policy, tech


@tool(name="lookup_order")
def lookup_order(order_id: str) -> str:
    """Look up a customer's order."""
    return f"{order_id}: Annual Unlimited, bought 12 days ago, $480"


def supervisor_team(policy: LocalKB) -> Supervisor:
    llm = LLMClient(provider="openai", model=MODEL)
    orders = Agent(
        name="orders",
        system_prompt="Look orders up with your tool and report what you find.",
        llm=llm,
        tools=[lookup_order],
    )
    refunds = Agent(
        name="refunds",
        system_prompt="Answer refund questions from your knowledge base. Quote amounts and codes.",
        llm=llm,
        tools=[policy.as_tool()],
    )
    return Supervisor(
        name="support-lead",
        llm=llm,
        workers=[
            Worker(orders, role="orders", description="looks up orders"),
            Worker(refunds, role="refunds", description="refund policy and fees"),
        ],
        memory=Memory(),  # the lead's own conversation (1.99.0)
    )


def swarm_team(policy: LocalKB, tech: LocalKB) -> Swarm:
    llm = LLMClient(provider="openai", model=MODEL)
    triage = Agent(
        name="triage",
        system_prompt="Hand technical problems to tech and billing questions to billing.",
        llm=llm,
    )
    tech_agent = Agent(
        name="tech",
        system_prompt=(
            "Solve router problems from your knowledge base. If a billing question is "
            "still open, hand off to billing."
        ),
        llm=llm,
        tools=[tech.as_tool()],
    )
    billing = Agent(
        name="billing",
        system_prompt=(
            "Answer billing questions from your knowledge base. If a technical problem "
            "is still unsolved, hand off to tech; otherwise give the customer one answer "
            "covering everything the team found."
        ),
        llm=llm,
        tools=[policy.as_tool()],
    )
    return Swarm(
        name="support-swarm",
        agents=[triage, tech_agent, billing],
        entrypoint="triage",
        handoffs={"triage": ["tech", "billing"], "tech": ["billing"], "billing": ["tech"]},
        max_handoffs=4,
    )


def main() -> None:
    if os.environ.get("FASTAIAGENT_API_KEY"):
        fa.connect(
            api_key=os.environ["FASTAIAGENT_API_KEY"],
            target=os.environ.get("FASTAIAGENT_TARGET"),  # your program address
        )
        print("Connected: each member registers as itself on its first run.\n")

    policy, tech = knowledge_bases()
    print("KB tools the agents see:")
    for kb in (policy, tech):
        t = kb.as_tool()
        print(f"  {t.name}: {t.description}")

    # ── Supervisor: a conversation, whole-team totals, worker tool calls ─────
    lead = supervisor_team(policy)
    first = lead.run("Order A100: I want to cancel. What fee do I pay?")
    print("\nSupervisor, turn 1:", first.output)
    print(f"  tokens (whole team): {first.tokens_used}  cost: ${first.cost:.5f}")
    print("  tool calls:")
    for call in first.tool_calls:
        print(f"    {call['agent']:>13} -> {call['tool_name']}")
    second = lead.run("Which order was that, and what was the refund code?")
    print("Supervisor, turn 2 (from memory):", second.output)

    # ── Swarm: findings carried across handoffs ─────────────────────────────
    swarm = swarm_team(policy, tech)
    question = (
        "My router shows E-311 since the update. Also, I bought the annual plan 12 days "
        "ago: what fee do I pay if I cancel, and under which refund code?"
    )
    result = swarm.run(question)
    hops = [c["agent"] for c in result.tool_calls if c["tool_name"].startswith("handoff_to_")]
    print("\nSwarm path:", " -> ".join([*hops, "(answer)"]))
    print("Swarm answer:", result.output)

    # ── a streamed swarm run is one trace ───────────────────────────────────
    async def stream() -> None:
        async for event in swarm.astream(question):
            if isinstance(event, HandoffEvent):
                print(f"  handoff {event.from_agent} -> {event.to_agent}")

    print("\nStreaming the swarm:")
    asyncio.run(stream())
    print("  (one swarm.support-swarm trace — open it with `fastaiagent ui`)")

    if os.environ.get("FASTAIAGENT_API_KEY"):
        fa.disconnect()


if __name__ == "__main__":
    main()
