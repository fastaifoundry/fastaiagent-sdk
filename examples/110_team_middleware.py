"""Example 110: one middleware for a whole team (1.91.0).

A support desk is a Swarm — ``triage`` hands billing questions to ``billing``.
Two team-wide rules, written once on the Swarm instead of copied onto each agent:

    Audit                     log every hop and every tool call
    ToolBudget(max_calls=6)   one budget for the whole conversation, every hop counted

``Swarm(middleware=...)`` runs on every hop, outside each agent's own
middleware; ``Supervisor(middleware=...)`` and ``Chain(middleware=...)`` work the
same way (the supervisor's own loop; every agent and tool node). The run's root
span names the team's middleware in ``fastaiagent.middleware``.

Usage:
    python examples/110_team_middleware.py --offline   # no keys
    zsh -lc 'python examples/110_team_middleware.py'    # OPENAI_API_KEY
"""

from __future__ import annotations

import os
import sys

from fastaiagent import Agent, AgentMiddleware, FunctionTool, LLMClient, Swarm, ToolBudget


def get_invoice(invoice_id: str) -> str:
    """Look up an invoice: its amount and due date."""
    return {"INV-9": "amount $40.00, due 2026-11-01"}.get(invoice_id, "no such invoice")


class Audit(AgentMiddleware):
    name = "audit"

    async def before_run(self, ctx, input):
        print(f"  [audit] hop → {ctx.agent_name}")
        return input

    async def wrap_tool(self, ctx, tool, args, call_next):
        print(f"  [audit] {ctx.agent_name} calls {tool.name}({args})")
        return await call_next(tool, args)


def _llms(offline: bool) -> tuple[LLMClient, LLMClient]:
    if not offline:
        llm = LLMClient(provider="openai", model="gpt-5-mini")
        return llm, llm
    from fastaiagent.testing import FunctionModel

    def triage(messages):
        return "", [{"name": "handoff_to_billing", "arguments": {"reason": "an invoice question"}}]

    def billing(messages):
        if any(m.role.value == "tool" for m in messages):
            return "Invoice INV-9 is $40.00, due 2026-11-01."
        return "", [{"name": "get_invoice", "arguments": {"invoice_id": "INV-9"}}]

    return FunctionModel(triage), FunctionModel(billing)


def main(offline: bool) -> None:
    triage_llm, billing_llm = _llms(offline)
    triage = Agent(
        name="triage",
        system_prompt="You route requests. For anything about invoices, hand off to billing.",
        llm=triage_llm,
    )
    billing = Agent(
        name="billing",
        system_prompt="You answer invoice questions with get_invoice. Be brief.",
        llm=billing_llm,
        tools=[FunctionTool(name="get_invoice", fn=get_invoice)],
    )
    desk = Swarm(
        name="support-desk",
        agents=[triage, billing],
        entrypoint="triage",
        handoffs={"triage": ["billing"], "billing": []},
        middleware=[Audit(), ToolBudget(max_calls=6)],
    )
    result = desk.run("How much is invoice INV-9?")
    print(f"  answer: {result.output}")


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    if not offline and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — running --offline")
        offline = True
    main(offline)
