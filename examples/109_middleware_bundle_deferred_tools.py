"""Example 109: a policy as one middleware, and tools loaded only when needed (1.91.0).

A refunds policy is three things — the tool that acts, the instruction that says
when to use it, and a limit that must hold across the whole conversation, pauses
included. ``Refunds`` below carries all three:

    tools          issue_refund, added to the agent's tools
    instructions   one system message after the system prompt
    ctx.state      the refund count — saved in checkpoints, so it survives a resume

The billing tools are ``deferred=True``: their definitions are not sent until the
model asks for them through the built-in ``load_tools`` tool, so a large toolset
costs no context on turns that do not use it.

Usage:
    python examples/109_middleware_bundle_deferred_tools.py --offline   # no keys
    zsh -lc 'python examples/109_middleware_bundle_deferred_tools.py'    # OPENAI_API_KEY
"""

from __future__ import annotations

import os
import sys
import tempfile

from fastaiagent import (
    Agent,
    AgentMiddleware,
    FunctionTool,
    LLMClient,
    SQLiteCheckpointer,
    StopAgent,
)


def issue_refund(order_id: str) -> str:
    """Issue a refund for an order."""
    return f"refund REF-{order_id} issued"


def get_invoice(invoice_id: str) -> str:
    """Look up an invoice: its amount and due date."""
    return {"INV-9": "amount $40.00, due 2026-11-01"}.get(invoice_id, "no such invoice")


def track_shipment(order_id: str) -> str:
    """Track a shipment by order id."""
    return f"order {order_id}: in transit"


def update_address(order_id: str, address: str) -> str:
    """Change the delivery address of an order."""
    return f"order {order_id} now ships to {address}"


class Refunds(AgentMiddleware):
    name = "refunds"
    tools = [FunctionTool(name="issue_refund", fn=issue_refund)]
    instructions = (
        "Refunds: you may refund an order the customer names, at most once per order. "
        "Use issue_refund and quote the refund id."
    )

    def __init__(self, max_refunds: int = 1):
        self.max_refunds = max_refunds

    async def wrap_tool(self, ctx, tool, args, call_next):
        if tool.name == "issue_refund":
            done = ctx.state.get("refunds", 0)
            if done >= self.max_refunds:
                raise StopAgent("This conversation has reached its refund limit.", reason="refunds")
            ctx.state["refunds"] = done + 1
        return await call_next(tool, args)


def build(offline: bool, store: SQLiteCheckpointer) -> Agent:
    if offline:
        from fastaiagent.testing import FunctionModel

        def scripted(messages):
            tool_turns = [m for m in messages if m.role.value == "tool"]
            user = next(m.content for m in reversed(messages) if m.role.value == "user")
            if "refund" in user.lower() and not tool_turns:
                return "", [{"name": "issue_refund", "arguments": {"order_id": "A-17"}}]
            if "invoice" in user.lower() and not tool_turns:
                return "", [{"name": "load_tools", "arguments": {"names": ["get_invoice"]}}]
            if "invoice" in user.lower() and len(tool_turns) == 1:
                return "", [{"name": "get_invoice", "arguments": {"invoice_id": "INV-9"}}]
            return f"Done: {tool_turns[-1].content}" if tool_turns else "How can I help?"

        llm: LLMClient = FunctionModel(scripted)
    else:
        llm = LLMClient(provider="openai", model="gpt-5-mini")
    return Agent(
        name="support-desk",
        system_prompt="You are a support agent. Use the tools; be brief.",
        llm=llm,
        tools=[
            FunctionTool(name="get_invoice", fn=get_invoice, deferred=True),
            FunctionTool(name="track_shipment", fn=track_shipment, deferred=True),
            FunctionTool(name="update_address", fn=update_address, deferred=True),
        ],
        middleware=[Refunds(max_refunds=1)],
        checkpointer=store,
    )


def main(offline: bool) -> None:
    store = SQLiteCheckpointer(db_path=os.path.join(tempfile.mkdtemp(), "cp.db"))
    agent = build(offline, store)

    print("── a refund (the middleware's own tool)")
    r = agent.run("Please refund order A-17.", execution_id="desk-1")
    print(f"  tools: {[c['tool_name'] for c in r.tool_calls]}  →  {r.output}")
    saved = store.get_last("desk-1").state_snapshot.get("middleware_state")
    print(f"  saved in the checkpoint: {saved}")

    print("── an invoice question (a deferred tool, loaded on demand)")
    r = agent.run("How much is invoice INV-9?", execution_id="desk-2")
    print(f"  tools: {[c['tool_name'] for c in r.tool_calls]}  →  {r.output}")


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    if not offline and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — running --offline")
        offline = True
    main(offline)
