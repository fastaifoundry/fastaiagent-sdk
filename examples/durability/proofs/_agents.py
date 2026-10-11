"""The small offline agent every durability proof runs: a refund bot on the SDK's
FunctionModel whose tool can pause, charge, crash or fail on demand.

The model decides from the conversation, not from a counter: no tool result yet →
ask for the tool; a tool result present → answer. That is what lets a run that
paused in one process be resumed in another, or be resumed at all.
"""

from __future__ import annotations

import os
from typing import Any

from fastaiagent import Agent, FunctionTool, interrupt
from fastaiagent.testing import FunctionModel

#: Side-effect counters a proof reads back.
charges: list[dict[str, Any]] = []
notified: list[str] = []
model_calls: list[int] = []
_raised = {"done": False}


def _tool_results(messages) -> int:
    return sum(1 for m in messages if m.role.value == "tool")


def make_responder(amount: int, *, two_tools: bool = False):
    def responder(messages):
        model_calls.append(len(model_calls) + 1)
        answered = _tool_results(messages) > 0
        if answered and os.environ.get("PROOF_CRASH") == "model":
            os._exit(3)  # the process dies while 'waiting on the LLM'
        if answered and os.environ.get("PROOF_RAISE") == "model" and not _raised["done"]:
            _raised["done"] = True
            raise RuntimeError("provider returned 503")
        if answered:
            return "Refund for order 1042 is on its way."
        calls = [{"name": "refund", "arguments": {"order": "1042", "amount": amount}}]
        if two_tools:
            calls.append({"name": "notify", "arguments": {"order": "1042"}})
        return "", calls

    return responder


def refund(order: str, amount: int) -> dict[str, Any]:
    """Refund an order; amounts over 10,000 need a manager."""
    if os.environ.get("PROOF_CRASH") == "tool":
        os._exit(2)  # the process dies inside the tool, after its checkpoint
    if amount > 10_000:
        decision = interrupt(reason="manager_approval",
                             context={"order": order, "amount": amount, "balance": 100})
        if not decision.approved:
            return {"approved": False}
    charges.append({"order": order, "amount": amount})
    return {"approved": True, "charge": len(charges)}


def notify(order: str) -> str:
    """Tell the customer."""
    notified.append(order)
    return "sent"


def agent(checkpointer, *, amount: int = 50_000, two_tools: bool = False,
          tools: list[FunctionTool] | None = None) -> Agent:
    return Agent(
        name="refund-bot",
        llm=FunctionModel(make_responder(amount, two_tools=two_tools)),
        tools=tools or [FunctionTool(name="refund", fn=refund), FunctionTool(name="notify", fn=notify)],
        checkpointer=checkpointer,
    )
