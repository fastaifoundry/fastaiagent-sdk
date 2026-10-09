"""Example 103: route on meaning with the Decisions API (1.84.0).

Two ways to put a ~200 ms classifier in front of your agents:

  1. **A Chain decision node.** A ``condition`` node with ``decision=`` asks a
     ``Choice`` about the input and follows the edge labelled with the answer. A
     refusal, or a confidence under ``min_confidence``, takes the default edge —
     a human, here — never a guessed branch.
  2. **``decision_tool``.** Hand an agent the classifier as a tool and let *it*
     decide when to triage. The tool returns the answers with their probabilities.

Usage:
    zsh -lc 'python examples/103_decision_routing.py'   # needs OPENAI_API_KEY

Output from a live run (the agent's wording varies):
    'The app crashes every time I log in.' -> tech_agent (technical, confidence 1.00)
    'I was charged twice for my renewal.' -> billing_agent (billing, confidence 1.00)
    'Can I talk to someone about my account?' -> human (other, confidence 0.99)
    agent called triage_ticket -> {'department': {'choice': 'billing', 'confidence': 1.0, ...}}
    agent reply: The billing team will help you with your issue regarding the double charge ...
"""

from __future__ import annotations

import asyncio
import json
import os

from fastaiagent.agent import Agent
from fastaiagent.chain import Chain
from fastaiagent.chain.node import NodeType
from fastaiagent.llm import Choice, LLMClient
from fastaiagent.testing import TestModel
from fastaiagent.tool import decision_tool

TEAMS = Choice(
    instructions="Which team should handle this support ticket?",
    options={
        "billing": "Payments, invoices and refunds.",
        "technical": "Bugs, crashes and outages.",
        "other": "Anything else.",
    },
)


def build_chain() -> Chain:
    chain = Chain("triage", checkpoint_enabled=False)
    chain.add_node(
        "triage",
        type=NodeType.condition,
        decision={
            "question": TEAMS,
            "input": "{{input.message}}",
            "llm": {"model": "gpt-6-luna"},
            "min_confidence": 0.6,
        },
    )
    # Stand-in agents so the routing is what you see; swap in real ones.
    for name in ("billing_agent", "tech_agent", "human"):
        chain.add_node(name, agent=Agent(name=name, llm=TestModel(response=f"{name} on it")))
    chain.connect("triage", "billing_agent", label="billing")
    chain.connect("triage", "tech_agent", label="technical")
    chain.connect("triage", "human", label="other")
    chain.connect("triage", "human")  # default: refusal or low confidence
    return chain


async def run_chain() -> None:
    chain = build_chain()
    for message in (
        "The app crashes every time I log in.",
        "I was charged twice for my renewal.",
        "Can I talk to someone about my account?",
    ):
        result = await chain.aexecute({"message": message})
        routed = next(n for n in ("billing_agent", "tech_agent", "human") if n in result.node_results)
        d = result.node_results["triage"]["decision"]
        print(f"{message!r} -> {routed} ({d['choice']}, confidence {d['confidence']:.2f})")


def run_agent_with_tool() -> None:
    triage = decision_tool({"department": TEAMS}, llm=LLMClient(model="gpt-6-luna"), name="triage_ticket")
    agent = Agent(
        name="support",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[triage],
        system_prompt="Call triage_ticket on the user's message, then tell them which team will help.",
    )
    result = agent.run("I was charged twice for my renewal, please fix it.")
    for call in result.tool_calls:
        if call.get("tool_name") == "triage_ticket":
            output = call.get("output") or call.get("result")
            if isinstance(output, str):
                output = json.loads(output)
            print(f"agent called triage_ticket -> {output}")
    print(f"agent reply: {result.output}")


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — it's in ~/.zshrc; run via zsh -lc.")
        return 1
    asyncio.run(run_chain())
    run_agent_with_tool()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
