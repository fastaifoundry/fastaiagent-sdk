"""Example 107: the call-centre router as a Chain (1.84.0).

The same routing as ``106_call_center_supervisor.py`` — complaints, product
enquiries, everything else — written as a declarative graph instead of a
Supervisor. The router is a ``condition`` node with ``decision=``: it asks
OpenAI's Decisions API (``gpt-6-luna``) a ``Choice`` about the ticket and follows
the edge labelled with the answer. The queues are ``gpt-5.1`` agents.

    ticket ──▶ triage (decision node, gpt-6-luna)
                 ├─ "complaint"        ──▶ complaints agent   (gpt-5.1)
                 ├─ "product_enquiry"  ──▶ products agent     (gpt-5.1)
                 ├─ "other"            ──▶ general agent      (gpt-5.1)
                 └─ default            ──▶ general agent      refusal or confidence < 0.6

Pick the Chain when the flow is a fixed graph you want to see, validate and
version (``chain.validate()`` refuses an option with no edge; ``chain.to_dict()``
is plain JSON). Pick ``Supervisor(routing="decisions")`` when you also want the
extra routing answers (urgency, mood) handed to the worker and every reply
reviewed before it goes out.

Usage:
    zsh -lc 'python examples/107_call_center_chain.py'   # needs OPENAI_API_KEY

Output from a live run (replies abridged; wording and timings vary):

    validate(): ok
    triage node config: {"value": "complaint", "description": "Unhappy about something ..."}

    > My order A1042 arrived with a cracked lamp base. Sort it out today or I'm cancelling.
      triage → complaint (confidence 1.00) → complaints  [3180 ms]
      I'm sorry your Aurora desk lamp arrived with a cracked base. I can sort this out
      today ... 1) Free replacement ... 2) Full refund: we'll refund the $89.00 ...
    > Does the Aurora desk lamp work with my 5V 2A USB-C phone charger?
      triage → product_enquiry (confidence 1.00) → products  [2660 ms]
      Yes, it does. The Aurora desk lamp is powered via USB-C ...
    > What time do your phone lines open on Saturday?
      triage → other (confidence 0.94) → general  [3152 ms]
      Our phone lines open at 9:00 on Saturdays and close at 17:00. ...
    > Hi, I need some help please.
      triage → other (confidence 0.93) → general  [2302 ms]
      Of course, I'm here to help. What do you need assistance with today ...
"""

from __future__ import annotations

import asyncio
import json
import os
import time

from fastaiagent.agent import Agent
from fastaiagent.chain import Chain
from fastaiagent.chain.node import NodeType
from fastaiagent.llm import Choice, LLMClient
from fastaiagent.tool import FunctionTool

WORKER_MODEL = "gpt-5.1"
DECISION_MODEL = "gpt-6-luna"

# --- back-office tools (same fake data as example 106) ----------------------

_ORDERS = {"A1042": {"item": "Aurora desk lamp", "price": 89.0, "status": "delivered",
                     "note": "arrived with a cracked base"}}
_CATALOG = [
    {"name": "Aurora desk lamp", "price": 89.0, "in_stock": True,
     "facts": "LED, USB-C power, works with any 5V/2A USB-C charger"},
    {"name": "Orbit smart speaker", "price": 129.0, "in_stock": False,
     "facts": "Wi-Fi + Bluetooth 5.3, works with Alexa and Google Home, restock in 2 weeks"},
]


def lookup_order(order_id: str) -> dict:
    """Look up an order by its id (e.g. A1042)."""
    return _ORDERS.get(order_id.strip().upper(), {"error": f"no order {order_id}"})


def search_catalog(query: str) -> list[dict]:
    """Search the product catalogue by name or feature."""
    words = query.lower().split()
    return [p for p in _CATALOG if any(w in (p["name"] + p["facts"]).lower() for w in words)] or _CATALOG


def store_info(topic: str) -> str:
    """Opening hours, returns policy, delivery and account help."""
    return ("Phone lines: Mon-Fri 8:00-20:00, Sat 9:00-17:00. Returns: 30 days, free for damaged "
            "items. Address changes: until the order ships.")


# --- the graph ---------------------------------------------------------------

QUEUES = Choice(
    instructions="Which queue should handle this customer contact?",
    options={
        "complaint": "Unhappy about something that already happened: a late, damaged or "
        "wrong order, a billing error, poor service, a refund demand.",
        "product_enquiry": "A question about a product: features, compatibility, "
        "availability, sizing or price.",
        "other": "Anything else: account access, opening hours, delivery changes, unclear requests.",
    },
)

STYLE = (
    "You work in customer support at Lumen Home. The customer's message is in the "
    "'message' field of your input. Be warm, concise (under 100 words) and concrete; "
    "use your tools rather than guessing."
)


def build_chain() -> Chain:
    llm = LLMClient(provider="openai", model=WORKER_MODEL)
    chain = Chain("call-center", checkpoint_enabled=False)
    chain.add_node(
        "triage",
        type=NodeType.condition,
        decision={
            "question": QUEUES,
            "input": "{{input.message}}",
            "llm": {"model": DECISION_MODEL},
            "min_confidence": 0.6,
        },
    )
    chain.add_node("complaints", agent=Agent(
        name="complaints", llm=llm,
        system_prompt=STYLE + " Handle complaints: apologise once, look up the order, offer a "
        "free replacement or full refund for damaged items.",
        tools=[FunctionTool(name="chain_lookup_order", fn=lookup_order)]))
    chain.add_node("products", agent=Agent(
        name="products", llm=llm,
        system_prompt=STYLE + " Answer product questions from the catalogue.",
        tools=[FunctionTool(name="chain_search_catalog", fn=search_catalog)]))
    chain.add_node("general", agent=Agent(
        name="general", llm=llm,
        system_prompt=STYLE + " Handle everything else; ask a short clarifying question when "
        "the request is vague.",
        tools=[FunctionTool(name="chain_store_info", fn=store_info)]))
    chain.connect("triage", "complaints", label="complaint")
    chain.connect("triage", "products", label="product_enquiry")
    chain.connect("triage", "general", label="other")
    chain.connect("triage", "general")  # default: refused or unsure
    return chain


TICKETS = [
    "My order A1042 arrived with a cracked lamp base. Sort it out today or I'm cancelling.",
    "Does the Aurora desk lamp work with my 5V 2A USB-C phone charger?",
    "What time do your phone lines open on Saturday?",
    "Hi, I need some help please.",
]


async def run() -> None:
    chain = build_chain()
    print("validate():", chain.validate() or "ok")
    graph = json.loads(json.dumps(chain.to_dict()))  # plain JSON — version it, diff it
    print("triage node config:", json.dumps(graph["nodes"][0]["config"]["decision"]["question"]["choices"][0]))

    for message in TICKETS:
        start = time.monotonic()
        result = await chain.aexecute({"message": message})
        ms = int((time.monotonic() - start) * 1000)
        triage = result.node_results["triage"]
        handler = next(n for n in ("complaints", "products", "general") if n in result.node_results)
        reply = result.node_results[handler]["output"]
        d = triage["decision"]
        print(f"\n> {message}")
        print(f"  triage → {triage['matched']} (confidence {d['confidence']:.2f}) → {handler}  [{ms} ms]")
        print("  " + str(reply).replace("\n", "\n  "))


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — it's in ~/.zshrc; run via zsh -lc.")
        return 1
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
