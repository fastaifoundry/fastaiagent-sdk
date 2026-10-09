"""Example 106: a call-centre team whose supervisor routes with the Decisions API (1.84.0).

Every contact that reaches a call centre needs the same first move: *who should
handle this?* That is a classification, not a conversation — so the supervisor
asks OpenAI's Decisions API (served by the model ``gpt-6-luna``) and only the
workers are chat models (``gpt-5.1``):

    customer message
          │
          ▼
    ┌───────────────────────────────────┐  ONE decide() call:
    │ Supervisor(routing="decisions")   │    worker   which queue (from the workers' descriptions)
    │   router: gpt-6-luna              │    urgent   needs an answer today?
    └─────────────────┬─────────────────┘    mood     Calm < Frustrated < Angry
            ┌─────────┼──────────┐
            ▼         ▼          ▼
       complaint  product_   other            gpt-5.1 agents with back-office tools
                  enquiry
            └─────────┼──────────┘
                      ▼
     the reply is reviewed with one Decisions predicate; a weak one goes back
     once with feedback. A refused or unsure route goes to "other".

``Supervisor`` has two routing modes:

* ``routing="tools"`` (default) — a chat model delegates through tool calls and
  writes the final answer. Right for multi-step work: decompose, call several
  workers, synthesise.
* ``routing="decisions"`` — one ``decide()`` picks exactly one worker and the
  worker's reply *is* the answer. Right for single-hop routing like this.

``--compare`` runs the same tickets through the tool-calling supervisor on
``gpt-5.1`` so you can see the difference in time and routes.

Usage:
    zsh -lc 'python examples/106_call_center_supervisor.py'             # needs OPENAI_API_KEY
    zsh -lc 'python examples/106_call_center_supervisor.py --compare'

Output from a live run (replies abridged; wording, scores and timings vary):

    > My order A1042 arrived with a cracked lamp base. This is the second time — ...
      → complaint · confidence 0.99 · urgent 0.99 · mood Angry · routed in 486 ms ($0.000049)
      reply (3859 ms end to end, review 0.90):
      I'm really sorry this has happened to you twice ... I've opened a high-priority
      case for you: ID CS-2001 ... you can choose either: 1) A free replacement lamp,
      or 2) A full refund ...
    > Does the Aurora desk lamp work with my phone charger? It's a 5V 2A USB-C one.
      → product_enquiry · confidence 0.99 · urgent 0.00 · mood Calm · routed in 164 ms
      reply (2359 ms end to end, review 0.97): Yes, it will work. ...
    > What time do your phone lines open on Saturday?
      → other · confidence 1.00 · urgent 0.00 · mood Calm · routed in 170 ms
      reply (2164 ms end to end, review 0.54): Our phone lines open at 9:00 on Saturdays ...
    > Is the Orbit speaker compatible with Google Home, and when can I get one?
      → product_enquiry · confidence 0.99 · urgent 0.10 · mood Calm · routed in 191 ms
      reply (3118 ms end to end, review 0.95): ... does work with Google Home ... out of
      stock ... restock in about 2 weeks ...
    > Hi, I need some help please.
      → other · confidence 0.89 · urgent 0.00 · mood Calm · routed in 167 ms
      reply (1678 ms end to end, review 0.96): I'm here to help. What do you need ...

    cases opened: CS-2001 [high]

    routing='decisions' vs routing='tools' (gpt-5.1 delegating by tool call), no review:
      decisions   3935 ms → complaint       tools  11549 ms → complaint       | My order A1042 ...
      decisions   2676 ms → product_enquiry tools   5144 ms → product_enquiry | Does the Aurora ...
      decisions   2008 ms → other           tools   5247 ms → other           | What time do ...
      decisions   2619 ms → product_enquiry tools   7685 ms → product_enquiry | Is the Orbit ...
      decisions   1838 ms → other           tools   1375 ms → (answered itself)| Hi, I need ...

Note the last row: the tool-calling supervisor skipped delegation and answered
itself. A decision router always lands the contact in exactly one queue.
"""

from __future__ import annotations

import os
import sys
import time

from fastaiagent.agent import Agent, Supervisor, Worker
from fastaiagent.llm import LLMClient, Predicate, Score
from fastaiagent.tool import FunctionTool

WORKER_MODEL = "gpt-5.1"
DECISION_MODEL = "gpt-6-luna"

# ---------------------------------------------------------------------------
# Back-office tools (fake data, deterministic so a write-up can quote the output).
# ---------------------------------------------------------------------------

_ORDERS = {
    "A1042": {"item": "Aurora desk lamp", "price": 89.0, "status": "delivered",
              "note": "arrived with a cracked base"},
    "A1077": {"item": "Nimbus office chair", "price": 249.0, "status": "in transit", "eta": "Friday"},
}
_CATALOG = [
    {"sku": "AUR-LAMP", "name": "Aurora desk lamp", "price": 89.0, "in_stock": True,
     "facts": "LED, 3 colour temperatures, USB-C power, works with any 5V/2A USB-C charger"},
    {"sku": "NIM-CHAIR", "name": "Nimbus office chair", "price": 249.0, "in_stock": True,
     "facts": "Mesh back, adjustable lumbar support, fits users 160-195 cm, 5-year warranty"},
    {"sku": "ORB-SPKR", "name": "Orbit smart speaker", "price": 129.0, "in_stock": False,
     "facts": "Wi-Fi + Bluetooth 5.3, works with Alexa and Google Home, restock in 2 weeks"},
]
CASES: list[dict] = []


def lookup_order(order_id: str) -> dict:
    """Look up an order by its id (e.g. A1042). Returns item, price, status and notes."""
    return _ORDERS.get(order_id.strip().upper(), {"error": f"no order {order_id}"})


def open_case(queue: str, summary: str, priority: str) -> dict:
    """Open a support case. priority is 'normal' or 'high'. Returns the case id."""
    case_id = f"CS-{2000 + len(CASES) + 1}"
    CASES.append({"id": case_id, "queue": queue, "summary": summary, "priority": priority})
    return {"case_id": case_id, "priority": priority}


def search_catalog(query: str) -> list[dict]:
    """Search the product catalogue by name or feature."""
    words = query.lower().split()
    hits = [p for p in _CATALOG if any(w in (p["name"] + " " + p["facts"]).lower() for w in words)]
    return hits or _CATALOG


def store_info(topic: str) -> str:
    """Company information: opening hours, returns policy, delivery, account help."""
    return (
        "Phone lines: Mon-Fri 8:00-20:00, Sat 9:00-17:00. Returns: 30 days, free for "
        "damaged items, refunds within 5 working days. Delivery address changes: possible "
        "until the order ships. Password resets: via the 'Forgot password' link."
    )


# ---------------------------------------------------------------------------
# The team. Worker descriptions double as the router's options — write them
# the way you'd brief a new colleague on which queue owns what.
# ---------------------------------------------------------------------------

_HOUSE_STYLE = (
    "You work in customer support at Lumen Home, an online furniture and electronics "
    "store. Be warm, concise (under 120 words) and concrete. Never invent order details "
    "or product facts — use your tools. A supervisor routing note may follow the "
    "customer's message (urgency, mood); use it to set tone and priority."
)


def build_workers() -> list[Worker]:
    llm = LLMClient(provider="openai", model=WORKER_MODEL)
    return [
        Worker(
            role="complaint",
            description=(
                "The customer is unhappy about something that already happened: a late, "
                "damaged or wrong order, a billing error, poor service, or a refund demand."
            ),
            agent=Agent(
                name="complaints",
                llm=llm,
                system_prompt=_HOUSE_STYLE
                + " You handle complaints: apologise once, look up the order, open a case "
                "(priority 'high' when the note says urgent), and offer what the returns "
                "policy allows — damaged items get a free replacement or a full refund.",
                tools=[FunctionTool(name="lookup_order", fn=lookup_order),
                       FunctionTool(name="open_case", fn=open_case)],
            ),
        ),
        Worker(
            role="product_enquiry",
            description=(
                "The customer asks about a product: features, compatibility, availability, "
                "sizing or price — before or after buying."
            ),
            agent=Agent(
                name="products",
                llm=llm,
                system_prompt=_HOUSE_STYLE
                + " You answer product questions from the catalogue: features, compatibility, "
                "stock and price. If something is out of stock, say when it is expected back.",
                tools=[FunctionTool(name="search_catalog", fn=search_catalog)],
            ),
        ),
        Worker(
            role="other",
            description=(
                "Anything else: account access, opening hours, delivery address changes, "
                "general or unclear questions."
            ),
            agent=Agent(
                name="general",
                llm=llm,
                system_prompt=_HOUSE_STYLE
                + " You handle everything else: hours, account access, delivery changes, and "
                "anything unclear — ask a short clarifying question when the request is vague.",
                tools=[FunctionTool(name="store_info", fn=store_info),
                       FunctionTool(name="open_case", fn=open_case)],
            ),
        ),
    ]


def decision_desk(review: bool = True) -> Supervisor:
    decider = LLMClient(model=DECISION_MODEL)
    return Supervisor(
        name="call-center",
        workers=build_workers(),
        routing="decisions",
        router_llm=decider,
        routing_instructions="Which queue should handle this customer contact?",
        fallback_worker="other",          # refused or unsure → never guess a specialist
        routing_min_confidence=0.6,
        routing_questions={               # asked in the SAME request as the route
            "urgent": Predicate(
                instructions="The customer needs an answer today, or threatens to cancel, "
                "leave a bad review, or escalate."
            ),
            "mood": Score(instructions="How upset is the customer?",
                          levels=["Calm", "Frustrated", "Angry"]),
        },
        validate_outputs=review,          # review every reply before it goes out
        validation_mode="decisions",
        validation_llm=decider,
        # Judgeable from the message and reply alone — the reviewer hasn't seen
        # the order system, so don't ask it whether the facts are right.
        validation_criteria=(
            "The agent's reply directly addresses what the customer asked and tells "
            "the customer the next step."
        ),
    )


def tools_desk() -> Supervisor:
    """The classic supervisor: gpt-5.1 delegates by tool call, then synthesises."""
    return Supervisor(
        name="call-center-tools",
        llm=LLMClient(provider="openai", model=WORKER_MODEL),
        workers=build_workers(),
    )


TICKETS = [
    "My order A1042 arrived with a cracked lamp base. This is the second time — I want "
    "this sorted today or I'm cancelling my account.",
    "Does the Aurora desk lamp work with my phone charger? It's a 5V 2A USB-C one.",
    "What time do your phone lines open on Saturday?",
    "Is the Orbit speaker compatible with Google Home, and when can I get one?",
    "Hi, I need some help please.",
]


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — it's in ~/.zshrc; run via zsh -lc.")
        return 1

    desk = decision_desk()
    for message in TICKETS:
        start = time.monotonic()
        result = desk.run(message)
        total_ms = int((time.monotonic() - start) * 1000)
        r = result.route
        urgent = r.answers.predicates["urgent"].probability
        mood = r.answers.scores["mood"].level
        conf = f"{r.confidence:.2f}" if r.confidence is not None else "refused"
        reviews = ", ".join(f"{p:.2f}" for p in r.reviews if p is not None)
        print(f"\n> {message}")
        print(
            f"  → {r.worker}{' (fallback)' if r.fallback else ''} · confidence {conf} · "
            f"urgent {urgent:.2f} · mood {mood} · routed in {r.latency_ms} ms (${r.cost_usd:.6f})"
        )
        print(f"  reply ({total_ms} ms end to end, review {reviews}):")
        print("  " + str(result.output).replace("\n", "\n  "))
    if CASES:
        print("\ncases opened:", ", ".join(f"{c['id']} [{c['priority']}]" for c in CASES))

    if "--compare" in sys.argv:
        # Like for like: both without the review step.
        print("\nrouting='decisions' vs routing='tools' (gpt-5.1 delegating by tool call), no review:")
        routed, classic = decision_desk(review=False), tools_desk()
        for message in TICKETS:
            start = time.monotonic()
            d = routed.run(message)
            d_ms = int((time.monotonic() - start) * 1000)
            start = time.monotonic()
            t = classic.run(message)
            t_ms = int((time.monotonic() - start) * 1000)
            called = [c.get("tool_name", "").removeprefix("delegate_to_") for c in t.tool_calls]
            print(
                f"  decisions {d_ms:>6} ms → {d.route.worker:<16}"
                f"tools {t_ms:>6} ms → {', '.join(called) or '(answered itself)':<16}| {message[:38]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
