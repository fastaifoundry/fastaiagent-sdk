"""Example 102: OpenAI's Decisions API — typed answers, not text (1.84.0).

``LLMClient.decide()`` asks fixed-answer questions about some evidence and gets
probabilities back — no prompt to write, no JSON to parse:

  * ``Predicate`` — how likely is a condition true?         → ``probability``
  * ``Choice``    — which of these options?                  → ``choice`` + ``confidence``
  * ``Score``     — where on this ordered scale?             → ``score`` (0..n-1), ``level``

One request can ask several questions about the same text (or image). The
endpoint serves ``gpt-6-luna``; it bills input tokens only, so ``cost_usd`` is
tiny. The last part runs with **no network at all**: ``TestModel(decisions=...)``
is how you test code that calls ``decide()``.

Usage:
    zsh -lc 'python examples/102_decisions_basics.py'   # needs OPENAI_API_KEY

Output from a live run (probabilities and token counts vary a little):
    department   -> billing (confidence 1.00)
    duplicate    -> p=1.00
    severity     -> Inconvenient (score 1.04 of 2, normalized 0.52)
    complaint    -> True (a bool, not the string "True")
    cost $0.000050 · 503 input tokens · request req_1f77c08e...
    image colour -> red
    offline      -> p=0.97 (TestModel, no network)
"""

from __future__ import annotations

import base64
import os

from fastaiagent.llm import Choice, LLMClient, Predicate, Score
from fastaiagent.multimodal.image import Image
from fastaiagent.testing import TestModel

TICKET = "I was charged twice for the same subscription renewal. Please refund one."

# An 8x8 solid red PNG, inline so the example needs no files.
RED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAEklEQVR4nGP4z8CAFWEXHbQSACj/P8Fu7N9hAAAAAElFTkSuQmCC"
)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — it's in ~/.zshrc; run via zsh -lc.")
        return 1

    llm = LLMClient(model="gpt-6-luna")

    # Several questions about one piece of text, keyed by name.
    r = llm.decide(
        TICKET,
        {
            "department": Choice(
                instructions="Which department should handle this?",
                options={
                    "billing": "Payments, invoices, refunds.",
                    "technical": "Product bugs.",
                    "other": "Anything else.",
                },
            ),
            "duplicate": Predicate(instructions="The customer reports a duplicate charge."),
            "severity": Score(
                instructions="How severe is this issue for the customer?",
                levels=["Cosmetic", "Inconvenient", "Blocking"],
            ),
            # Choice values are typed: True is a bool, not the string "True".
            "complaint": Choice(instructions="Is this a complaint?", options=[True, False]),
        },
    )
    dept = r.choices["department"]
    sev = r.scores["severity"]
    print(f"department   -> {dept.choice} (confidence {dept.confidence:.2f})")
    print(f"duplicate    -> p={r.predicates['duplicate'].probability:.2f}")
    print(f"severity     -> {sev.level} (score {sev.score:.2f} of 2, normalized {sev.normalized:.2f})")
    print(f"complaint    -> {r.choices['complaint'].choice!r} (a bool, not the string \"True\")")
    print(f"cost ${r.cost_usd:.6f} · {r.usage['input_tokens']} input tokens · request {r.request_id}")

    # Images go in as evidence too (sent inline as a data URL).
    img = llm.decide(
        ["What colour fills this image?", Image.from_bytes(RED_PNG, "image/png")],
        Choice(name="colour", instructions="Dominant colour", options=["red", "green", "blue"]),
    )
    print(f"image colour -> {img['colour'].choice}")

    # Testing code that calls decide(): canned answers, a real span, no network.
    offline = TestModel(
        decisions={"answers": [{"type": "predicate", "name": "spam", "probability": 0.97}]}
    )
    spam = offline.decide("WIN $$$ NOW", Predicate(name="spam", instructions="This is spam."))
    print(f"offline      -> p={spam['spam'].probability:.2f} (TestModel, no network)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
