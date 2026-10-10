"""Proof for "5 · Between the prompt and the model" on docs/prompts/prompt-boundaries.md.

Live: needs OPENAI_API_KEY (gpt-4.1-mini, temperature 0). Costs a few cents.

Two versions of one routing prompt, the same ten tickets, the same model. The
dataset decides which version the production alias points at — not an eyeball.
"""

import os
import sys

import _common
from _common import heading

from fastaiagent import Agent, LLMClient
from fastaiagent.eval import Dataset, Scorer, ScorerResult, evaluate
from fastaiagent.prompt import PromptRegistry

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set — this proof calls a real model.")

MODEL = os.environ.get("PROOF_MODEL", "gpt-4.1-mini")

V1 = (
    "You route support tickets for a SaaS product to exactly one queue: "
    "billing, technical or account. Reply with the queue name only, in lowercase."
)
V2 = V1 + (
    "\n\nRules:\n"
    "- billing: anything that asks about money — a charge, a refund, an invoice, a price, "
    "a plan change. Money comes first: a charge for a feature that is broken is billing.\n"
    "- account: sign-in, passwords, 2FA, teammates and roles, deleting an account or its data.\n"
    "- technical: errors, outages, the API, webhooks, exports and integrations, when no money "
    "is involved."
)

CASES = [
    ("I was charged twice this month.", "billing"),
    ("The API returns 500 on every request since 9am.", "technical"),
    ("I can't log in; the 2FA code never arrives.", "account"),
    ("Please delete my account and all my data.", "account"),
    ("You billed me for the analytics add-on but the dashboard never loads.", "billing"),
    ("How do I add a teammate as an admin?", "account"),
    ("The webhook stopped firing after your maintenance window.", "technical"),
    ("Can I get an invoice with our VAT number on it?", "billing"),
    ("The export button does nothing on the Pro plan.", "technical"),
    ("Why did my plan price go up this month?", "billing"),
]


class QueueMatch(Scorer):
    name = "queue_match"

    def score(self, input: str, output: str, expected: str | None = None, **kw) -> ScorerResult:
        got = (output or "").strip().strip(".").lower()
        passed = got == (expected or "")
        return ScorerResult(score=1.0 if passed else 0.0, passed=passed,
                            reason="" if passed else f"got {got!r}, expected {expected!r}")


reg = PromptRegistry(path=str(_common.SCRATCH))
reg.register("ticket-router", V1)  # v1
reg.register("ticket-router", V2)  # v2
reg.set_alias("ticket-router", version=1, alias="production")
dataset = Dataset.from_list([{"input": i, "expected": e} for i, e in CASES])

rates: dict[int, float] = {}
for version in (1, 2):
    prompt = reg.load("ticket-router", version=version)
    agent = Agent(
        name="ticket-router",
        system_prompt=prompt,
        llm=LLMClient(provider="openai", model=MODEL, temperature=0),
    )
    results = evaluate(agent_fn=agent.run, dataset=dataset, scorers=[QueueMatch()], persist=False)
    scored = results.scores["queue_match"]
    rates[version] = sum(1 for r in scored if r.passed) / len(scored)
    heading(f"v{version} on {len(CASES)} tickets ({MODEL})")
    print(f"pass rate {rates[version]:.2f}")
    for case, r in zip(CASES, scored):
        if not r.passed:
            print(f"  ✗ {case[0]!r}: {r.reason}")

heading("the gate")
if rates[2] >= rates[1]:
    reg.set_alias("ticket-router", version=2, alias="production")
    print(f"v2 ({rates[2]:.2f}) ≥ v1 ({rates[1]:.2f}): production moved to v2")
else:
    print(f"v2 ({rates[2]:.2f}) < v1 ({rates[1]:.2f}): production stays on v1")
print("production →", f"v{reg.load('ticket-router', alias='production').version}")
