"""Proof for "6 · Between the agent loop and everything else" on docs/guardrails/guardrail-boundaries.md.

Offline: real guardrails through to_dict()/from_dict() and run_guardrail().

A rule leaves the process as a dict — to the plane, to a replay, to local.db. The
config-based types come back whole. A builtin comes back armed. A custom fn cannot
travel, and the restored rule says so by erroring, never by passing. A runtime
that borrows the verdict (another framework's agent) calls run_guardrail() on the
text it holds and acts on the outcome: continue, stop, or send the masked text.
"""

import asyncio

import _common  # noqa: F401
from _common import heading

import fastaiagent as fa
from fastaiagent.guardrail import (
    Guardrail,
    GuardrailType,
    cost_limit,
    no_pii,
)

TEXT = "mail bob@acme.com, ssn 123-45-6789"


def roundtrip(g: Guardrail) -> Guardrail:
    return Guardrail.from_dict(g.to_dict())


heading("what survives a round-trip")
cases = [
    ("regex (config)", Guardrail(name="ssn", guardrail_type=GuardrailType.regex, config={"pattern": r"\d{3}-\d{2}-\d{4}"})),
    ("pii (config)", Guardrail(name="pii", guardrail_type=GuardrailType.pii, config={"entities": ["email"]})),
    ("no_pii() builtin", no_pii()),
    ("cost_limit(1.0) builtin", cost_limit(max_usd=1.0)),
    ("custom fn", Guardrail(name="mine", fn=lambda t: "acme" not in t)),
]
print(f"{'rule':<24} {'before: passed':<15} after: passed  errored  message")
for label, g in cases:
    before = g.execute(TEXT)
    after = roundtrip(g).execute(TEXT)
    print(f"{label:<24} {str(before.passed):<15} {str(after.passed):<14} {str(after.errored):<8} {(after.message or '')[:60]}")

heading("a borrowing runtime: run_guardrail() on the text it holds")
for action in ("warn", "block", "mask"):
    g = Guardrail(name="ssn", guardrail_type=GuardrailType.regex,
                  config={"pattern": r"\d{3}-\d{2}-\d{4}"}, action=action)
    r = asyncio.run(fa.run_guardrail(g, TEXT))
    if r.passed or r.action_taken == "warned":
        step = f"continue with: {TEXT}"
    elif r.action_taken == "masked":
        step = f"send instead:  {r.modified_data}"
    else:
        step = f"stop:          {r.message}"
    print(f"action={action:<6} action_taken={r.action_taken:<8} → {step}")
