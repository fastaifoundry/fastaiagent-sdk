"""Proof for "6 · Between the agent loop and everything else" on docs/guardrails/guardrail-boundaries.md.

Offline: real guardrails through to_dict()/from_dict() and the harness helper.

A rule leaves the process as a dict — to the plane, to a replay, to local.db. The
config-based types come back whole. A builtin comes back armed. A custom fn cannot
travel, and the restored rule says so by erroring, never by passing. A foreign
runtime that borrows the verdict can block and warn, and must block where it
cannot rewrite.
"""

import _common  # noqa: F401
from _common import heading

from fastaiagent.guardrail import (
    Guardrail,
    GuardrailResult,
    GuardrailType,
    cost_limit,
    no_pii,
)
from fastaiagent.guardrail.actions import harness_halts

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

heading("a foreign runtime: what a proxy can honour")
g = Guardrail(name="r", guardrail_type=GuardrailType.regex, config={"pattern": "x"})
for taken, passed in (("none", True), ("warned", False), ("blocked", False), ("masked", False), ("reask", False)):
    r = GuardrailResult(passed=passed, action_taken=taken, message="ssn in output",
                        modified_data="[REDACTED]" if taken == "masked" else None)
    verdict = harness_halts(g, r)
    print(f"action_taken={taken:<8} → {'continue' if verdict is None else 'stop: ' + verdict[:70] + '…'}")
