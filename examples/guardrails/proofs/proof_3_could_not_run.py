"""Proof for "3 · Between 'could not run' and 'found nothing'" on docs/guardrails/guardrail-boundaries.md.

Offline: real rules through run_guardrail and a real Agent on the SDK's TestModel.

A check that cannot run reports that it could not run — errored=True — and on_error
decides what that costs. A configuration that cannot check anything is the same
case: it errors, it never reports a clean verdict. Either way the outcome blocks
whatever action the rule was configured with.
"""

import asyncio

import _common  # noqa: F401
from _common import firings, heading

from fastaiagent import Agent
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail import (
    Guardrail,
    GuardrailPosition,
    GuardrailType,
    cost_limit,
    run_guardrail,
)
from fastaiagent.testing import TestModel


def down(_):
    raise RuntimeError("moderation API timed out")


heading("on_error: what an un-runnable check means")
for on_error in ("allow", "block"):
    rule = Guardrail(name="moderation", position=GuardrailPosition.output, fn=down, on_error=on_error)
    agent = Agent(name="t", llm=TestModel(response="fine"), guardrails=[rule])
    try:
        r = agent.run("go")
        print(f"on_error={on_error:<5} → run completed · firing={firings(r)[0]}")
    except GuardrailBlockedError as e:
        print(f"on_error={on_error:<5} → GuardrailBlockedError: {e}")

heading("a configuration that cannot check anything")
UNUSABLE = [
    ("schema", {}, "no schema at all"),
    ("schema", {"schema": {}}, "an empty schema validates everything"),
    ("regex", {"pattern": ""}, "an empty pattern matches everywhere"),
    ("classifier", {"categories": {}}, "an empty category map"),
    ("pii", {"entities": []}, "scans for no entities"),
    ("pii", {"backend": "presidoo"}, "a backend that does not exist"),
    ("topic", {"topics": []}, "an empty topic list"),
    ("topic", {"topics": ["ok"], "mode": "denyy"}, "a mistyped polarity"),
    ("content_safety", {"categories": ["S99"]}, "no known hazard category"),
    ("groundedness", {}, "no context to score against"),
]
print(f"{'type':<15} {'config':<32} passed errored  message")
for impl, config, why in UNUSABLE:
    rule = Guardrail(name=f"u-{impl}", guardrail_type=GuardrailType(impl), config=config)
    r = asyncio.run(run_guardrail(rule, "mail bob@acme.com about the refund"))
    print(f"{impl:<15} {str(config):<32} {str(r.passed):<6} {str(r.errored):<8} {(r.message or '')[:52]}")

heading("errored always blocks, whatever the action")
rule = Guardrail(name="pii-mask", guardrail_type=GuardrailType.pii, config={"entities": []}, action="mask")
r = asyncio.run(run_guardrail(rule, "mail bob@acme.com"))
print(f"action={r.action!r} → action_taken={r.action_taken!r} errored={r.errored}")

heading("a builtin that cannot price the run")
r = cost_limit(max_usd=1.0).execute("any text")   # outside a run: no spend is tracked
print(f"cost_limit outside a run → passed={r.passed} errored={r.errored}: {(r.message or '')[:70]}…")
