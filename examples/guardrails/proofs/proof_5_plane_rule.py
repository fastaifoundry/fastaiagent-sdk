"""Proof for "5 · Between your code and the plane's rule" on docs/guardrails/guardrail-boundaries.md.

Offline: the plane's rule shape through the SDK's own builder, the shared conformance
fixture through the SDK's own runners. No plane is contacted.

A rule authored on the control plane arrives as a dict and is rebuilt onto the same
runners a local rule uses. A rule the SDK cannot rebuild is skipped, never passed.
A plane rule is never pushed back up as the agent's own. And the fixture both repos
share says the detectors agree, case by case.
"""

import asyncio
import json
import pathlib

import _common  # noqa: F401
from _common import heading

from fastaiagent import Agent
from fastaiagent.guardrail import (
    Guardrail,
    GuardrailType,
    guardrail_from_policy_rule,
)
from fastaiagent.guardrail.actions import mask_payload
from fastaiagent.guardrail.implementations import _run_pii, _run_secrets
from fastaiagent.testing import TestModel

heading("a plane rule becomes a runtime guardrail")
RULE = {"name": "no-ssn-in-tools", "implementation_type": "regex", "guardrail_type": "tool",
        "config": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b"}, "validation_mode": "blocking",
        "action": "redact-and-notify", "severity": "high", "floor": True, "on_error": "block"}
g = guardrail_from_policy_rule(RULE)
print(f"type={g.guardrail_type.value} position={g.position.value} blocking={g.blocking} "
      f"action={g.action!r} severity={g.severity} floor={g.floor} origin={g.origin}")
for impl in ("code", "quantum"):
    print(f"implementation_type={impl!r:<10} → {guardrail_from_policy_rule({**RULE, 'implementation_type': impl})}")

heading("a plane rule is enforced beside local rules, never pushed back as one")
local = Guardrail(name="local-length", fn=lambda t: len(t) < 500)
agent = Agent(name="t", llm=TestModel(response="ok"), guardrails=[local, g])
print("agent.to_dict()['guardrails'] →", [x["name"] for x in agent.to_dict()["guardrails"]])

heading("the shared conformance fixture, run through the SDK's detectors")
fixture = pathlib.Path(__file__).resolve().parents[3] / "tests" / "data" / "guardrail_conformance.json"
cases = json.loads(fixture.read_text())


def rule(impl, config):
    return Guardrail(name=f"c-{impl}", guardrail_type=GuardrailType(impl), config=config)


agree = total = 0
for impl in ("pii", "secrets"):
    runner = _run_pii if impl == "pii" else _run_secrets
    for case in cases[impl]:
        total += 1
        try:
            got = asyncio.run(runner(rule(impl, case["config"]), case["input"])).metadata
            ok = not case["expect"].get("raises") and got == case["expect"]["detail"]
        except (ValueError, ImportError):
            ok = bool(case["expect"].get("raises"))
        agree += ok
        if not ok:
            print("  DISAGREE:", impl, case["name"])
for case in cases["mask"]:
    total += 1
    out = asyncio.run(mask_payload(rule(case["type"], case["config"]), case["input"]))
    agree += (case["input"] if out is None else out) == case["expect"]["masked"]
print(f"fixture version {cases['version']}: {agree}/{total} cases agree "
      f"({len(cases['pii'])} pii · {len(cases['secrets'])} secrets · {len(cases['mask'])} mask)")
raises = [c["name"] for c in cases["pii"] if c["expect"].get("raises")]
print("cases that must raise, not pass:", raises)
