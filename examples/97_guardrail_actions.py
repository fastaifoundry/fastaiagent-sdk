"""Example 97: Guardrail actions — what a failure costs, not just that it failed.

Until 1.57.0 a guardrail had exactly one consequence: raise. A rule an operator
authored in the console as "Mask PII in output" blocked the run instead of
redacting it. Wire v1.9 adds an ``action`` to every rule, and this example shows
each one doing what it says.

A guardrail answers three independent questions. Keep them apart:

    blocking   — does it run inline, and can it halt?      (scheduling)
    on_error   — what does an *un-runnable* check mean?    (degradation)
    action     — what does a genuine failure cost?         (consequence)

``blocking=False`` means "run concurrently, don't wait". ``action="warn"`` is a
different thing: the check still runs inline and the caller still waits — it just
doesn't stop.

Deterministic: no API key, no live plane, no model call. It populates the
connection's policy cache with the shape the plane returns from
``GET /public/v1/policy`` and runs an agent over the offline TestModel.

Usage:
    python examples/97_guardrail_actions.py

Expected output (snapshot — real run, no credentials):
    1. block    'his ssn is 123-45-6789'   -> BLOCKED: Pattern matched
    2. warn     'his ssn is 123-45-6789'   -> 'his ssn is 123-45-6789'  (recorded, not stopped)
    3. mask     'his ssn is 123-45-6789'   -> 'his ssn is [REDACTED]'
    4. override 'his ssn is 123-45-6789'   -> 'I cannot share account identifiers.'
    5. reask    'his ssn is 123-45-6789'   -> 'I cannot share that.'  (model re-prompted)

    unknown action -> BLOCKED   (an action this build cannot perform fails closed)
    pre-v1.9 rule  -> BLOCKED   (no `action` key at all: exactly as it behaved before)
    mask, no span  -> BLOCKED   (a mask with nothing to redact never passes the payload)

    What actually fired (1.64.0):
      output     -> 'his ssn is 123-45-6789'   (unchanged — `warn` does not stop the run)
      fired=True  ssn-warn at output: warned

See docs/guardrails/actions.md for the full contract.
"""

from __future__ import annotations

from fastaiagent import Agent
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.agent.agent import AgentConfig
from fastaiagent.client import _connection
from fastaiagent.guardrail.from_policy import clear_cache  # test-only cache reset
from fastaiagent.testing.models import TestModel

SSN = r"\b\d{3}-\d{2}-\d{4}\b"
LEAK = "his ssn is 123-45-6789"


def plane_rule(action: str, **config) -> dict:
    """One rule, exactly as GET /public/v1/policy returns it at wire v1.9."""
    return {
        "id": f"gr_{action}",
        "name": f"ssn-{action}",
        "guardrail_type": "output",
        "validation_mode": "blocking",
        "implementation_type": "regex",
        "config": {"pattern": SSN, "should_match": False, **config},
        "tripwire_message": "Output contained an SSN (plane policy).",
        "on_error": "block",
        "agent_ids": [],  # domain-wide
        # --- new at v1.9 ---
        "action": action,  # block | warn | mask | override | reask
        "severity": "high",  # carried and shown; changes no enforcement
        "floor": False,  # the domain baseline only an admin may change
    }


def run_with(rule: dict, *, responses: list[str] | str = LEAK) -> str:
    """Connect an agent to a one-rule policy and return what the caller receives."""
    _connection.policy_cache = {"version": rule["id"], "guardrail_rules": [rule]}
    clear_cache()
    try:
        agent = Agent(
            name="support",
            llm=TestModel(response=responses),
            # No local guardrails: everything here is authored on the plane.
            config=AgentConfig(guardrail_retries=1),
        )
        return agent.run("what is his ssn?").output
    except GuardrailBlockedError as exc:
        return f"BLOCKED: {exc}"


def main() -> int:
    print("Plane-authored guardrails, one rule per action.")
    print(f"The model keeps replying: {LEAK!r}\n")

    try:
        # 1. block — today's behaviour, and every rule that predates v1.9.
        print(f"1. block    -> {run_with(plane_rule('block'))}")

        # 2. warn — evaluated inline, recorded, does not stop the run. The
        #    verdict is still a failure; the evidence stays honest.
        print(f"2. warn     -> {run_with(plane_rule('warn'))!r}")

        # 3. mask — redact the offending span and carry on with the redacted
        #    value. Only regex and classifier can do this; they are the two
        #    types that locate the offending text.
        print(f"3. mask     -> {run_with(plane_rule('mask', mask_token='[REDACTED]'))!r}")

        # 4. override — substitute the operator's copy for the payload.
        refusal = plane_rule("override")
        refusal["config"]["override_message"] = "I cannot share account identifiers."
        print(f"4. override -> {run_with(refusal)!r}")

        # 5. reask — re-prompt the model with the failure as feedback. The one
        #    action the plane cannot perform: it runs no agent loop, so centrally
        #    it records the intent and fails closed. Bounded by
        #    AgentConfig.guardrail_retries; exhausting the cap blocks, because a
        #    re-ask that never converges must not become a silent pass.
        print(
            "5. reask    -> "
            f"{run_with(plane_rule('reask'), responses=[LEAK, 'I cannot share that.'])!r}"
        )

        # An unknown action, or none at all, fails closed to `block`. The strict
        # reading is the safe one for a safety control: an SDK that meets an
        # action it cannot perform must not let the payload through.
        unknown = plane_rule("block")
        unknown["action"] = "teleport"
        print(f"\nunknown action -> {run_with(unknown)}")

        pre_v19 = plane_rule("block")
        del pre_v19["action"], pre_v19["severity"], pre_v19["floor"]
        print(f"pre-v1.9 rule  -> {run_with(pre_v19)}")

        # A mask that finds nothing to mask blocks rather than passing the
        # payload through untouched. A `should_match=True` rule fails when its
        # pattern is *absent*, so there is no span to redact — the honest
        # outcome is a block, not a pass.
        no_span = plane_rule("mask")
        no_span["config"]["should_match"] = True
        print(f"mask, no span  -> {run_with(no_span, responses='all clear')}")

        # Case 2 above is the awkward one: `warn` returns the payload unchanged,
        # so the printed string is identical to a run where nothing fired at
        # all. Since 1.64.0 the result says which rules ran and what they did —
        # the only way to see a non-halting outcome with no UI and no plane.
        print("\nWhat actually fired (1.64.0):")
        _connection.policy_cache = {"version": "w", "guardrail_rules": [plane_rule("warn")]}
        clear_cache()
        agent = Agent(name="support", llm=TestModel(response=LEAK))
        result = agent.run("what is his ssn?")
        print(f"  output     -> {result.output!r}   (unchanged — `warn` does not stop the run)")
        for g in result.guardrails:
            print(f"  fired={g.fired()!s:<5} {g.name} at {g.position}: {g.action_taken}")
    finally:
        # Leave the process as we found it (unconnected).
        _connection.policy_cache = None
        clear_cache()

    print("\nWhen connected for real: author the rule in the console, fa.connect(), done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
