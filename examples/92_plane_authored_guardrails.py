"""Example 92: Plane-authored guardrails — authored centrally, enforced at the edge.

A guardrail an admin creates on the Enterprise plane is distributed to connected
SDKs via ``GET /public/v1/policy`` and enforced *locally* by the runtime — right
beside an agent's own ``guardrails=[...]``. A connected agent with **no local
guardrails** still blocks on a centrally-authored rule (added in 1.45.0).

This example is deterministic and needs no API key or live plane: it populates the
connection's policy cache with the same ``guardrail_rules`` shape the plane returns,
then shows the two mechanisms:

    1. ``guardrail_from_policy_rule`` turns one plane rule into a runtime Guardrail
       (reusing the SDK's own regex/schema/classifier/llm_judge runners — no new
       check engine; a ``code`` rule whose logic lives server-side is skipped).
    2. ``Agent._effective_guardrails()`` merges plane rules with local ones, scoped
       by ``agent_ids``, so an agent enforces both.

When you actually ``fa.connect(...)``, step 0 (the ``/policy`` pull) happens for
real and everything below is automatic — no code change on the agent.
"""

from __future__ import annotations

from fastaiagent import (
    Agent,
    LLMClient,
    guardrail_from_policy_rule,
    no_pii,
    plane_guardrails_for_agent,
)
from fastaiagent.client import _connection
from fastaiagent.guardrail.from_policy import clear_cache  # test-only cache reset

# The shape the plane returns from GET /public/v1/policy. An admin authored a regex
# rule that blocks US SSNs in output, domain-wide (empty agent_ids).
PLANE_RULE = {
    "id": "gr_ssn",
    "name": "block-ssn-output",
    "guardrail_type": "output",
    "validation_mode": "blocking",
    "implementation_type": "regex",
    "config": {"pattern": r"\b\d{3}-\d{2}-\d{4}\b"},
    "tripwire_message": "Output contained an SSN (plane policy).",
    "on_error": "block",
    "agent_ids": [],  # domain-wide; a non-empty list scopes it to those agents
}


def main() -> int:
    # 1. One plane rule -> one runtime Guardrail, enforced by the SDK's own runner.
    g = guardrail_from_policy_rule(PLANE_RULE)
    assert g is not None
    print(f"Built '{g.name}' from the plane rule: type={g.guardrail_type.value}, "
          f"position={g.position.value}, blocking={g.blocking}, on_error={g.on_error}")
    print("  blocks SSN output :", g.execute("your ssn is 123-45-6789").passed is False)
    print("  passes clean output:", g.execute("no sensitive data here").passed is True)

    # A `code` rule's logic is a server-side callable the SDK doesn't have -> skipped
    # (not silently passed).
    code_rule = {**PLANE_RULE, "name": "server-side", "implementation_type": "code", "config": {}}
    print("  code rule skipped  :", guardrail_from_policy_rule(code_rule) is None)

    # 2. Simulate being connected: the policy cache is what fa.connect() pulls.
    #    (When you really connect, you don't do this — it's automatic.)
    _connection.policy_cache = {"version": "v1", "guardrail_rules": [PLANE_RULE]}
    clear_cache()
    try:
        # An agent with NO local guardrails still enforces the plane rule.
        agent = Agent(
            name="support",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            # guardrails=[]  -> nothing local
        )
        enforced = [gr.name for gr in agent._effective_guardrails()]
        print(f"\nAgent with 0 local guardrails, connected -> enforces: {enforced}")

        # Local + plane merge: a local guardrail is enforced alongside the plane one.
        agent_local = Agent(
            name="support2",
            llm=LLMClient(provider="openai", model="gpt-4o"),
            guardrails=[no_pii()],
        )
        merged = sorted(gr.name for gr in agent_local._effective_guardrails())
        print(f"Agent with a local no_pii, connected  -> enforces: {merged}")

        # Scoping: a rule scoped to other agents does not apply here.
        scoped = {**PLANE_RULE, "agent_ids": ["some-other-agent"]}
        _connection.policy_cache = {"version": "v2", "guardrail_rules": [scoped]}
        clear_cache()
        print(f"Rule scoped to another agent           -> enforces: "
              f"{[gr.name for gr in plane_guardrails_for_agent('me')]}")
    finally:
        # Leave the process as we found it (unconnected).
        _connection.policy_cache = None
        clear_cache()

    print("\nWhen connected for real: create the guardrail on the plane, fa.connect(), done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
