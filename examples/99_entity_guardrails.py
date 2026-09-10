"""Example 99: Entity guardrails — the first rules that can redact, not just refuse.

Every guardrail before these answered one question: does this pass? A judge
returns a verdict, so the only thing it can do with a failure is stop the run.
``pii`` and ``secrets`` ask a *detector* instead, and a detector returns
**offsets** — which is what makes redaction possible:

    action="block"   the run stops
    action="mask"    the offending spans are replaced and the run continues

The detection itself is not new. ``detect_pii`` and ``detect_secrets`` have
backed the ``no_pii()`` / ``no_secrets()`` builtins and the ``PIILeakage`` scorer
for a long time. What these types add is the ability to rebuild that check from a
rule's config — so an operator can author it centrally and every connected agent
enforces it.

Two properties are worth watching in the output below:

  * **Overlapping spans merge.** One secret routinely matches two patterns, and
    replacing the ranges independently leaves fragments of the very value being
    redacted.
  * **The result carries counts, never values.** Guardrail metadata reaches a
    control plane's durable, tenant-visible row. The control that *finds*
    personal data must not become a standing database of it.

Deterministic: no API key, no live plane, no model call. These detectors are
regex.

Usage:
    zsh -lc 'python examples/99_entity_guardrails.py'

See docs/guardrails/actions.md for the full contract.
"""

from __future__ import annotations

import json

from fastaiagent.guardrail.from_policy import guardrail_from_policy_rule
from fastaiagent.guardrail.guardrail import Guardrail, GuardrailType

CUSTOMER = "Email dana@example.com or call 555-123-4567 — SSN 123-45-6789."
DEPLOY = 'ship it with api_key = "sk-proj-abcdefghijklmnopqrstuvwxyz123456" tonight'


def rule(impl: str, action: str = "block", **config) -> Guardrail:
    return Guardrail(
        name=f"{impl}-{action}",
        guardrail_type=GuardrailType(impl),
        config=config,
        action=action,
    )


def main() -> None:
    # ---------------------------------------------------------------- #
    # 1. Block vs mask, same detector
    # ---------------------------------------------------------------- #
    print("\n  Same detection, two consequences:\n")
    print(f"    input   {CUSTOMER!r}")

    blocked = rule("pii").execute(CUSTOMER)
    print(f"    block   -> {blocked.action_taken}  found={blocked.metadata['found']}")

    masked = rule("pii", "mask").execute(CUSTOMER)
    print(f"    mask    -> {masked.action_taken}   {masked.modified_data!r}")

    # ---------------------------------------------------------------- #
    # 2. Overlapping patterns merge into one redaction
    # ---------------------------------------------------------------- #
    from fastaiagent._internal.safety_detectors import detect_secrets

    print("\n  One secret, two patterns — merged before replacement:\n")
    print(f"    input   {DEPLOY!r}")
    for m in detect_secrets(DEPLOY):
        print(f"      matched {m.kind:<16} [{m.start}:{m.end}]")
    creds = rule("secrets", "mask").execute(DEPLOY)
    print(f"    mask    -> {creds.modified_data!r}")
    print("    (replacing the two ranges independently would leave key fragments behind)")

    # ---------------------------------------------------------------- #
    # 3. Counts, never values
    # ---------------------------------------------------------------- #
    print("\n  What a control plane is allowed to keep:\n")
    print(f"    {json.dumps(blocked.metadata)}")
    blob = json.dumps(blocked.metadata)
    for secret in ("dana@example.com", "123-45-6789", "555-123-4567"):
        assert secret not in blob, secret
    print("    ...and none of the matched values appear in it.")

    # ---------------------------------------------------------------- #
    # 4. A rule authored on the plane, enforced at the edge
    # ---------------------------------------------------------------- #
    plane_rule = guardrail_from_policy_rule(
        {
            "id": "gr_pii",
            "name": "Redact personal data",
            "guardrail_type": "output",
            "validation_mode": "blocking",
            "implementation_type": "pii",
            "config": {
                "entities": ["email", "phone", "ssn", "credit_card"],
                "backend": "regex",
                "mask_token": "[REDACTED]",
            },
            "on_error": "block",
            "action": "mask",
            "agent_ids": [],
        }
    )
    out = plane_rule.execute(CUSTOMER)
    print("\n  A console-authored rule, reconstructed and run locally:\n")
    print(
        f"    {plane_rule.name}: type={plane_rule.guardrail_type.value} origin={plane_rule.origin}"
    )
    print(f"    -> {out.action_taken}  {out.modified_data!r}")

    # ---------------------------------------------------------------- #
    # 5. A rule that cannot run says so
    # ---------------------------------------------------------------- #
    print("\n  Detection that cannot run is not detection that found nothing:\n")
    typo = rule("pii", entities=["emial"]).execute(CUSTOMER)
    print(f"    entities=['emial'] -> errored={typo.errored} passed={typo.passed}")
    print(f"      {typo.message}")
    print(
        "\n  Reporting 'no PII found' there would read as a healthy control\n"
        "  while inspecting nothing.\n"
    )


if __name__ == "__main__":
    main()
