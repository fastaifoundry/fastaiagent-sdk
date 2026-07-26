"""Example 91: Guardrail fail policy — `on_error` (fail-open vs fail-closed).

A model-judged guardrail (toxicity/groundedness/moderation/llm_judge) depends on
an LLM call that can *fail* — a timeout, a 5xx, an unparseable response. That's
different from the check running and returning a verdict, and you choose what it
means with `on_error`:

    on_error="block"   # fail closed — an errored check blocks (default)
    on_error="allow"   # fail open  — an errored check passes through

Either way the failure is never silent: the result is flagged `errored=True` (and
the Local UI / connected plane log an `errored` outcome), so a degraded check is
never mistaken for a genuine pass.

This example is deterministic and needs no API key: a guardrail whose check
raises stands in for a model-judged check whose LLM call failed.
"""

from fastaiagent.guardrail import GuardrailPosition
from fastaiagent.guardrail.guardrail import Guardrail


def flaky_check(_text: str) -> bool:
    """Stand-in for a model-judged detector whose LLM call just failed."""
    raise RuntimeError("classifier unavailable (simulated outage)")


# Same check, two fail policies.
fail_open = Guardrail(
    name="toxicity_like",
    fn=flaky_check,
    position=GuardrailPosition.output,
    on_error="allow",  # availability over strictness
)
fail_closed = Guardrail(
    name="toxicity_like",
    fn=flaky_check,
    position=GuardrailPosition.output,
    on_error="block",  # strictness over availability
)

if __name__ == "__main__":
    # Fail open: the errored check does NOT block; the run continues, but the
    # result carries errored=True so you can see the guardrail degraded.
    r_open = fail_open.execute("some model output")
    print(f"on_error='allow' -> passed={r_open.passed}, errored={r_open.errored}")
    print(f"                    message: {r_open.message}")
    print(f"                    metadata: {r_open.metadata}")

    # Fail closed: the errored check blocks — passed=False (an Agent would raise
    # GuardrailBlockedError and halt the run).
    r_closed = fail_closed.execute("some model output")
    print(f"on_error='block' -> passed={r_closed.passed}, errored={r_closed.errored}")

    # Deterministic guardrails (no_pii, no_secrets, json_valid, allowed_domains)
    # can't make a fallible call, so on_error doesn't apply — they're reliable
    # hard blocks. The built-in model-judged factories take on_error too, e.g.:
    #   toxicity_check(mode="llm", on_error="allow")
    #   grounded(reference, on_error="block")
    #   responsible_ai(on_error="block", ...)   # override the whole bundle
    print("\nModel-judged built-ins accept on_error; deterministic ones don't need it.")
