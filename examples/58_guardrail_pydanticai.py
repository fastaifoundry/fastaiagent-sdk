"""Example 58 — Wrap a PydanticAI agent with FastAIAgent guardrails.

Demonstrates the block-only guardrail contract: a failing blocking
guardrail (a) writes a ``guardrail_events`` row that the Local UI's
Guardrail Events page will surface, and (b) raises
``GuardrailBlocked`` so the caller knows the run was halted. There's
no redaction — that's by design (decision A in the harness spec).

Run:
    pip install "fastaiagent[pydanticai,ui]"
    OPENAI_API_KEY=sk-... FASTAIAGENT_UI_ENABLED=1 \\
        python examples/58_guardrail_pydanticai.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set — skipping example.")
        return 0

    # Guardrail events only persist when the UI is enabled. Set the
    # env var here so this example is self-contained — in real usage
    # users typically set it in their shell.
    os.environ.setdefault("FASTAIAGENT_UI_ENABLED", "1")

    try:
        from pydantic_ai import Agent
    except ImportError:
        print(
            'pydantic-ai is not installed. Install with: '
            'pip install "fastaiagent[pydanticai]"'
        )
        return 0

    from fastaiagent.guardrail.builtins import no_pii
    from fastaiagent.integrations import pydanticai as pa
    from fastaiagent.integrations._registry import GuardrailBlocked

    pa.enable()

    agent = Agent("openai:gpt-4o-mini", system_prompt="be terse")
    guarded = pa.with_guardrails(
        agent,
        name="example-58-guarded-agent",
        # ``no_pii`` is positioned for output by default; we attach it
        # to the input side here so the guardrail catches PII *before*
        # the LLM ever sees it.
        input_guardrails=[no_pii()],
    )

    # First call: should succeed.
    print("\n[1] benign input — should pass:")
    print("  output:", guarded.run_sync("What colour is the sky?").output)

    # Second call: should raise.
    print("\n[2] PII input — should raise GuardrailBlocked:")
    try:
        guarded.run_sync("My SSN is 123-45-6789, please summarise it.")
    except GuardrailBlocked as e:
        print(f"  blocked as expected: {e}")
    else:
        print("  unexpected: no GuardrailBlocked raised")

    print(
        "\nVisit http://127.0.0.1:7842/guardrail-events to see the logged event."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
