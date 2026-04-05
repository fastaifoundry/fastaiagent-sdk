"""
Guardrails — Safety checks for input and output.

These run automatically on every agent interaction.
"""

import fastaiagent as fa
from fastaiagent.guardrail import GuardrailPosition

# ─── PII Filter (Output) ────────────────────────────────────────────────────
# Blocks any response that contains SSN, credit card, email, or phone patterns

pii_filter = fa.no_pii(position=GuardrailPosition.output)


# ─── Toxicity Check (Input) ─────────────────────────────────────────────────
# Blocks abusive or harmful input before it reaches the LLM

toxicity_check = fa.toxicity_check(position=GuardrailPosition.input)
