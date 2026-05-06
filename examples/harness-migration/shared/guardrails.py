"""Shared guardrails — wrap any framework agent with the same set.

Each integration's ``with_guardrails()`` accepts the same FastAIAgent
``Guardrail`` type used by native ``fa.Agent``. So an output-side
PII filter you trust against your fastaiagent agents drops in unchanged
on a LangGraph / CrewAI / PydanticAI agent.
"""

from __future__ import annotations

import fastaiagent as fa
from fastaiagent.guardrail import GuardrailPosition

input_guardrails: list[fa.Guardrail] = [
    fa.toxicity_check(position=GuardrailPosition.input),
]

output_guardrails: list[fa.Guardrail] = [
    fa.no_pii(position=GuardrailPosition.output),
]
