"""Shared FastAIAgent assets used by all three framework sub-examples.

The point of this template is that the SAME ``LocalKB`` and the SAME
registered ``Prompt`` and the SAME guardrails work across LangChain,
CrewAI, and PydanticAI agents — only the tiny adapter call differs.
"""

from shared.kb import support_kb
from shared.prompts import register_support_prompt
from shared.guardrails import input_guardrails, output_guardrails

__all__ = [
    "support_kb",
    "register_support_prompt",
    "input_guardrails",
    "output_guardrails",
]
