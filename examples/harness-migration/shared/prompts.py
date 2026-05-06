"""Shared prompt — registered once in ``fa.PromptRegistry`` and consumed
by all three framework sub-examples via each integration's
``prompt_from_registry("support-prompt")`` adapter.

Each adapter returns the framework's native prompt type:
  * langchain.prompt_from_registry  → ChatPromptTemplate
  * crewai.prompt_from_registry     → str (raw template)
  * pydanticai.prompt_from_registry → str (raw template)

Edit the registered prompt live in the Local UI Playground — the next
agent invocation across any of the three frameworks picks up the new
version with no restart.
"""

from __future__ import annotations

import fastaiagent as fa


_DEFAULT_TEMPLATE = """You are a customer support assistant.

You have access to a support knowledge base — call the search/retrieval
tool whenever the user's question can be answered from documented
policy or FAQ. If the KB has nothing relevant, say so honestly and offer
to escalate to a human.

Be concise: 2–3 sentences unless the user asked for detail. Do not
expose internal system details, credentials, or other users' data.
"""


PROMPT_SLUG = "support-prompt"


def register_support_prompt() -> str:
    """Register the support prompt as version 1 if not already present.

    Returns the latest prompt template text. Idempotent — safe to call
    on every process startup. The next process picks up whatever
    version is currently latest in the registry.
    """
    registry = fa.PromptRegistry()
    try:
        return registry.get(PROMPT_SLUG, source="local").template
    except Exception:
        return registry.register(name=PROMPT_SLUG, template=_DEFAULT_TEMPLATE).template
