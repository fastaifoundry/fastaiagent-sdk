"""Agent factory shared by all five steps of the loop.

The system prompt is intentionally terse so the agent leans on
``lookup_order`` to answer order-status questions — making the tool
bug the source of the failure that the rest of the loop demonstrates
how to fix.
"""

from __future__ import annotations

import os

from tools import buggy_lookup_order_tool, fixed_lookup_order_tool

from fastaiagent import Agent, LLMClient

DEFAULT_MODEL = "gpt-4.1-mini"
SYSTEM_PROMPT = (
    "You are a customer support agent. To answer any question about an order, "
    "call the lookup_order tool with the order id. Reply in exactly one sentence. "
    "If the tool reports the order was not found, say so plainly — never invent "
    "details for an order you don't have data for."
)


def _llm() -> LLMClient:
    """LLMClient using OPENAI_API_KEY from the env (set via .env / zshrc)."""
    return LLMClient(
        provider="openai",
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
    )


def build_buggy_agent() -> Agent:
    """The agent we run in ``capture.py`` to reproduce the failure."""
    return Agent(
        name="support-bot",
        system_prompt=SYSTEM_PROMPT,
        llm=_llm(),
        tools=[buggy_lookup_order_tool()],
    )


def build_fixed_agent() -> Agent:
    """The agent ``verify.py`` runs against the regression dataset.

    Same shape as the buggy agent, but with the fixed tool wired in.
    Keeping the LLM/prompt identical isolates the variable — when the
    eval passes, we know the tool fix is what saved it.
    """
    return Agent(
        name="support-bot",
        system_prompt=SYSTEM_PROMPT,
        llm=_llm(),
        tools=[fixed_lookup_order_tool()],
    )
