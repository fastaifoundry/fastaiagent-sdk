"""End-to-end gate — Pydantic-model tool arguments (OpenAI + Anthropic).

A tool whose parameter is a nested Pydantic model with an ``Enum`` field
exercises the rich schema-generation path (``$defs``/``$ref``/``enum``). The
test asserts the model populated the nested arguments correctly against that
schema — proving native function calling handles structured arguments, not
just primitives.
"""

from __future__ import annotations

import enum
import os
from typing import Any

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


def _require(provider_key: str) -> None:
    if os.environ.get(provider_key):
        return
    message = f"{provider_key} not set — skipping pydantic-args gate step"
    if os.environ.get("E2E_REQUIRED") == "1":
        pytest.fail(message)
    pytest.skip(message)


class Priority(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Ticket(BaseModel):
    title: str
    body: str
    priority: Priority


# Captures the raw arguments the model produced for the tool call.
_CAPTURED: dict[str, Any] = {}


def _build_agent(provider: str, model: str):
    from fastaiagent import Agent, FunctionTool, LLMClient

    def create_ticket(ticket: Ticket) -> str:
        """Create a support ticket.

        Args:
            ticket: the ticket to create, with title, body and priority
        """
        # Argument coercion is on by default, so ``ticket`` arrives as a fully
        # validated ``Ticket`` instance — not a raw dict. Capture for assertions.
        _CAPTURED["ticket"] = ticket
        return f"Created ticket: {ticket.title}"

    return Agent(
        name=f"{provider}-ticketing",
        system_prompt=(
            "You are a helpdesk bot. Use create_ticket to file the user's "
            "issue. Choose a sensible priority."
        ),
        llm=LLMClient(provider=provider, model=model),
        tools=[FunctionTool(name="create_ticket", fn=create_ticket)],
    )


def _assert_structured_args(result: Any) -> None:
    assert len(result.tool_calls) >= 1, "create_ticket was not called"
    assert _CAPTURED, "tool did not receive any arguments"
    ticket = _CAPTURED["ticket"]
    # Coercion contract: the tool received a validated Ticket instance.
    assert isinstance(ticket, Ticket), f"expected a coerced Ticket, got {type(ticket)}"
    assert ticket.title, "title should be populated from the request"
    assert isinstance(ticket.priority, Priority), (
        f"priority should be coerced to the Enum, got {ticket.priority!r}"
    )


def test_openai_pydantic_tool_args() -> None:
    _require("OPENAI_API_KEY")
    _CAPTURED.clear()
    agent = _build_agent("openai", "gpt-4o")
    result = agent.run(
        "My laptop won't boot after the latest update — please file a ticket."
    )
    _assert_structured_args(result)


def test_anthropic_pydantic_tool_args() -> None:
    _require("ANTHROPIC_API_KEY")
    _CAPTURED.clear()
    agent = _build_agent("anthropic", "claude-sonnet-4-6")
    result = agent.run(
        "My laptop won't boot after the latest update — please file a ticket."
    )
    _assert_structured_args(result)
