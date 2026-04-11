"""End-to-end quality gate — structured output (``output_type=...``).

Agents can be constructed with ``output_type=<PydanticModel>``, in which
case the SDK feeds a ``response_format`` hint to the provider and parses
the LLM response into an instance of the declared model. The parsed
object shows up on ``AgentResult.parsed``. This gate proves the happy
path against a real LLM end-to-end — not just unit coverage of the
parser.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


class OrderStatus(BaseModel):
    """Structured response describing an order lookup result."""

    order_id: str = Field(description="The order identifier")
    status: str = Field(description="Status: shipped, processing, cancelled, or unknown")
    summary: str = Field(description="One-sentence human summary")


class TestStructuredOutputGate:
    """Agent with output_type returns a populated Pydantic instance on .parsed."""

    def test_01_parsed_is_populated(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="structured-output-gate",
            system_prompt=(
                "You are a support bot. When asked about an order, respond "
                "with the order_id, a status (one of: shipped, processing, "
                "cancelled, unknown), and a one-sentence summary."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            output_type=OrderStatus,
        )
        result = agent.run(
            "Order ORD-500 was shipped yesterday via UPS. Give me a structured summary."
        )

        assert result.output, "agent.run returned empty output"
        assert result.parsed is not None, (
            f"AgentResult.parsed is None — structured output parsing failed. "
            f"Raw output: {result.output!r}"
        )
        assert isinstance(result.parsed, OrderStatus), (
            f"Parsed value is not an OrderStatus instance: {type(result.parsed)}"
        )
        parsed: OrderStatus = result.parsed
        assert parsed.order_id, "parsed.order_id is empty"
        assert "ord-500" in parsed.order_id.lower() or "500" in parsed.order_id
        assert parsed.status.lower() in {"shipped", "processing", "cancelled", "unknown"}, (
            f"parsed.status is not in the declared enum: {parsed.status!r}"
        )
        assert parsed.summary, "parsed.summary is empty"
        gate_state["structured_result"] = result

    def test_02_parsed_none_on_unparseable_output(
        self, gate_state: dict[str, Any]
    ) -> None:
        """Agents without output_type have .parsed == None (sanity check)."""
        require_env()
        from fastaiagent import Agent, LLMClient

        plain_agent = Agent(
            name="structured-output-gate-plain",
            system_prompt="You are helpful. Reply briefly in plain prose.",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )
        result = plain_agent.run("Say hi.")
        assert result.output, "plain agent returned empty output"
        assert result.parsed is None, (
            "Agent without output_type returned a non-None .parsed "
            "— parser should not attempt to parse unstructured output"
        )
