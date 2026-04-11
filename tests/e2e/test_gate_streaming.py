"""End-to-end quality gate — streaming path (``agent.astream``).

The main quality gate exercises ``agent.run`` exhaustively. This gate
proves the streaming code path produces real tokens against a real LLM,
and that tool calls still fire inside the stream loop. Without this,
streaming is a feature marketed on the box that could rot silently.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


def _lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: shipped on 2026-04-01"


class TestStreamingGate:
    """Agent streaming path — token delivery + tool-call + trace integrity."""

    def test_01_stream_emits_text_deltas(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient
        from fastaiagent.llm.stream import TextDelta

        agent = Agent(
            name="streaming-gate",
            system_prompt="Reply with exactly: The quick brown fox jumps.",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )

        async def _collect() -> list[Any]:
            events: list[Any] = []
            async for event in agent.astream("Say the sentence."):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) >= 1, (
            f"astream emitted no TextDelta events — streaming is broken. "
            f"Events received: {[type(e).__name__ for e in events]}"
        )
        full_text = "".join(d.text for d in text_deltas)
        assert len(full_text) > 0, "TextDeltas were empty"
        assert "fox" in full_text.lower() or "quick" in full_text.lower(), (
            f"streamed text does not match system prompt's instruction: {full_text!r}"
        )
        gate_state["stream_text"] = full_text
        gate_state["stream_events"] = events

    def test_02_stream_collects_via_sync_helper(
        self, gate_state: dict[str, Any]
    ) -> None:
        require_env()
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="streaming-gate-sync",
            system_prompt="Reply with exactly: The quick brown fox jumps.",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
        )
        # Agent.stream() is the sync helper that collects a stream into
        # an AgentResult — still exercises astream under the hood.
        result = agent.stream("Say the sentence.")
        assert result.output, "Agent.stream() returned empty output"
        assert result.latency_ms > 0

    def test_03_stream_with_tool_call(self, gate_state: dict[str, Any]) -> None:
        require_env()
        from fastaiagent import Agent, FunctionTool, LLMClient
        from fastaiagent.llm.stream import TextDelta, ToolCallEnd, ToolCallStart

        agent = Agent(
            name="streaming-gate-with-tool",
            system_prompt=(
                "You are a support agent. "
                "Use the lookup_order tool when asked about an order."
            ),
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="lookup_order", fn=_lookup_order)],
        )

        async def _collect() -> list[Any]:
            events: list[Any] = []
            async for event in agent.astream("What is the status of order ORD-200?"):
                events.append(event)
            return events

        events = asyncio.run(_collect())
        assert len(events) > 0, "astream produced zero events — loop broken"

        # Must have at least one tool-call lifecycle pair.
        tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
        assert len(tool_starts) >= 1, (
            f"No ToolCallStart events in stream — tool-call streaming broken. "
            f"Event types: {[type(e).__name__ for e in events]}"
        )
        assert len(tool_ends) >= 1, "No ToolCallEnd events — argument parsing broken"
        assert tool_starts[0].tool_name == "lookup_order"
        assert tool_ends[0].arguments.get("order_id"), (
            "ToolCallEnd did not carry parsed arguments"
        )

        # And final assistant text must be non-empty.
        text_deltas = [e for e in events if isinstance(e, TextDelta)]
        assert len(text_deltas) >= 1, "Stream ended without any final text"
