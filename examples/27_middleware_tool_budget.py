"""Example 27: Agent middleware — tool budget, trim history, PII redaction.

Demonstrates the built-in middleware:
  1. ToolBudget       — stop a run after N tool invocations
  2. TrimLongMessages — cap how many messages the LLM sees
  3. RedactPII        — scrub PII from outbound prompts and inbound responses

For offline demonstration without an API key, a MockLLM fallback is shown at
the bottom — useful for CI and test harnesses.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/27_middleware_tool_budget.py
"""

from __future__ import annotations

import asyncio
import os

from fastaiagent import (
    Agent,
    FunctionTool,
    LLMClient,
    RedactPII,
    ToolBudget,
    TrimLongMessages,
)

# --- Tools ---


def lookup(query: str) -> str:
    """Pretend to look something up."""
    return f"Result for {query!r}: matched 3 documents."


search_tool = FunctionTool(
    name="lookup",
    fn=lookup,
    description="Look up information for a query",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


# --- Real LLM demo ---


def run_with_real_llm() -> None:
    agent = Agent(
        name="controlled",
        system_prompt=(
            "You help users find information. You MAY call lookup() multiple times. "
            "Stop once you have enough to answer."
        ),
        llm=LLMClient(provider="openai", model="gpt-4.1"),
        tools=[search_tool],
        middleware=[
            TrimLongMessages(keep_last=30),
            RedactPII(),
            ToolBudget(max_calls=3, message="Tool budget of 3 calls reached."),
        ],
    )
    result = agent.run("My email is alice@example.com. Find 5 things about octopuses.")
    print("Output:", result.output)
    print("Tool calls made:", len(result.tool_calls))


# --- Offline demo with MockLLMClient ---


def run_with_mock_llm() -> None:
    """Offline demo of ToolBudget short-circuiting.

    Simulates an LLM that keeps requesting lookup() forever. ToolBudget
    steps in at the third call.
    """
    from fastaiagent.llm.client import LLMResponse
    from fastaiagent.llm.message import ToolCall
    from tests.conftest import MockLLMClient  # type: ignore[import-not-found]

    # 5 tool-call turns followed by a would-be-final response. The budget
    # should stop us after the third tool call.
    llm = MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id=f"call_{i}", name="lookup", arguments={"query": f"q{i}"})
                ],
                finish_reason="tool_calls",
            )
            for i in range(5)
        ]
        + [LLMResponse(content="Never reached.", finish_reason="stop")]
    )

    agent = Agent(
        name="offline",
        llm=llm,
        tools=[search_tool],
        middleware=[
            ToolBudget(max_calls=3, message="Tool budget of 3 calls reached."),
        ],
    )
    result = asyncio.run(agent.arun("Find anything", trace=False))
    print("Output:", result.output)
    assert "budget" in result.output.lower(), "expected ToolBudget to stop the run"
    print("Offline demo passed — ToolBudget short-circuited the run.")


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        run_with_real_llm()
    else:
        print("OPENAI_API_KEY not set; running offline demo with MockLLMClient.\n")
        run_with_mock_llm()
