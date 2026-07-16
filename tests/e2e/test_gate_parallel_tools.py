"""End-to-end gate — opt-in parallel tool execution (OpenAI + Anthropic).

Verifies that when ``AgentConfig(parallel_tools=True)`` is set and the model
emits multiple tool calls in a single turn, the executor runs them
concurrently and still returns a correct, ordered result.

Concurrency is asserted precisely: each tool call records its live-concurrency
via a shared counter. The overlap assertion only fires when the model actually
placed >=2 calls in the *same* iteration (real parallel tool calling) — if a
model chooses to split calls across turns, parallelism legitimately doesn't
apply and the test doesn't flake on it. The always-on assertion is the weaker
correctness contract: both cities were looked up and appear in the answer.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any

import pytest

pytestmark = pytest.mark.e2e


def _require(provider_key: str) -> None:
    if os.environ.get(provider_key):
        return
    message = f"{provider_key} not set — skipping parallel-tools gate step"
    if os.environ.get("E2E_REQUIRED") == "1":
        pytest.fail(message)
    pytest.skip(message)


class _ConcurrencyTracker:
    """Records peak simultaneous tool executions across the whole run."""

    def __init__(self) -> None:
        self._live = 0
        self._peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self._live += 1
            self._peak = max(self._peak, self._live)

    def exit(self) -> None:
        with self._lock:
            self._live -= 1

    @property
    def peak(self) -> int:
        return self._peak


def _build_agent(provider: str, model: str, tracker: _ConcurrencyTracker):
    from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient

    weather = {"paris": "18C and cloudy", "tokyo": "26C and clear"}

    async def get_weather(city: str) -> str:
        """Get the current weather for a city.

        Args:
            city: the city name
        """
        tracker.enter()
        try:
            await asyncio.sleep(0.6)  # make concurrency observable
            return weather.get(city.strip().lower(), f"no data for {city}")
        finally:
            tracker.exit()

    return Agent(
        name=f"{provider}-parallel-weather",
        system_prompt=(
            "You are a weather assistant. When asked about multiple cities, "
            "call get_weather for every city. Then summarize all results."
        ),
        llm=LLMClient(provider=provider, model=model),
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
        config=AgentConfig(parallel_tools=True, max_parallel_tools=4),
    )


def _assert_parallel_run(result: Any, tracker: _ConcurrencyTracker) -> None:
    lower = result.output.lower()
    assert "paris" in lower and "tokyo" in lower, (
        f"both cities should appear in the answer: {result.output!r}"
    )
    assert len(result.tool_calls) >= 2, (
        f"expected >=2 tool calls, got {result.tool_calls!r}"
    )

    # Group tool calls by the turn they were issued in.
    by_iter: dict[int, int] = {}
    for c in result.tool_calls:
        by_iter[c["iteration"]] = by_iter.get(c["iteration"], 0) + 1
    max_in_one_turn = max(by_iter.values())

    if max_in_one_turn >= 2:
        # The model issued parallel tool calls — our executor must have
        # overlapped them.
        assert tracker.peak >= 2, (
            f"model issued {max_in_one_turn} calls in one turn but peak "
            f"concurrency was {tracker.peak} — parallel execution not engaged"
        )


def test_openai_parallel_tools() -> None:
    _require("OPENAI_API_KEY")
    tracker = _ConcurrencyTracker()
    agent = _build_agent("openai", "gpt-4o", tracker)
    result = agent.run("What's the weather in Paris and Tokyo right now?")
    _assert_parallel_run(result, tracker)


def test_anthropic_parallel_tools() -> None:
    _require("ANTHROPIC_API_KEY")
    tracker = _ConcurrencyTracker()
    agent = _build_agent("anthropic", "claude-sonnet-4-6", tracker)
    result = agent.run("What's the weather in Paris and Tokyo right now?")
    _assert_parallel_run(result, tracker)
