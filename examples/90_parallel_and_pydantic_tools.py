"""Example 90 — Native tool calling: parallel, typed, and resilient.

Native tool-calling upgrades (v1.41.0), shown end to end:

  1. **Parallel tools** — with ``AgentConfig(parallel_tools=True)`` the tool
     calls the model emits in a single turn run concurrently. Each ``fetch``
     call sleeps ~0.6s; asked about three cities at once, the turn finishes in
     ~0.6s instead of ~1.8s. A live-concurrency counter proves the overlap.

  2. **Pydantic tool args (auto-coerced)** — ``file_ticket(ticket: Ticket)``
     takes a nested Pydantic model with an ``Enum`` field. The SDK generates a
     full JSON Schema from the type hint *and* validates + coerces the model's
     arguments, so the function receives a ready-to-use ``Ticket`` instance.

  3. **Execution policy** — ``@tool(timeout=..., max_retries=...)`` adds a
     per-call timeout and automatic retries with one line of keyword args.
     (An optional ``output_type=`` also validates/coerces the *return* value —
     handy when a tool returns a loose type; see docs/tools/function-tools.md.)

Prereqs:
    pip install 'fastaiagent[openai]'
    export OPENAI_API_KEY=sk-...

Run:
    python examples/90_parallel_and_pydantic_tools.py
"""

from __future__ import annotations

import asyncio
import enum
import os
import sys
import threading
import time

from pydantic import BaseModel

from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient, tool

# ─── Live concurrency tracker (proves the tools actually overlap) ───────────


class _Concurrency:
    def __init__(self) -> None:
        self._live = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        with self._lock:
            self._live += 1
            self.peak = max(self.peak, self._live)

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self._live -= 1


CONCURRENCY = _Concurrency()
_WEATHER = {"paris": "18°C, cloudy", "tokyo": "26°C, clear", "cairo": "34°C, sunny"}


# ─── 1) A slow async tool, called several times in one turn ─────────────────


async def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: the city name
    """
    with CONCURRENCY:
        await asyncio.sleep(0.6)  # simulate a slow API so overlap is visible
        return _WEATHER.get(city.strip().lower(), f"no data for {city}")


# ─── 2) A tool whose argument is a nested Pydantic model + Enum ─────────────


class Priority(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Ticket(BaseModel):
    title: str
    body: str
    priority: Priority


def file_ticket(ticket: Ticket) -> str:
    """File a support ticket.

    Args:
        ticket: the ticket to create, with title, body and priority
    """
    # `ticket` arrives already validated and coerced to a Ticket instance —
    # argument coercion is on by default, so no dict-handling boilerplate.
    return f"Filed [{ticket.priority.value.upper()}] {ticket.title!r}"


# ─── 3) Execution policy: timeout + retry, in one line of kwargs ────────────


_ATTEMPTS = {"n": 0}


@tool(name="fx_rate", timeout=2.0, max_retries=2)
def fx_rate(base: str, quote: str) -> float:
    """Get the FX rate between two currencies.

    Args:
        base: base currency code, e.g. USD
        quote: quote currency code, e.g. EUR
    """
    # Flaky on the first call to demonstrate automatic retries.
    _ATTEMPTS["n"] += 1
    if _ATTEMPTS["n"] < 2:
        raise RuntimeError("upstream FX provider hiccup")
    return 0.92


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        sys.exit(0)

    weather_tool = FunctionTool(name="get_weather", fn=get_weather)

    # Show the rich schema generated from the Pydantic type hint.
    ticket_tool = FunctionTool(name="file_ticket", fn=file_ticket)
    print("file_ticket schema $defs:", list(ticket_tool.parameters.get("$defs", {})))

    agent = Agent(
        name="parallel-demo",
        system_prompt=(
            "You are a helpful assistant. When asked about multiple cities, "
            "call get_weather for every city. Use file_ticket to log issues."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o"),
        tools=[weather_tool, ticket_tool, fx_rate],
        config=AgentConfig(parallel_tools=True, max_parallel_tools=4),
    )

    # 1) Parallel tool calls.
    start = time.perf_counter()
    weather = agent.run("What's the weather in Paris, Tokyo, and Cairo right now?")
    elapsed = time.perf_counter() - start
    print(f"\nWeather answer: {weather.output}")
    print(f"tool calls: {len(weather.tool_calls)}  |  elapsed: {elapsed:.1f}s  "
          f"|  peak concurrency: {CONCURRENCY.peak}")

    # 2) Pydantic-typed structured arguments (auto-coerced to a Ticket).
    ticket = agent.run(
        "My laptop won't boot after the update — file a high-priority ticket."
    )
    print(f"\nTicket answer: {ticket.output}")

    # 3) Execution policy — the fx_rate tool fails once, then the SDK
    #    automatically retries and succeeds (see tool attempts below).
    fx = agent.run("What's the USD to EUR exchange rate?")
    print(f"\nFX answer: {fx.output}  (tool attempts: {_ATTEMPTS['n']})")


if __name__ == "__main__":
    main()
