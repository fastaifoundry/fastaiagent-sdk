"""Example 113: bound a run, stop it, and pick up where it left off (1.91.0).

    AgentConfig(timeout_seconds=, max_total_tokens=, max_cost_usd=)
        A wall clock and two budgets. The budgets end the run cleanly
        (``stop_reason``); the timeout raises ``AgentTimeoutError``.
    agent.cancel(execution_id)
        Stops a run in progress, from any thread: ``status="cancelled"``.
    chain.cancel(execution_id)
        The same for a Chain run (1.93.0), and ``swarm.cancel`` for a Swarm.
    result.messages
        The run's conversation, ready for the next run's ``messages=``.
    agent.run(..., output_type=T)
        A typed ``AgentResult[T]``.
    RunContext.run
        What a tool can see of the run calling it.

It runs on Gemini 3.8 Flash when GEMINI_API_KEY is set, else OpenAI gpt-5-mini.

Usage:
    zsh -lc 'python examples/113_run_limits_and_cancel.py'
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel

from fastaiagent import (
    Agent,
    AgentConfig,
    AgentTimeoutError,
    FunctionTool,
    LLMClient,
    RunContext,
)


class Forecast(BaseModel):
    city: str
    celsius: float


def get_weather(city: str, ctx: RunContext[str]) -> str:
    """Current weather for a city."""
    run = ctx.run
    if run is not None:
        print(f"    [tool] turn {run.turn}, spent so far ${run.usage.cost:.6f} on {run.model}")
    return {"paris": "18C, cloudy", "rome": "24C, sunny"}.get(city.lower(), "20C")


def llm() -> LLMClient:
    if os.environ.get("GEMINI_API_KEY"):
        return LLMClient(provider="gemini", model="gemini-3.8-flash")
    if os.environ.get("OPENAI_API_KEY"):
        return LLMClient(provider="openai", model="gpt-5-mini")
    raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY.")


def budgets() -> None:
    print("── a cost budget ends the run cleanly")
    agent = Agent(
        name="weather",
        system_prompt="Use the weather tool for each city, one call at a time.",
        llm=llm(),
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
        config=AgentConfig(max_cost_usd=0.000001),
    )
    result = agent.run("Paris or Rome: which is warmer?", context=RunContext("demo"))
    print(f"  stop_reason={result.stop_reason!r}, spent ${result.cost:.6f}")


def timeout() -> None:
    print("── a timeout stops a long generation mid-call")
    agent = Agent(name="essayist", llm=llm(), config=AgentConfig(timeout_seconds=1.5))
    try:
        agent.run("Write a 3000-word essay on the printing press.")
    except AgentTimeoutError as error:
        print(f"  AgentTimeoutError: {error}")


def cancel() -> None:
    print("── cancel a run in progress")
    agent = Agent(name="essayist", llm=llm())

    async def main() -> None:
        task = asyncio.create_task(
            agent.arun("Write a 3000-word essay on the printing press.", execution_id="job-7")
        )
        await asyncio.sleep(1)
        print(f"  cancel('job-7') -> {agent.cancel('job-7')}")
        result = await task
        print(f"  status={result.status!r}, stop_reason={result.stop_reason!r}")

    asyncio.run(main())


def cancel_a_chain() -> None:
    print("── cancel a chain run (1.93.0; Swarm.cancel works the same way)")
    from fastaiagent import Chain

    chain = Chain("report", checkpoint_enabled=False)
    chain.add_node("draft", agent=Agent(name="essayist", llm=llm()))

    async def main() -> None:
        task = asyncio.create_task(
            chain.aexecute({"input": "Write a 3000-word essay on canals."}, execution_id="job-8")
        )
        await asyncio.sleep(1)
        print(f"  chain.cancel('job-8') -> {chain.cancel('job-8')}")
        print(f"  status={(await task).status!r}")

    asyncio.run(main())


def continue_and_type() -> None:
    print("── messages carry the conversation; output_type types the result")
    agent = Agent(
        name="chat",
        system_prompt="Use the weather tool when asked about weather. Be brief.",
        llm=llm(),
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
    )
    first = agent.run("What's the weather in Rome?", context=RunContext("demo"))
    print(f"  first run: {[m.role.value for m in first.messages]}")
    second = agent.run(
        "Give that as JSON with city and celsius.",
        messages=first.messages,
        output_type=Forecast,
        context=RunContext("demo"),
    )
    forecast = second.parsed  # AgentResult[Forecast]: typed for your editor too
    print(f"  parsed: {forecast!r}")


if __name__ == "__main__":
    budgets()
    timeout()
    cancel()
    cancel_a_chain()
    continue_and_type()
