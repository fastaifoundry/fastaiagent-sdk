"""Run limits, cancel, typed results and messages against real providers (1.91.0, 2H-2J).

What only a real provider can prove:

* a cancel or a timeout lands **inside** a real model call that is still
  generating (a long essay), not after it;
* the cost cap reads real, priced usage and stops a real tool loop;
* ``result.messages`` is a valid history for the provider: a second run sent with
  it is accepted — for Claude with its thinking blocks, which the API verifies;
* ``output_type=`` per call returns the typed value from a real model.

Cents per run. Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_run_limits_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

LONG = "Write a 3000-word essay on the history of the printing press, with headings."

MODELS = [
    ("openai", "gpt-5-mini", "OPENAI_API_KEY"),
    ("anthropic", "claude-haiku-5-5", "ANTHROPIC_API_KEY"),
    ("gemini", "gemini-3.8-flash", "GEMINI_API_KEY"),
]


def _need(key: str) -> None:
    if not os.environ.get(key):
        pytest.skip(f"{key} not set")


def _weather(city: str) -> str:
    """Current weather for a city."""
    return {"paris": "18C, cloudy", "rome": "24C, sunny"}.get(city.lower(), "20C")


class Capital(BaseModel):
    country: str
    capital: str


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_cancel_lands_inside_a_real_model_call(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import Agent, LLMClient

    agent = Agent(name="essayist", llm=LLMClient(provider=provider, model=model))

    async def main() -> tuple[Any, float]:
        task = asyncio.ensure_future(agent.arun(LONG, execution_id=f"live-cx-{provider}"))
        await asyncio.sleep(1.0)
        started = time.monotonic()
        assert agent.cancel(f"live-cx-{provider}") is True
        result = await task
        return result, time.monotonic() - started

    result, waited = asyncio.run(main())
    assert result.status == "cancelled"
    assert waited < 2, f"the cancel waited {waited:.1f}s for the model to finish"


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_a_timeout_lands_inside_a_real_model_call(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import Agent, AgentConfig, AgentTimeoutError, LLMClient

    agent = Agent(
        name="essayist",
        llm=LLMClient(provider=provider, model=model),
        config=AgentConfig(timeout_seconds=1.5),
    )
    started = time.monotonic()
    with pytest.raises(AgentTimeoutError):
        agent.run(LONG)
    assert time.monotonic() - started < 4


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_a_cost_cap_stops_a_real_tool_loop(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient

    agent = Agent(
        name="weather",
        system_prompt="Use the weather tool for each city, one call at a time.",
        llm=LLMClient(provider=provider, model=model),
        tools=[FunctionTool(name="get_weather", fn=_weather)],
        # Below one call's price on every model here: the run stops after its
        # first call instead of finishing the loop.
        config=AgentConfig(max_cost_usd=0.000001),
    )
    result = agent.run("Is it warmer in Paris or in Rome? Check both.")
    assert result.stop_reason == "max_cost_usd", result
    assert result.cost_known and result.cost > 0.000001


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_messages_continue_a_real_conversation(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import Agent, FunctionTool, LLMClient

    agent = Agent(
        name="chat",
        system_prompt="Use the weather tool when asked about weather. Be brief.",
        llm=LLMClient(provider=provider, model=model, reasoning="low"),
        tools=[FunctionTool(name="get_weather", fn=_weather)],
    )
    first = agent.run("What's the weather in Rome? My name is Priya.")
    assert [m.role.value for m in first.messages][:3] == ["user", "assistant", "tool"]
    second = agent.run("What is my name, and which city did I ask about?", messages=first.messages)
    text = second.output.lower()
    assert "priya" in text and "rome" in text


@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_a_per_call_output_type_is_typed(provider: str, model: str, key: str) -> None:
    _need(key)
    from fastaiagent import Agent, LLMClient

    agent = Agent(name="geo", llm=LLMClient(provider=provider, model=model))
    result = agent.run("The capital of France, as JSON.", output_type=Capital)
    assert isinstance(result.parsed, Capital)
    assert result.parsed.capital.lower() == "paris"
