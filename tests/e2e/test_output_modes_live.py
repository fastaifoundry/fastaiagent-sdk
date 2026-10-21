"""Structured output modes and partial streaming against real providers (1.91.0, 2G).

What only a real provider can prove:

* each provider accepts what each ``output_mode`` sends — Anthropic's native
  ``output_config.format`` (with the schema adapted to its rules, beside a
  reasoning ``effort``), OpenAI/Gemini native schemas, the prompted form;
* ``"tool"`` mode answers through ``final_result`` after real tool calls, on
  models that refuse forced tool use (Sonnet 5.5) — the tool is offered, not forced;
* ``PartialOutput`` arrives during a real stream, growing toward the answer.

Cents per run. Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_output_modes_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from pydantic import BaseModel, Field

pytestmark = pytest.mark.e2e

MODELS = [
    ("openai", "gpt-5-mini", "OPENAI_API_KEY"),
    ("anthropic", "claude-haiku-5-5", "ANTHROPIC_API_KEY"),
    ("anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY"),
    ("gemini", "gemini-3.8-flash", "GEMINI_API_KEY"),
]


class Forecast(BaseModel):
    city: str = Field(max_length=40, description="The city asked about")
    celsius: float = Field(ge=-90, le=60)
    summary: str


def _need(key: str) -> None:
    if not os.environ.get(key):
        pytest.skip(f"{key} not set")


def _weather(city: str) -> str:
    """Current weather for a city."""
    return {"rome": "24C, sunny", "oslo": "-4C, snow"}.get(city.lower(), "20C")


def _agent(provider: str, model: str, mode: str, **kw: Any) -> Any:
    from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient

    return Agent(
        name=f"forecaster-{mode}",
        system_prompt="Use the weather tool for the city asked about, then answer.",
        llm=LLMClient(provider=provider, model=model, **kw),
        tools=[FunctionTool(name="get_weather", fn=_weather)],
        output_type=Forecast,
        config=AgentConfig(output_mode=mode, max_iterations=5),
    )


@pytest.mark.parametrize("mode", ["auto", "native", "tool", "prompt"])
@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_every_mode_answers_on_every_provider(
    provider: str, model: str, key: str, mode: str
) -> None:
    _need(key)
    result = _agent(provider, model, mode).run("What's the weather in Rome?")
    assert isinstance(result.parsed, Forecast), result.output
    assert result.parsed.city.lower() == "rome" and result.parsed.celsius == 24


def test_anthropic_native_takes_a_reasoning_effort_beside_the_schema() -> None:
    _need("ANTHROPIC_API_KEY")
    result = _agent("anthropic", "claude-haiku-5-5", "native", reasoning="low").run(
        "What's the weather in Oslo?"
    )
    assert isinstance(result.parsed, Forecast) and result.parsed.celsius == -4


@pytest.mark.parametrize("mode", ["auto", "tool"])
@pytest.mark.parametrize(("provider", "model", "key"), MODELS)
def test_partial_output_streams(provider: str, model: str, key: str, mode: str) -> None:
    _need(key)
    from fastaiagent.llm.stream import PartialOutput

    agent = _agent(provider, model, mode)

    async def drain() -> list[Any]:
        return [e async for e in agent.astream("What's the weather in Rome?")]

    partials = [e.data for e in asyncio.run(drain()) if isinstance(e, PartialOutput)]
    assert partials, "no PartialOutput"
    final = partials[-1]
    assert isinstance(final, dict) and str(final.get("city", "")).lower() == "rome"
