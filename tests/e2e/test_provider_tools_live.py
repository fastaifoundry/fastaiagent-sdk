"""OpenAI's Responses API and provider-run tools against the real providers
(1.91.0, 2C + 2D).

What only a real provider can prove:

* ``gpt-6-luna`` takes tools with reasoning on through ``api="responses"`` — the
  call Chat Completions refuses — and the reasoning items the SDK sends back
  across the tool loop are accepted, run and ``astream``;
* each provider accepts the SDK's shape for ``web_search()`` / ``code_execution()``
  beside a function tool, and the turn the SDK sends back verbatim is accepted —
  after a round trip through a checkpoint's JSON too (Anthropic's encrypted
  results and Gemini's thought signatures are verified by the provider);
* each call is recorded, with something a guardrail can judge, and priced.

Gemini 3.8 Flash first (the recommended Gemini), Claude 5.5, GPT-5.1 and
gpt-6-luna. Cents per run. Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_provider_tools_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from fastaiagent import Agent, FunctionTool, LLMClient, SQLiteCheckpointer
from fastaiagent._internal.errors import GuardrailBlockedError, LLMProviderError
from fastaiagent.chain.interrupt import Resume, interrupt
from fastaiagent.guardrail import Guardrail, GuardrailPosition, GuardrailResult, GuardrailType
from fastaiagent.llm.stream import TextDelta
from fastaiagent.tool import code_execution, web_search

pytestmark = pytest.mark.e2e

QUESTION = (
    "Search the web for the latest stable Python release, and get the weather in Paris "
    "with the get_weather tool. Answer in two short sentences."
)


def _need(*keys: str) -> None:
    for key in keys:
        if not os.environ.get(key):
            pytest.skip(f"{key} not set")


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"paris": "18C, cloudy"}.get(city.lower(), "20C")


WEATHER = FunctionTool(name="get_weather", fn=get_weather)

#: (provider, model, api, key) — Gemini 3.5+ first.
SEARCHERS = [
    ("gemini", "gemini-3.8-flash", "chat", "GEMINI_API_KEY"),
    ("anthropic", "claude-sonnet-5-5", "chat", "ANTHROPIC_API_KEY"),
    ("anthropic", "claude-haiku-5-5", "chat", "ANTHROPIC_API_KEY"),
    ("openai", "gpt-5.1", "responses", "OPENAI_API_KEY"),
    ("openai", "gpt-6-luna", "responses", "OPENAI_API_KEY"),
]


def _llm(provider: str, model: str, api: str, **kw: Any) -> LLMClient:
    return LLMClient(provider=provider, model=model, api=api, max_retries=2, **kw)


def _assert_searched(result: Any) -> None:
    names = [c["tool_name"] for c in result.tool_calls]
    assert "web_search" in names, names
    assert "get_weather" in names, names
    (search, *_) = [c for c in result.tool_calls if c["tool_name"] == "web_search"]
    assert search.get("output", "").strip(), search  # something a guardrail can judge
    assert "18" in result.output


# ── the Responses API ───────────────────────────────────────────────────────


def test_gpt_6_luna_takes_tools_with_reasoning_on_responses() -> None:
    _need("OPENAI_API_KEY")
    agent = Agent(
        name="w",
        llm=_llm("openai", "gpt-6-luna", "responses", reasoning="medium"),
        tools=[WEATHER],
    )
    result = agent.run("What's the weather in Paris? Use the tool.")
    assert [c["tool_name"] for c in result.tool_calls] == ["get_weather"]
    assert "18" in result.output
    assert result.cost > 0


def test_reasoning_items_go_back_and_are_accepted() -> None:
    _need("OPENAI_API_KEY")
    agent = Agent(
        name="w",
        llm=_llm("openai", "gpt-5.1", "responses", reasoning="high"),
        tools=[WEATHER],
    )
    puzzle = (
        "A bat and a ball cost $1.10; the bat costs $1.00 more than the ball. Work out the "
        "ball's price in cents, then call get_weather once for Paris, then answer both."
    )
    result = agent.run(puzzle)
    assert "get_weather" in [c["tool_name"] for c in result.tool_calls]
    assert "5" in result.output and "18" in result.output


def test_responses_stream_tool_loop() -> None:
    _need("OPENAI_API_KEY")
    agent = Agent(
        name="w", llm=_llm("openai", "gpt-6-luna", "responses", reasoning="low"), tools=[WEATHER]
    )

    async def go() -> str:
        return "".join(
            [
                e.text
                async for e in agent.astream("Weather in Paris? Use the tool.")
                if isinstance(e, TextDelta)
            ]
        )

    assert "18" in asyncio.run(go())


def test_chat_completions_still_refuses_it_and_says_how() -> None:
    _need("OPENAI_API_KEY")
    agent = Agent(
        name="w", llm=_llm("openai", "gpt-6-luna", "chat", reasoning="medium"), tools=[WEATHER]
    )
    with pytest.raises(LLMProviderError, match=r"api='responses'"):
        agent.run("Weather in Paris?")


# ── web_search() beside a function tool ─────────────────────────────────────


@pytest.mark.parametrize(("provider", "model", "api", "key"), SEARCHERS)
def test_web_search_beside_a_function_tool(provider: str, model: str, api: str, key: str) -> None:
    _need(key)
    agent = Agent(
        name="r",
        system_prompt="Use web_search for facts and get_weather for weather.",
        llm=_llm(provider, model, api),
        tools=[web_search(), WEATHER],
    )
    result = agent.run(QUESTION)
    _assert_searched(result)
    assert result.cost > 0


@pytest.mark.parametrize(("provider", "model", "api", "key"), SEARCHERS[:2] + SEARCHERS[3:4])
def test_web_search_streamed(provider: str, model: str, api: str, key: str) -> None:
    _need(key)
    agent = Agent(
        name="r",
        system_prompt="Use web_search for facts and get_weather for weather.",
        llm=_llm(provider, model, api),
        tools=[web_search(), WEATHER],
    )

    async def go() -> str:
        return "".join([e.text async for e in agent.astream(QUESTION) if isinstance(e, TextDelta)])

    assert "18" in asyncio.run(go())


@pytest.mark.parametrize(("provider", "model", "api", "key"), SEARCHERS[:2])
def test_the_turn_survives_a_checkpoint_and_resume(
    provider: str, model: str, api: str, key: str, tmp_path
) -> None:
    _need(key)

    def get_weather_approved(city: str) -> str:
        """Current weather for a city, after a human approves the lookup."""
        decision = interrupt(reason="look up weather?", context={"city": city})
        return f"{city}: 18C, cloudy" if decision.approved else "not approved"

    agent = Agent(
        name="r",
        system_prompt="Use web_search for facts and get_weather_approved for weather.",
        llm=_llm(provider, model, api),
        tools=[web_search(), FunctionTool(name="get_weather_approved", fn=get_weather_approved)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "cp.db")),
    )
    paused = agent.run(
        QUESTION.replace("get_weather", "get_weather_approved"), execution_id="live-p"
    )
    if paused.status != "paused":
        pytest.skip("the model answered without the approval tool this time")
    resumed = agent.resume("live-p", resume_value=Resume(approved=True))
    assert "18" in resumed.output


# ── code_execution() ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model", "api", "key"),
    [
        ("gemini", "gemini-3.8-flash", "chat", "GEMINI_API_KEY"),
        ("anthropic", "claude-haiku-5-5", "chat", "ANTHROPIC_API_KEY"),
        ("openai", "gpt-5.1", "responses", "OPENAI_API_KEY"),
    ],
)
def test_code_execution(provider: str, model: str, api: str, key: str) -> None:
    _need(key)
    agent = Agent(name="c", llm=_llm(provider, model, api), tools=[code_execution()])
    result = agent.run("Use code to compute the 30th Fibonacci number. Reply with just the number.")
    assert "832040" in result.output.replace(",", "")
    (call, *_) = [c for c in result.tool_calls if c["tool_name"] == "code_execution"]
    assert "832040" in call["output"]


# ── guardrails ──────────────────────────────────────────────────────────────


def test_tool_result_guardrail_judges_the_real_sources() -> None:
    _need("GEMINI_API_KEY")
    seen: list[str] = []

    def no_sources(text: str) -> GuardrailResult:
        seen.append(text)
        return GuardrailResult(passed="sources:" not in text, message="no web sources allowed")

    rule = Guardrail(
        name="no_sources",
        guardrail_type=GuardrailType.code,
        position=GuardrailPosition.tool_result,
        blocking=True,
        fn=no_sources,
    )
    agent = Agent(
        name="r",
        llm=_llm("gemini", "gemini-3.8-flash", "chat"),
        tools=[web_search()],
        guardrails=[rule],
    )
    with pytest.raises(GuardrailBlockedError, match="no web sources"):
        agent.run("Search the web: what is the latest stable Python release?")
    assert any("http" in s for s in seen)
