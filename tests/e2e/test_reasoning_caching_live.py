"""Reasoning, Claude's thinking in the tool loop, and prompt caching against real
providers (1.91.0, 2A + 2B + X17 + X18).

What only a real provider can prove:

* each provider accepts the reasoning fields the SDK sends for each level;
* Claude's thinking blocks go back **byte-exact** — the API verifies their
  signature and rejects an altered block with a 400;
* they go back at all: with extended thinking (Haiku 4.5's budget path) a tool
  turn whose thinking was dropped is not an error — the API silently turns
  thinking off for it, and the response then carries no
  ``output_tokens_details`` (measured 2026-10-18). That is X17, and the check below;
* a streamed thinking block is reassembled exactly (text + ``signature_delta``);
* a second identical prompt is served from the cache, is counted as cached, and
  costs less than the first.

Cents per run. Run with the keys from ``~/.zshrc``::

    zsh -lc 'unset FASTAIAGENT_API_KEY FASTAIAGENT_TARGET; \
      FASTAIAGENT_LOCAL_DB=<scratch>/local.db \
      .venv/bin/python -m pytest tests/e2e/test_reasoning_caching_live.py -m e2e -v -rs'
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

PUZZLE = (
    "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. "
    "How much is the ball? Reply with just the amount."
)


def _need(*keys: str) -> None:
    for key in keys:
        if not os.environ.get(key):
            pytest.skip(f"{key} not set")


def _ask(client: Any, messages: list[Any], **kw: Any) -> Any:
    return asyncio.run(client.acomplete(messages, **kw))


def _weather(city: str) -> str:
    """Current weather for a city."""
    return {"paris": "18C, cloudy", "rome": "24C, sunny"}.get(city.lower(), "20C")


def _long_system(tokens: int) -> str:
    # A fresh prefix per run, so the first call is a cache write, not a stale hit.
    return f"Run {uuid.uuid4()}. Reference notes. " + (
        "The sky is blue and grass is green. " * (tokens // 8)
    )


# ── reasoning is accepted and comes back ───────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "model", "level", "key"),
    [
        ("openai", "gpt-5-mini", "low", "OPENAI_API_KEY"),
        ("openai", "gpt-5.1", "none", "OPENAI_API_KEY"),
        ("openai", "gpt-5.5", "medium", "OPENAI_API_KEY"),
        ("openai", "gpt-5.6-luna", "low", "OPENAI_API_KEY"),
        ("openai", "gpt-6-luna", "none", "OPENAI_API_KEY"),
        ("openai", "gpt-6-luna", "xhigh", "OPENAI_API_KEY"),
        ("openai", "gpt-6.1-sol", "medium", "OPENAI_API_KEY"),
        ("anthropic", "claude-haiku-5-5", "medium", "ANTHROPIC_API_KEY"),
        ("anthropic", "claude-haiku-5-5", "none", "ANTHROPIC_API_KEY"),
        ("anthropic", "claude-sonnet-5-5", "none", "ANTHROPIC_API_KEY"),
        ("anthropic", "claude-opus-4-8", "low", "ANTHROPIC_API_KEY"),
        ("anthropic", "claude-haiku-4-5", "low", "ANTHROPIC_API_KEY"),
        ("gemini", "gemini-3.8-flash", "low", "GEMINI_API_KEY"),
        ("gemini", "gemini-3.5-flash-lite", "minimal", "GEMINI_API_KEY"),
        ("gemini", "gemini-4-argon", "high", "GEMINI_API_KEY"),
        ("gemini", "gemini-2.5-flash", "none", "GEMINI_API_KEY"),
    ],
)
def test_each_provider_accepts_the_level(provider: str, model: str, level: str, key: str) -> None:
    _need(key)
    from fastaiagent import LLMClient
    from fastaiagent.llm.message import UserMessage

    response = _ask(
        LLMClient(provider=provider, model=model, reasoning=level), [UserMessage(PUZZLE)]
    )
    # Accepted (no 400), and the answer is the answer: no reasoning glued into it.
    assert "0.05" in (response.content or ""), response.content
    assert len(response.content or "") < 40, response.content


def test_gemini_shows_its_thoughts_apart_from_the_answer() -> None:
    _need("GEMINI_API_KEY")
    from fastaiagent import LLMClient
    from fastaiagent.llm.message import UserMessage

    client = LLMClient(provider="gemini", model="gemini-3.8-flash", reasoning="medium")
    for _ in range(3):  # it may skip thinking on an easy question
        response = _ask(client, [UserMessage(PUZZLE)])
        if response.reasoning:
            break
    assert response.reasoning, "includeThoughts returns a thought summary"
    assert response.usage.get("reasoning_tokens", 0) > 0
    assert response.content and len(response.content) < len(response.reasoning) + 40


def test_a_streamed_gemini_call_streams_its_thoughts() -> None:
    _need("GEMINI_API_KEY")
    from fastaiagent import LLMClient
    from fastaiagent.llm.message import UserMessage
    from fastaiagent.llm.stream import ReasoningDelta, TextDelta

    client = LLMClient(provider="gemini", model="gemini-3.8-flash", reasoning="medium")

    async def drain() -> list[Any]:
        return [e async for e in client.astream([UserMessage(PUZZLE)])]

    # The model may skip thinking on an easy question (seen 1 in 5), so up to three
    # tries; the answer must never carry the thoughts.
    thoughts = ""
    for _ in range(3):
        events = asyncio.run(drain())
        answer = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert "0.05" in answer and len(answer) < 300, answer
        thoughts = "".join(e.text for e in events if isinstance(e, ReasoningDelta))
        if thoughts:
            break
    assert thoughts, "includeThoughts streams thought summaries"
    assert thoughts not in answer


def test_claude_shows_summarized_thinking() -> None:
    _need("ANTHROPIC_API_KEY")
    from fastaiagent import LLMClient
    from fastaiagent.llm.message import UserMessage

    response = _ask(
        LLMClient(provider="anthropic", model="claude-haiku-4-5", reasoning="medium"),
        [UserMessage(PUZZLE)],
    )
    assert response.reasoning, "extended thinking returns the thinking text"
    assert response.reasoning_blocks and response.reasoning_blocks[0].data.get("signature")
    assert response.usage["reasoning_tokens"] > 0


# ── X17: thinking goes back byte-exact through the tool loop ───────────────


@pytest.mark.parametrize(
    "model",
    [
        # Extended thinking: the API REJECTS a tool turn without its thinking block.
        "claude-haiku-4-5",
        # Adaptive thinking, on by default.
        "claude-haiku-5-5",
        "claude-sonnet-5-5",
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["run", "astream"])
def test_a_thinking_claude_agent_completes_a_tool_loop(model: str, stream: bool) -> None:
    _need("ANTHROPIC_API_KEY")
    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent.llm.stream import ReasoningDelta

    agent = Agent(
        name="thinking-tools",
        system_prompt="Use the weather tool for each city, one call at a time.",
        llm=LLMClient(provider="anthropic", model=model, reasoning="medium"),
        tools=[FunctionTool(name="get_weather", fn=_weather)],
    )
    question = "Is it warmer in Paris or in Rome right now? Check both."
    if not stream:
        result = agent.run(question)
        assert result.status == "completed", result
        assert "rome" in result.output.lower()
        return

    async def drain() -> list[Any]:
        return [e async for e in agent.astream(question)]

    events = asyncio.run(drain())
    text = "".join(getattr(e, "text", "") for e in events if type(e).__name__ == "TextDelta")
    assert "rome" in text.lower()
    if model == "claude-haiku-4-5":
        assert any(isinstance(e, ReasoningDelta) and e.block for e in events)


def test_extended_thinking_stays_on_through_the_tool_loop(monkeypatch, tmp_path) -> None:
    """Every model call of the loop reports thinking tokens — none had its thinking
    silently switched off for a dropped block (X17)."""
    _need("ANTHROPIC_API_KEY")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    otel.reset()
    try:
        spans = InMemorySpanExporter()
        otel.add_span_processor(SimpleSpanProcessor(spans))
        agent = Agent(
            name="thinking-stays-on",
            system_prompt="Use the weather tool for each city, one call at a time.",
            llm=LLMClient(provider="anthropic", model="claude-haiku-4-5", reasoning="low"),
            tools=[FunctionTool(name="get_weather", fn=_weather)],
        )
        assert agent.run("Is it warmer in Paris or in Rome? Check both.").status == "completed"
        otel.get_tracer_provider().force_flush()
        calls = [
            dict(s.attributes) for s in spans.get_finished_spans() if s.name.startswith("llm.")
        ]
        assert len(calls) >= 2, "the loop made a tool call"
        assert all("gen_ai.usage.reasoning_tokens" in c for c in calls), [
            c.get("gen_ai.usage.reasoning_tokens") for c in calls
        ]
    finally:
        otel.reset()
        reset_config()


@pytest.mark.parametrize(
    ("model", "level"),
    # GPT-6 takes function tools on Chat Completions only at reasoning "none"
    # (next test); GPT-5 takes them at any level.
    [("gpt-5-mini", "low"), ("gpt-5.1", "medium"), ("gpt-6-luna", "none")],
)
@pytest.mark.parametrize("stream", [False, True], ids=["run", "astream"])
def test_a_reasoning_gpt_agent_completes_a_tool_loop(model: str, level: str, stream: bool) -> None:
    """``reasoning_effort`` on every turn of a tool loop (OpenAI Chat returns no
    reasoning text to carry back)."""
    _need("OPENAI_API_KEY")
    from fastaiagent import Agent, FunctionTool, LLMClient

    agent = Agent(
        name="gpt-reasoning-tools",
        system_prompt="Use the weather tool for each city, one call at a time.",
        llm=LLMClient(provider="openai", model=model, reasoning=level),
        tools=[FunctionTool(name="get_weather", fn=_weather)],
    )
    question = "Is it warmer in Paris or in Rome right now? Check both."
    if not stream:
        result = agent.run(question)
        assert result.status == "completed" and "rome" in result.output.lower(), result
        return

    async def drain() -> str:
        text = ""
        async for event in agent.astream(question):
            if type(event).__name__ == "TextDelta":
                text += event.text
        return text

    assert "rome" in asyncio.run(drain()).lower()


def test_gpt6_refuses_tools_with_reasoning_on_chat_completions_and_says_why() -> None:
    """Measured 2026-10-18: "Function tools with reasoning_effort are not supported
    for gpt-6-luna in /v1/chat/completions". The SDK sends what was asked and the
    provider's reason reaches the caller — no silent drop of the setting."""
    _need("OPENAI_API_KEY")
    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent._internal.errors import LLMProviderError

    agent = Agent(
        name="gpt6-tools-reasoning",
        llm=LLMClient(provider="openai", model="gpt-6-luna", reasoning="low"),
        tools=[FunctionTool(name="get_weather", fn=_weather)],
    )
    with pytest.raises(LLMProviderError, match="reasoning_effort"):
        agent.run("Weather in Paris?")


def test_a_thinking_gemini_agent_completes_a_tool_loop() -> None:
    _need("GEMINI_API_KEY")
    from fastaiagent import Agent, FunctionTool, LLMClient

    agent = Agent(
        name="gemini-thinking-tools",
        llm=LLMClient(provider="gemini", model="gemini-3.8-flash", reasoning="low"),
        tools=[FunctionTool(name="get_weather", fn=_weather)],
    )
    result = agent.run("Is it warmer in Paris or in Rome? Use the tool for each.")
    assert result.status == "completed" and "rome" in result.output.lower()


# ── prompt caching: a hit is counted and costs less ────────────────────────


@pytest.mark.parametrize(
    ("provider", "model", "key", "tokens"),
    [
        ("anthropic", "claude-haiku-5-5", "ANTHROPIC_API_KEY", 2000),
        ("anthropic", "claude-haiku-4-5", "ANTHROPIC_API_KEY", 6000),
        ("openai", "gpt-6-luna", "OPENAI_API_KEY", 3000),
        ("openai", "gpt-5-mini", "OPENAI_API_KEY", 3000),
    ],
)
def test_a_repeated_prompt_is_served_from_the_cache(
    provider: str, model: str, key: str, tokens: int
) -> None:
    _need(key)
    from fastaiagent import LLMClient
    from fastaiagent._internal.pricing import usage_cost
    from fastaiagent.llm.message import SystemMessage, UserMessage

    client = LLMClient(provider=provider, model=model, cache=True)
    messages = [SystemMessage(_long_system(tokens)), UserMessage("Reply with the single word OK.")]
    first = _ask(client, messages)
    second = None
    for _ in range(3):  # OpenAI routes to a cache-warm machine on a best-effort basis
        second = _ask(client, messages)
        if second.usage.get("cached_tokens"):
            break
    assert second is not None
    assert second.usage.get("cached_tokens", 0) > 0, second.usage
    assert second.usage["prompt_tokens"] >= second.usage["cached_tokens"]
    first_usd, known = usage_cost(model, first.usage, provider=provider)
    second_usd, _ = usage_cost(model, second.usage, provider=provider)
    assert known and second_usd < first_usd, (first.usage, second.usage)


def test_a_streamed_claude_call_reports_its_cache_hit() -> None:
    _need("ANTHROPIC_API_KEY")
    from fastaiagent import LLMClient
    from fastaiagent.llm.message import SystemMessage, UserMessage
    from fastaiagent.llm.stream import Usage

    client = LLMClient(provider="anthropic", model="claude-haiku-5-5", cache=True)
    messages = [SystemMessage(_long_system(2000)), UserMessage("Reply with OK.")]
    _ask(client, messages)

    async def drain() -> list[Any]:
        return [e async for e in client.astream(messages)]

    (usage,) = [e for e in asyncio.run(drain()) if isinstance(e, Usage)]
    assert usage.cached_tokens > 0 and usage.prompt_tokens > usage.cached_tokens
