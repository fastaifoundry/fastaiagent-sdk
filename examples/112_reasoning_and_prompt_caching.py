"""Example 112: how hard the model thinks, and paying less for a repeated prompt (1.91.0).

Two settings on ``LLMClient``, one vocabulary for every provider:

    reasoning=   "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
                 → Gemini thinkingLevel, OpenAI reasoning_effort, Claude effort +
                 thinking. What the model shows of its thinking is
                 ``response.reasoning`` — never part of the answer.
    cache=       True (or "1h") → Anthropic cache_control / Bedrock cachePoint;
                 OpenAI and Gemini cache on their own. Cache reads and writes are
                 counted in ``usage`` and priced at their own rates.

It runs on whichever keys are set: Gemini 3.8 Flash first, then Claude and OpenAI.

Usage:
    zsh -lc 'python examples/112_reasoning_and_prompt_caching.py'
"""

from __future__ import annotations

import asyncio
import os
import uuid

from fastaiagent import Agent, FunctionTool, LLMClient, ReasoningDelta, TextDelta
from fastaiagent._internal.pricing import usage_cost
from fastaiagent.llm import SystemMessage, UserMessage

PUZZLE = (
    "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. "
    "How much is the ball? Reply with just the amount."
)


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"paris": "18C, cloudy", "rome": "24C, sunny"}.get(city.lower(), "20C")


def show_reasoning(provider: str, model: str) -> None:
    print(f"── reasoning on {model}")
    llm = LLMClient(provider=provider, model=model, reasoning="medium")
    response = llm.complete([UserMessage(PUZZLE)])
    print(f"  answer:    {response.content!r}")
    # None when the model skipped thinking (Claude may, on an easy question) or the
    # API never shows it (OpenAI Chat Completions — the count is still in usage).
    print(f"  reasoning: {(response.reasoning or '(none shown)')[:110]!r}")
    print(
        f"  tokens:    {response.usage.get('reasoning_tokens', 0)} of "
        f"{response.usage['completion_tokens']} output tokens spent thinking"
    )


def show_streamed_tool_loop(provider: str, model: str) -> None:
    print(f"── a thinking agent with a tool, streamed ({model})")
    agent = Agent(
        name="weather",
        system_prompt="Use the weather tool for each city, one call at a time.",
        llm=LLMClient(provider=provider, model=model, reasoning="low"),
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
    )

    async def run() -> None:
        thinking = answer = ""
        async for event in agent.astream("Is it warmer in Paris or in Rome? Check both."):
            if isinstance(event, ReasoningDelta):
                thinking += event.text
            elif isinstance(event, TextDelta):
                answer += event.text
        print(f"  thought for {len(thinking)} chars; answered: {answer.strip()[:100]!r}")

    asyncio.run(run())


def show_caching(provider: str, model: str) -> None:
    print(f"── prompt caching on {model}")
    llm = LLMClient(provider=provider, model=model, cache=True)
    policy = f"Policy {uuid.uuid4()}. " + "Refunds are issued within 14 days of purchase. " * 300
    messages = [SystemMessage(policy), UserMessage("Reply with the single word OK.")]
    for attempt in ("first", "second"):
        usage = llm.complete(messages).usage
        usd, _ = usage_cost(model, usage, provider=provider)
        print(
            f"  {attempt:<6} prompt={usage['prompt_tokens']:>5} "
            f"cached={usage.get('cached_tokens', 0):>5} "
            f"written={usage.get('cache_write_tokens', 0):>5}  ${usd:.6f}"
        )


def main() -> None:
    lineup = [
        ("GEMINI_API_KEY", "gemini", "gemini-3.8-flash"),
        ("ANTHROPIC_API_KEY", "anthropic", "claude-haiku-5-5"),
        ("OPENAI_API_KEY", "openai", "gpt-5-mini"),
    ]
    available = [(p, m) for key, p, m in lineup if os.environ.get(key)]
    if not available:
        raise SystemExit("Set GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY.")
    for provider, model in available:
        show_reasoning(provider, model)
    show_streamed_tool_loop(*available[0])
    for provider, model in available:
        if provider != "gemini":  # Gemini's implicit cache is best-effort on short runs
            show_caching(provider, model)


if __name__ == "__main__":
    main()
