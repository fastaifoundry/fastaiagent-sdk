"""Example 115: OpenAI's Responses API, and tools the model's provider runs (1.91.0).

Two pieces:

    LLMClient(provider="openai", api="responses")
        OpenAI's Responses API instead of Chat Completions — the only way to give
        GPT-6 models, and GPT-5.4+ with reasoning on, tools. Same agent, same
        LLMResponse; reasoning summaries come back in ``response.reasoning``.

    web_search(), code_execution(), file_search()     (fastaiagent.tool)
        Run by the provider inside the model call — OpenAI (Responses), Anthropic
        and Gemini, each in its own shape. Every call is recorded (a
        ``tool.<name>`` span and a ``result.tool_calls`` entry), judged by your
        ``tool_result`` guardrails, and priced with the tokens.

It runs on whichever keys are set: Gemini 3.8 Flash first, then Claude and OpenAI.

Usage:
    zsh -lc 'python examples/115_responses_api_and_provider_tools.py'
"""

from __future__ import annotations

import os

from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.tool import code_execution, web_search

QUESTION = "What is the latest stable Python release? Then get the weather in Paris."


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"paris": "18C, cloudy"}.get(city.lower(), "20C")


def research(llm: LLMClient) -> None:
    print(f"── web_search() + a function tool on {llm.provider}/{llm.model}")
    agent = Agent(
        name="researcher",
        system_prompt="Search the web for facts you don't know. Use the weather tool for weather.",
        llm=llm,
        tools=[web_search(), FunctionTool(name="get_weather", fn=get_weather)],
    )
    result = agent.run(QUESTION)
    print(f"  answer: {result.output.strip()[:160]!r}")
    for call in result.tool_calls:
        first = (call.get("output") or call.get("error") or "").splitlines()[:2]
        print(f"  {call['tool_name']:<14} {call['arguments']}  →  {' | '.join(first)[:90]}")
    print(f"  cost: ${result.cost:.5f} (tokens + the provider's per-search fee)")


def compute(llm: LLMClient) -> None:
    print(f"── code_execution() on {llm.provider}/{llm.model}")
    agent = Agent(name="calc", llm=llm, tools=[code_execution()])
    result = agent.run("Use code to compute the 30th Fibonacci number. Reply with just the number.")
    print(f"  answer: {result.output.strip()!r}")
    for call in result.tool_calls:
        print(f"  ran: {call['arguments'].get('code') or call['arguments'].get('command')!r}"[:100])


def responses_api_reasoning() -> None:
    print("── OpenAI Responses API: tools with reasoning on gpt-6-luna")
    # On Chat Completions this exact agent is refused: "Function tools with
    # reasoning_effort are not supported for gpt-6-luna in /v1/chat/completions".
    llm = LLMClient(provider="openai", model="gpt-6-luna", api="responses", reasoning="medium")
    agent = Agent(name="weather", llm=llm, tools=[FunctionTool(name="get_weather", fn=get_weather)])
    result = agent.run("What's the weather in Paris?")
    print(f"  answer: {result.output.strip()[:120]!r}")


def main() -> None:
    ran = False
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        research(LLMClient(provider="gemini", model="gemini-3.8-flash"))
        compute(LLMClient(provider="gemini", model="gemini-3.8-flash"))
        ran = True
    if os.environ.get("ANTHROPIC_API_KEY"):
        research(LLMClient(provider="anthropic", model="claude-sonnet-5-5"))
        compute(LLMClient(provider="anthropic", model="claude-haiku-5-5"))
        ran = True
    if os.environ.get("OPENAI_API_KEY"):
        responses_api_reasoning()
        research(LLMClient(provider="openai", model="gpt-5.1", api="responses"))
        compute(LLMClient(provider="openai", model="gpt-5.1", api="responses"))
        ran = True
    if not ran:
        print("Set GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY to run this example.")


if __name__ == "__main__":
    main()
