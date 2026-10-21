"""Example 114: choose how structured output is asked for, and watch it form (1.91.0).

    AgentConfig(output_mode="auto" | "native" | "tool" | "prompt")
        "native" uses the provider's own schema-constrained output (Anthropic
        output_config.format, Bedrock outputConfig.textFormat); "tool" answers
        through a final_result tool; "prompt" puts the schema in the prompt only.
    PartialOutput(data=...)
        The structured reply so far, while it streams.

It runs on the first of Gemini 3.8 Flash, Claude Sonnet 5.5 and gpt-5-mini whose key
is set — and every mode on each.

Usage:
    zsh -lc 'python examples/114_output_modes_and_partial_output.py'
"""

from __future__ import annotations

import asyncio
import os

from pydantic import BaseModel, Field

from fastaiagent import Agent, AgentConfig, FunctionTool, LLMClient, PartialOutput


class Forecast(BaseModel):
    city: str
    celsius: float = Field(ge=-90, le=60)
    summary: str = Field(description="One short sentence")


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"rome": "24C, sunny", "oslo": "-4C, snow"}.get(city.lower(), "20C")


def agent(llm: LLMClient, mode: str) -> Agent:
    return Agent(
        name=f"forecaster-{mode}",
        system_prompt="Use the weather tool for the city asked about, then answer.",
        llm=llm,
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
        output_type=Forecast,
        config=AgentConfig(output_mode=mode),
    )


async def watch(llm: LLMClient) -> None:
    print("  ── streaming, mode auto")
    async for event in agent(llm, "auto").astream("What's the weather in Oslo?"):
        if isinstance(event, PartialOutput):
            print(f"    partial: {event.data}")


def main() -> None:
    lineup = [
        ("GEMINI_API_KEY", "gemini", "gemini-3.8-flash"),
        ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-5-5"),
        ("OPENAI_API_KEY", "openai", "gpt-5-mini"),
    ]
    available = [(p, m) for key, p, m in lineup if os.environ.get(key)]
    if not available:
        raise SystemExit("Set GEMINI_API_KEY, ANTHROPIC_API_KEY or OPENAI_API_KEY.")
    for provider, model in available:
        print(f"── {model}")
        llm = LLMClient(provider=provider, model=model)
        for mode in ("auto", "native", "tool", "prompt"):
            result = agent(llm, mode).run("What's the weather in Rome?")
            print(f"  {mode:<7} {result.parsed!r}")
        asyncio.run(watch(llm))


if __name__ == "__main__":
    main()
