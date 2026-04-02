"""Example 12: Streaming agent responses.

Demonstrates three streaming layers:
  1. LLMClient.astream()  — raw token streaming from the LLM
  2. stream_tool_loop()    — streaming with tool execution
  3. Agent.astream()       — full agent streaming with guardrails and memory

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/12_streaming.py
"""

import asyncio

from fastaiagent import Agent, FunctionTool, LLMClient
from fastaiagent.agent import AgentMemory
from fastaiagent.llm.message import UserMessage
from fastaiagent.llm.stream import TextDelta, ToolCallEnd, ToolCallStart, Usage


# --- Tool definitions ---


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny, 22C in {city}"


def search(query: str) -> str:
    """Search the web for information."""
    return f"Top result for '{query}': FastAIAgent SDK is an AI agent framework."


# --- Example 1: Raw LLM streaming ---


async def example_llm_stream():
    """Stream tokens directly from the LLM."""
    print("=== Example 1: LLMClient.astream() ===\n")

    llm = LLMClient(provider="openai", model="gpt-4.1")
    print("Assistant: ", end="", flush=True)

    async for event in llm.astream([UserMessage("What is 2 + 2? Answer briefly.")]):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Usage):
            print(f"\n  [{event.prompt_tokens} prompt + {event.completion_tokens} completion tokens]")

    print()


# --- Example 2: Streaming with tool calls ---


async def example_agent_stream():
    """Stream an agent response including tool calls."""
    print("\n=== Example 2: Agent.astream() with tools ===\n")

    agent = Agent(
        name="weather-bot",
        system_prompt="You are a helpful weather assistant. Use the get_weather tool to answer.",
        llm=LLMClient(provider="openai", model="gpt-4.1"),
        tools=[FunctionTool(name="get_weather", fn=get_weather)],
    )

    print("Assistant: ", end="", flush=True)
    async for event in agent.astream("What's the weather in Paris?"):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallStart):
            print(f"\n  [Calling {event.tool_name}...]", end="", flush=True)
        elif isinstance(event, ToolCallEnd):
            print(f" [done]", end="", flush=True)
        elif isinstance(event, Usage):
            print(f"\n  [{event.prompt_tokens}+{event.completion_tokens} tokens]", end="")
    print()


# --- Example 3: Streaming chat with memory ---


async def example_streaming_chat():
    """Multi-turn streaming chat with memory."""
    print("\n=== Example 3: Streaming chat with memory ===\n")

    agent = Agent(
        name="chatbot",
        system_prompt="You are a friendly assistant. Keep answers under 2 sentences.",
        llm=LLMClient(provider="openai", model="gpt-4.1"),
        tools=[FunctionTool(name="search", fn=search)],
        memory=AgentMemory(),
    )

    questions = [
        "What is FastAIAgent SDK?",
        "Can you search for more details about it?",
        "Summarize what you found.",
    ]

    for question in questions:
        print(f"You: {question}")
        print("Assistant: ", end="", flush=True)
        async for event in agent.astream(question):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                print(f"\n  [Searching...]", end="", flush=True)
            elif isinstance(event, ToolCallEnd):
                print(f" [done]", end="", flush=True)
        print("\n")


# --- Main ---


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY=sk-... && python examples/12_streaming.py")
    else:
        asyncio.run(example_llm_stream())
        asyncio.run(example_agent_stream())
        asyncio.run(example_streaming_chat())
