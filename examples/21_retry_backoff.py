"""Example 21: Retry with Backoff — resilient LLM calls.

Demonstrates using max_retries for automatic retry on rate limits (429)
and server errors (5xx) with exponential backoff.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/21_retry_backoff.py
"""

from fastaiagent import Agent, LLMClient
from fastaiagent._internal.errors import LLMProviderError
from fastaiagent.llm import UserMessage


# --- Example 1: LLMClient with retries ---


def example_llm_retry():
    """Configure retry on the LLMClient directly."""
    print("=== Example 1: LLMClient with max_retries ===\n")

    llm = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
        max_retries=3,  # Retry up to 3 times on 429/5xx
    )

    try:
        response = llm.complete([UserMessage("Hello! What is 2+2?")])
        print(f"Response: {response.content}")
        print(f"Latency: {response.latency_ms}ms (includes any retry waits)")
    except LLMProviderError as e:
        print(f"Failed after retries: {e}")
        if e.status_code:
            print(f"Status code: {e.status_code}")
    print()


# --- Example 2: Agent with retry-enabled LLM ---


def example_agent_retry():
    """Pass a retry-enabled LLMClient to an Agent."""
    print("=== Example 2: Agent with retry-enabled LLM ===\n")

    agent = Agent(
        name="resilient-bot",
        system_prompt="You are a helpful assistant.",
        llm=LLMClient(
            provider="openai",
            model="gpt-4o-mini",
            max_retries=2,
        ),
    )

    result = agent.run("What's the capital of France?")
    print(f"Output: {result.output}")
    print(f"Latency: {result.latency_ms}ms")
    print()


# --- Example 3: Error handling with status codes ---


def example_error_handling():
    """Access status_code on LLMProviderError for targeted handling."""
    print("=== Example 3: Error handling with status codes ===\n")

    llm = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
        max_retries=1,
    )

    try:
        response = llm.complete([UserMessage("Hello")])
        print(f"Success: {response.content}")
    except LLMProviderError as e:
        if e.status_code == 429:
            print("Rate limited even after retries — back off more")
        elif e.status_code and e.status_code >= 500:
            print("Server error — try again later")
        elif e.status_code == 401:
            print("Auth error — check your API key")
        else:
            print(f"Other error ({e.status_code}): {e}")
    print()


# --- Main ---


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
    else:
        example_llm_retry()
        example_agent_retry()
        example_error_handling()
