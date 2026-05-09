"""Example 16: Run an agent against Groq (fast Llama / Mixtral inference).

Groq is OpenAI-compatible; the v1.8.0 preset registry ships the right
``base_url`` and reads the API key from ``GROQ_API_KEY``. Tool calling
is supported on the larger Llama-3.x models.

Usage:
    export GROQ_API_KEY=gsk_...
    python examples/16_providers_groq.py
"""

from fastaiagent import Agent, FunctionTool, LLMClient


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


agent = Agent(
    name="groq-calc",
    system_prompt=(
        "You are a precise calculator. Use the multiply tool to compute products. "
        "Reply with just the number."
    ),
    llm=LLMClient(provider="groq", model="llama-3.3-70b-versatile"),
    tools=[FunctionTool(name="multiply", fn=multiply)],
)


if __name__ == "__main__":
    import os

    if not os.environ.get("GROQ_API_KEY"):
        print("Skipping: GROQ_API_KEY not set")
        print("Get a free key at https://console.groq.com/keys")
    else:
        result = agent.run("What is 137 multiplied by 42?")
        print(f"Output: {result.output}")
        print(f"Tool calls: {result.tool_calls}")
