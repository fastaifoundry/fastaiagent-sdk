"""Example 17: Route through OpenRouter to access many models with one key.

OpenRouter is OpenAI-compatible — fastaiagent's preset registry ships
the right ``base_url`` and reads the API key from
``OPENROUTER_API_KEY``. Choose any underlying model by its OpenRouter
slug (e.g. ``openai/gpt-4o-mini``, ``anthropic/claude-3-5-haiku``).

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/17_providers_openrouter.py
"""

from fastaiagent import Agent, LLMClient

agent = Agent(
    name="openrouter-poet",
    system_prompt="You write very short poems. Two lines max.",
    llm=LLMClient(provider="openrouter", model="openai/gpt-4o-mini"),
)


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Skipping: OPENROUTER_API_KEY not set")
        print("Get a key at https://openrouter.ai/keys")
    else:
        result = agent.run("Write a haiku about provider routing.")
        print(f"Output: {result.output}")
