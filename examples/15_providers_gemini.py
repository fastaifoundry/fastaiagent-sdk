"""Example 15: Run an agent against Google Gemini.

The ``gemini`` provider key uses fastaiagent's native Gemini wire — no
``google-generativeai`` runtime dependency, just ``httpx``. The preset
ships in v1.8.0; ``LLMClient(provider="gemini", ...)`` resolves
``base_url`` and reads the API key from ``GEMINI_API_KEY``.

Usage:
    export GEMINI_API_KEY=AIza...
    python examples/15_providers_gemini.py
"""

from fastaiagent import Agent, LLMClient

agent = Agent(
    name="gemini-greeter",
    system_prompt="You are a concise assistant. Reply in one short sentence.",
    llm=LLMClient(provider="gemini", model="gemini-2.5-flash"),
)


if __name__ == "__main__":
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        print("Skipping: GEMINI_API_KEY not set")
        print("Get a free key at https://aistudio.google.com/apikey")
    else:
        result = agent.run("Say hello to Alice in five words or fewer.")
        print(f"Output: {result.output}")
        print(f"Tokens used: {result.tokens_used}")
