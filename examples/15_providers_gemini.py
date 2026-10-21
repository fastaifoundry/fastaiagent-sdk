"""Example 15: Run a tool-using agent on Google Gemini 3.5+.

The ``gemini`` provider key uses fastaiagent's native Gemini wire — no
``google-generativeai`` runtime dependency, just ``httpx``.
``LLMClient(provider="gemini", ...)`` resolves ``base_url`` and reads the API
key from ``GEMINI_API_KEY``; without ``model=`` it uses ``gemini-3.8-flash``.

Gemini 3.5 and later sign the first function call of each turn and reject the
next request unless that signature comes back — the SDK carries it for you
(since 1.91.0), so the tool loop below just works. Try ``gemini-4-argon`` or
``gemini-flash-latest`` too.

Usage:
    export GEMINI_API_KEY=AIza...
    python examples/15_providers_gemini.py
"""

from fastaiagent import Agent, FunctionTool, LLMClient


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"oslo": "-4C, snow", "lisbon": "19C, sunny"}.get(city.lower(), "unknown")


agent = Agent(
    name="gemini-weather",
    system_prompt="You are a concise assistant. Use the weather tool, then reply in one sentence.",
    llm=LLMClient(provider="gemini", model="gemini-3.8-flash"),
    tools=[FunctionTool(name="get_weather", fn=get_weather)],
)


if __name__ == "__main__":
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        print("Skipping: GEMINI_API_KEY not set")
        print("Get a free key at https://aistudio.google.com/apikey")
    else:
        result = agent.run("What's the weather in Oslo?")
        print(f"Output: {result.output}")
        print(f"Tool calls: {[c['tool_name'] for c in result.tool_calls]}")
        print(f"Tokens used: {result.tokens_used}  Cost: ${result.cost:.6f}")
