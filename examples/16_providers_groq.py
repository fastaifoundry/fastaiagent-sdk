"""Example 16: Run an agent against Groq (fast OpenAI-compatible inference).

Groq is OpenAI-compatible; the v1.8.0 preset registry ships the right
``base_url`` and reads the API key from ``GROQ_API_KEY``.

Groq rotates its catalogue and retires model ids without notice — the
Llama-3.x ids this example used to pin are gone. If you get a 404
``model_not_found``, list what is actually served and pick a tool-calling
chat model from it::

    curl -H "Authorization: Bearer $GROQ_API_KEY" \
         https://api.groq.com/openai/v1/models

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
    llm=LLMClient(provider="groq", model="openai/gpt-oss-20b"),
    tools=[FunctionTool(name="multiply", fn=multiply)],
)


if __name__ == "__main__":
    import os

    if not os.environ.get("GROQ_API_KEY"):
        print("Skipping: GROQ_API_KEY not set")
        print("Get a free key at https://console.groq.com/keys")
    else:
        from fastaiagent._internal.errors import LLMProviderError

        try:
            result = agent.run("What is 137 multiplied by 42?")
        except LLMProviderError as exc:
            # A retired model id is a catalogue problem, not a bug in your code
            # — say so instead of dumping a traceback on the reader.
            if "model_not_found" in str(exc) or "does not exist" in str(exc):
                print(f"Groq no longer serves this model id: {exc}")
                print(
                    "List the live catalogue and edit the model= above:\n"
                    '  curl -H "Authorization: Bearer $GROQ_API_KEY" '
                    "https://api.groq.com/openai/v1/models"
                )
                raise SystemExit(0) from None
            raise
        print(f"Output: {result.output}")
        print(f"Tool calls: {result.tool_calls}")
