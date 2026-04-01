"""Example 01: Minimal agent with one tool.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/01_simple_agent.py
"""

from fastaiagent import Agent, FunctionTool, LLMClient


def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}! Welcome aboard."


agent = Agent(
    name="greeter",
    system_prompt="You are a friendly assistant. Use the greet tool to greet users.",
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[FunctionTool(name="greet", fn=greet)],
)

if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run with: export OPENAI_API_KEY=sk-... && python examples/01_simple_agent.py")
    else:
        result = agent.run("Please greet Alice")
        print(f"Output: {result.output}")
        print(f"Tool calls: {result.tool_calls}")
        print(f"Tokens used: {result.tokens_used}")
