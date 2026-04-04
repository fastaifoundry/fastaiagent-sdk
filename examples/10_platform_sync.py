"""Example 10: Connect to FastAIAgent Platform.

Shows how to connect the SDK to the platform for automatic
trace export, prompt management, and evaluation services.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export FASTAIAGENT_API_KEY=fa_k_...
    export FASTAIAGENT_TARGET=http://localhost:8001
    python examples/10_platform_sync.py
"""

import os

import fastaiagent as fa
from fastaiagent import Agent, FunctionTool, LLMClient

if __name__ == "__main__":
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:8001")

    if not api_key:
        print("Skipping: FASTAIAGENT_API_KEY not set")
        print("Run: export FASTAIAGENT_API_KEY=fa_k_... && python examples/10_platform_sync.py")
    else:
        # Connect to platform — traces auto-sent, prompts available
        fa.connect(api_key=api_key, target=target, project="demo")

        # Define an agent locally
        def lookup(query: str) -> str:
            """Look up information."""
            return f"Found info for: {query}"

        agent = Agent(
            name="sdk-demo-agent",
            system_prompt="You are a demo agent. Use tools to answer questions.",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="lookup", fn=lookup)],
        )

        # Run the agent — trace automatically sent to platform
        result = agent.run("Look up information about FastAIAgent")
        print(f"Output: {result.output}")
        print(f"Trace ID: {result.trace_id}")
        print("Trace automatically sent to platform dashboard!")

        # Disconnect when done
        fa.disconnect()
        print("Disconnected.")
