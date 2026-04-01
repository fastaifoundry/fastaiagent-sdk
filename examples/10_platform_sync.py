"""Example 10: Push agents/chains to FastAIAgent Platform.

Shows how to sync SDK resources to the platform for
visual editing, monitoring, and team collaboration.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export FASTAIAGENT_API_KEY=fa_k_...
    export FASTAIAGENT_TARGET=http://localhost:8001
    python examples/10_platform_sync.py
"""

import os

from fastaiagent import Agent, FastAI, FunctionTool, LLMClient

if __name__ == "__main__":
    api_key = os.environ.get("FASTAIAGENT_API_KEY", "")
    target = os.environ.get("FASTAIAGENT_TARGET", "http://localhost:8001")

    if not api_key:
        print("Skipping: FASTAIAGENT_API_KEY not set")
        print("Run: export FASTAIAGENT_API_KEY=fa_k_... && python examples/10_platform_sync.py")
    else:
        # Define an agent locally
        def lookup(query: str) -> str:
            """Look up information."""
            return f"Found info for: {query}"

        agent = Agent(
            name="sdk-demo-agent",
            system_prompt="You are a demo agent pushed from the SDK.",
            llm=LLMClient(provider="openai", model="gpt-4.1"),
            tools=[FunctionTool(name="lookup", fn=lookup)],
        )

        # Connect to platform and push
        fa = FastAI(api_key=api_key, target=target)

        print(f"Pushing agent '{agent.name}' to {target}...")
        result = fa.push(agent)
        print(f"  Resource: {result.resource_type}")
        print(f"  Name: {result.name}")
        print(f"  Created: {result.created}")
        print(f"  Dependencies: {result.dependencies_pushed}")
        print("Done!")
