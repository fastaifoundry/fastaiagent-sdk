"""Example 15: RunContext with Anthropic (Claude).

Same pattern as Example 14 but using the Anthropic provider
to verify RunContext works identically across providers.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/15_context_di_anthropic.py
"""

from dataclasses import dataclass

from fastaiagent import Agent, LLMClient, RunContext, tool


@dataclass
class UserSession:
    user_id: str
    api_key: str
    feature_flags: dict


@tool(name="get_profile")
def get_profile(ctx: RunContext[UserSession]) -> str:
    """Get the current user's profile."""
    return f"User: {ctx.state.user_id}, Features: {ctx.state.feature_flags}"


@tool(name="fetch_data")
async def fetch_data(ctx: RunContext[UserSession], query: str) -> str:
    """Fetch data using the user's API key."""
    return f"[Queried '{query}' with key={ctx.state.api_key[:8]}...]"


@tool(name="multiply")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


agent = Agent(
    name="claude-agent",
    system_prompt="You are a helpful assistant. Use tools when needed. Be concise.",
    llm=LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001"),
    tools=[get_profile, fetch_data, multiply],
)


if __name__ == "__main__":
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipping: ANTHROPIC_API_KEY not set")
        print("Run: export ANTHROPIC_API_KEY=sk-ant-... && python examples/15_context_di_anthropic.py")
        raise SystemExit(0)

    ctx = RunContext(
        state=UserSession(
            user_id="u-claude-test",
            api_key="sk-secret-key-12345",
            feature_flags={"v2_flow": True, "dark_mode": False},
        )
    )

    print("=" * 60)
    print("Example 1: Context-only tool via Anthropic")
    print("=" * 60)
    result = agent.run("Show me my profile", context=ctx)
    print(f"Output: {result.output}")
    print(f"Tool calls: {[tc['tool_name'] for tc in result.tool_calls]}")

    print()
    print("=" * 60)
    print("Example 2: Async tool with context via Anthropic")
    print("=" * 60)
    result = agent.run("Fetch data for query 'sales Q1'", context=ctx)
    print(f"Output: {result.output}")

    print()
    print("=" * 60)
    print("Example 3: No context needed — plain tool via Anthropic")
    print("=" * 60)
    result = agent.run("Multiply 7.5 by 12")
    print(f"Output: {result.output}")
