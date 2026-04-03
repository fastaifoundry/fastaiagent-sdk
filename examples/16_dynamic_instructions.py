"""Example 16: Dynamic Instructions — per-request system prompts.

Shows how to use a callable system_prompt so the same agent instance
adapts its behavior for each user/request without rebuilding.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/16_dynamic_instructions.py
"""

from dataclasses import dataclass
from datetime import date

from fastaiagent import Agent, LLMClient, RunContext, tool


# --- Runtime state ---


@dataclass
class UserState:
    user_name: str
    plan_tier: str  # "free", "pro", "enterprise"
    locale: str


# --- A tool that uses the same context ---


@tool(name="get_plan_limits")
def get_plan_limits(ctx: RunContext[UserState]) -> str:
    """Get the current user's plan limits."""
    limits = {
        "free": {"api_calls": 100, "storage_gb": 1, "support": "community"},
        "pro": {"api_calls": 10_000, "storage_gb": 50, "support": "email"},
        "enterprise": {"api_calls": "unlimited", "storage_gb": 500, "support": "24/7 phone"},
    }
    plan = limits.get(ctx.state.plan_tier, limits["free"])
    return f"Plan '{ctx.state.plan_tier}' limits: {plan}"


# --- Build agent with dynamic prompt ---

agent = Agent(
    name="support",
    system_prompt=lambda ctx: (
        f"You are a support agent. "
        f"The customer's name is {ctx.state.user_name}. "
        f"Their plan: {ctx.state.plan_tier}. "
        f"Respond in locale: {ctx.state.locale}. "
        f"Today is {date.today()}."
        if ctx
        else "You are a support agent. Be helpful and concise."
    ),
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[get_plan_limits],
)


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY=sk-... && python examples/16_dynamic_instructions.py")
        raise SystemExit(0)

    # --- Same agent, different users ---

    print("=" * 60)
    print("Example 1: Enterprise user (Alice)")
    print("=" * 60)
    ctx_alice = RunContext(
        state=UserState(user_name="Alice", plan_tier="enterprise", locale="en-US")
    )
    result = agent.run("What are my plan limits?", context=ctx_alice)
    print(f"Output: {result.output}\n")

    print("=" * 60)
    print("Example 2: Free user (Bob)")
    print("=" * 60)
    ctx_bob = RunContext(
        state=UserState(user_name="Bob", plan_tier="free", locale="en-US")
    )
    result = agent.run("What are my plan limits?", context=ctx_bob)
    print(f"Output: {result.output}\n")

    print("=" * 60)
    print("Example 3: No context — fallback prompt")
    print("=" * 60)
    result = agent.run("Hello, who are you?")
    print(f"Output: {result.output}\n")

    # --- Serialization guard ---

    print("=" * 60)
    print("Example 4: to_dict() raises on callable prompt")
    print("=" * 60)
    try:
        agent.to_dict()
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Compare with a static-prompt agent
    static_agent = Agent(
        name="static",
        system_prompt="You are a support agent.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )
    d = static_agent.to_dict()
    print(f"Static agent serializes fine: {d['system_prompt']}")
