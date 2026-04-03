"""Example 17: Advanced Dynamic Instructions — named functions, feature flags, streaming.

Shows:
  - Named function (not lambda) for complex prompt logic
  - Feature flags / A/B testing via context
  - Streaming with dynamic instructions

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/17_dynamic_instructions_advanced.py
"""

from dataclasses import dataclass, field

from fastaiagent import Agent, LLMClient, RunContext, tool
from fastaiagent.llm.stream import TextDelta


# --- Runtime state with feature flags ---


@dataclass
class AppState:
    user_name: str
    company: str
    plan_tier: str
    open_tickets: int = 0
    feature_flags: dict = field(default_factory=dict)


# --- Named function for complex prompt logic ---


def build_support_prompt(ctx: RunContext[AppState] | None) -> str:
    """Build a personalized support prompt.

    Using a named function (instead of a lambda) keeps complex
    prompt logic readable, testable, and debuggable.
    """
    if ctx is None:
        return "You are a customer support agent. Be helpful and concise."

    user = ctx.state
    lines = [
        f"You are a support agent for {user.company}.",
        f"Customer: {user.user_name}",
        f"Plan: {user.plan_tier}",
    ]

    # High-priority customers get special treatment
    if user.plan_tier == "enterprise":
        lines.append("This is a high-priority enterprise customer. Be thorough.")

    # Empathy for frustrated customers
    if user.open_tickets > 3:
        lines.append(
            f"Note: this customer has {user.open_tickets} open tickets. "
            f"Be extra empathetic and try to resolve their issue quickly."
        )

    # A/B test: concise vs detailed mode
    if user.feature_flags.get("concise_mode"):
        lines.append("Keep responses under 2 sentences.")
    else:
        lines.append("Provide detailed, thorough explanations.")

    return "\n".join(lines)


# --- Tools ---


@tool(name="check_status")
def check_status(ctx: RunContext[AppState], ticket_id: str) -> str:
    """Check the status of a support ticket."""
    # Simulated response
    return f"Ticket {ticket_id} for {ctx.state.user_name}: status=open, priority=high"


@tool(name="escalate")
def escalate(ctx: RunContext[AppState], reason: str) -> str:
    """Escalate an issue to a senior agent."""
    return f"Escalated for {ctx.state.user_name} ({ctx.state.plan_tier}): {reason}"


# --- Build agent ---

agent = Agent(
    name="advanced-support",
    system_prompt=build_support_prompt,
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[check_status, escalate],
)


if __name__ == "__main__":
    import asyncio
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print(
            "Run: export OPENAI_API_KEY=sk-... && "
            "python examples/17_dynamic_instructions_advanced.py"
        )
        raise SystemExit(0)

    # --- Example 1: Enterprise customer with many tickets ---

    print("=" * 60)
    print("Example 1: Enterprise customer, 5 open tickets")
    print("=" * 60)
    ctx = RunContext(
        state=AppState(
            user_name="Alice",
            company="Acme Corp",
            plan_tier="enterprise",
            open_tickets=5,
            feature_flags={"concise_mode": False},
        )
    )
    result = agent.run("Check ticket TK-789 and escalate if needed", context=ctx)
    print(f"Output: {result.output}")
    print(f"Tool calls: {[tc['tool_name'] for tc in result.tool_calls]}\n")

    # --- Example 2: A/B test — concise mode ---

    print("=" * 60)
    print("Example 2: Same user, concise_mode=True (A/B test)")
    print("=" * 60)
    ctx_concise = RunContext(
        state=AppState(
            user_name="Alice",
            company="Acme Corp",
            plan_tier="enterprise",
            open_tickets=5,
            feature_flags={"concise_mode": True},
        )
    )
    result = agent.run("Check ticket TK-789", context=ctx_concise)
    print(f"Output: {result.output}\n")

    # --- Example 3: Streaming with dynamic instructions ---

    print("=" * 60)
    print("Example 3: Streaming with dynamic prompt")
    print("=" * 60)
    ctx_stream = RunContext(
        state=AppState(
            user_name="Bob",
            company="StartupCo",
            plan_tier="pro",
            feature_flags={},
        )
    )

    async def stream_example():
        print("Response: ", end="")
        async for event in agent.astream("What can you help me with?", context=ctx_stream):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
        print("\n")

    asyncio.run(stream_example())

    # --- Example 4: The prompt function is independently testable ---

    print("=" * 60)
    print("Example 4: Prompt function is testable without LLM")
    print("=" * 60)
    # Test with context
    test_ctx = RunContext(
        state=AppState(
            user_name="Test",
            company="TestCo",
            plan_tier="enterprise",
            open_tickets=4,
            feature_flags={"concise_mode": True},
        )
    )
    prompt = build_support_prompt(test_ctx)
    print(f"Resolved prompt:\n{prompt}\n")

    # Test without context
    prompt_none = build_support_prompt(None)
    print(f"Fallback prompt: {prompt_none}")
