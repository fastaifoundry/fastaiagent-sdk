"""Example 18: Supervisor/Worker teams with context, streaming, and dynamic instructions.

Demonstrates:
  1. Basic supervisor/worker delegation
  2. RunContext flowing from supervisor to worker tools
  3. Dynamic instructions (callable system_prompt) on supervisor
  4. Streaming supervisor output

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/18_supervisor_worker.py
"""

import asyncio
from dataclasses import dataclass

from fastaiagent import Agent, LLMClient, RunContext, Supervisor, TextDelta, Worker, tool
from fastaiagent.llm.stream import ToolCallEnd, ToolCallStart


# --- Shared state ---


class FakeDB:
    """Simulates a database."""

    def __init__(self):
        self._tickets = {
            "u-1": [
                {"id": "T-100", "status": "open", "subject": "Login issue"},
                {"id": "T-101", "status": "closed", "subject": "Billing question"},
            ],
            "u-2": [
                {"id": "T-200", "status": "open", "subject": "Feature request"},
            ],
        }
        self._billing = {
            "u-1": {"plan": "enterprise", "balance": "$12,500.00", "next_invoice": "2026-05-01"},
            "u-2": {"plan": "starter", "balance": "$49.99", "next_invoice": "2026-04-15"},
        }

    def get_tickets(self, user_id: str, status: str | None = None) -> list[dict]:
        tickets = self._tickets.get(user_id, [])
        if status:
            tickets = [t for t in tickets if t["status"] == status]
        return tickets

    def get_billing(self, user_id: str) -> dict | None:
        return self._billing.get(user_id)


@dataclass
class AppState:
    db: FakeDB
    user_id: str
    company: str
    plan: str


# --- Worker tools ---


@tool(name="get_tickets")
def get_tickets(ctx: RunContext[AppState], status: str = "") -> str:
    """Get support tickets for the current user, optionally filtered by status."""
    tickets = ctx.state.db.get_tickets(ctx.state.user_id, status or None)
    if not tickets:
        return f"No tickets found for user {ctx.state.user_id}."
    lines = [f"- [{t['id']}] {t['subject']} ({t['status']})" for t in tickets]
    return f"Tickets for {ctx.state.user_id}:\n" + "\n".join(lines)


@tool(name="get_billing")
def get_billing(ctx: RunContext[AppState]) -> str:
    """Get billing info for the current user."""
    info = ctx.state.db.get_billing(ctx.state.user_id)
    if not info:
        return f"No billing info found for {ctx.state.user_id}."
    return f"Billing for {ctx.state.user_id}: plan={info['plan']}, balance={info['balance']}, next_invoice={info['next_invoice']}"


@tool(name="calculate")
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic arithmetic is allowed."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# --- Build workers ---

llm = LLMClient(provider="openai", model="gpt-4o-mini")

support_agent = Agent(
    name="support",
    system_prompt="You handle support tickets. Use get_tickets to find tickets.",
    llm=llm,
    tools=[get_tickets],
)

billing_agent = Agent(
    name="billing",
    system_prompt="You handle billing queries. Use get_billing and calculate tools.",
    llm=llm,
    tools=[get_billing, calculate],
)

support_worker = Worker(agent=support_agent, role="support", description="Manages support tickets")
billing_worker = Worker(agent=billing_agent, role="billing", description="Handles billing queries")


# --- Example 1: Basic delegation with context ---


def example_basic_delegation():
    """Supervisor delegates to workers, context flows to tools."""
    print("=== Example 1: Basic delegation with context ===\n")

    supervisor = Supervisor(
        name="customer-service",
        llm=llm,
        workers=[support_worker, billing_worker],
    )

    ctx = RunContext(state=AppState(
        db=FakeDB(), user_id="u-1", company="Acme Corp", plan="enterprise",
    ))

    result = supervisor.run("Show my open tickets and billing info", context=ctx)
    print(f"Output:\n{result.output}")
    print(f"\nTool calls: {[tc['tool_name'] for tc in result.tool_calls]}")
    print(f"Latency: {result.latency_ms}ms")


# --- Example 2: Dynamic instructions ---


def example_dynamic_instructions():
    """Callable system_prompt customizes behavior per request."""
    print("\n=== Example 2: Dynamic instructions ===\n")

    supervisor = Supervisor(
        name="adaptive-lead",
        llm=llm,
        workers=[support_worker, billing_worker],
        system_prompt=lambda ctx: (
            f"You are the customer service lead for {ctx.state.company}. "
            f"The customer ({ctx.state.user_id}) is on the {ctx.state.plan} plan.\n"
            + ("PRIORITY: Enterprise customer. Be thorough and proactive.\n"
               if ctx.state.plan == "enterprise" else "")
            + "Delegate to workers and synthesize a helpful response."
        ),
    )

    ctx = RunContext(state=AppState(
        db=FakeDB(), user_id="u-1", company="Acme Corp", plan="enterprise",
    ))
    result = supervisor.run("I need help with everything", context=ctx)
    print(f"Output:\n{result.output}")


# --- Example 3: Streaming ---


async def example_streaming():
    """Stream the supervisor's response in real-time."""
    print("\n=== Example 3: Streaming ===\n")

    supervisor = Supervisor(
        name="stream-lead",
        llm=llm,
        workers=[support_worker, billing_worker],
    )

    ctx = RunContext(state=AppState(
        db=FakeDB(), user_id="u-2", company="StartupCo", plan="starter",
    ))

    print("Assistant: ", end="", flush=True)
    async for event in supervisor.astream("What are my open tickets?", context=ctx):
        if isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallStart):
            print(f"\n  [Delegating to {event.tool_name}...]", end="", flush=True)
        elif isinstance(event, ToolCallEnd):
            print(" [done]", end="", flush=True)
    print("\n")


# --- Example 4: Sync streaming ---


def example_sync_streaming():
    """Sync stream() collects output into AgentResult."""
    print("\n=== Example 4: Sync streaming ===\n")

    supervisor = Supervisor(
        name="sync-lead",
        llm=llm,
        workers=[support_worker, billing_worker],
    )

    ctx = RunContext(state=AppState(
        db=FakeDB(), user_id="u-1", company="Acme Corp", plan="enterprise",
    ))

    result = supervisor.stream("Show my billing info", context=ctx)
    print(f"Output: {result.output}")
    print(f"Latency: {result.latency_ms}ms")


# --- Main ---


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY=sk-... && python examples/18_supervisor_worker.py")
        raise SystemExit(0)

    example_basic_delegation()
    example_dynamic_instructions()
    asyncio.run(example_streaming())
    example_sync_streaming()

    print(
        "\nTo render the supervisor topology in the Local UI, register the "
        "Supervisor with build_app:\n"
        "    from fastaiagent.ui.server import build_app\n"
        "    app = build_app(runners=[supervisor])\n"
        "Then visit http://127.0.0.1:7843/workflows/supervisor/customer-service"
    )
