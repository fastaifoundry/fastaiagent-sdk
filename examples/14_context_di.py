"""Example 14: Context & Dependency Injection with RunContext.

Shows how to pass runtime dependencies (DB, user sessions, config)
to tools cleanly using RunContext — no globals, no closures.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/14_context_di.py
"""

from dataclasses import dataclass

from fastaiagent import Agent, LLMClient, RunContext, tool


# --- Define your runtime dependencies ---


class FakeDB:
    """Simulates a database client."""

    def __init__(self):
        self._data = {
            "C-100": {"name": "Alice Johnson", "plan": "Enterprise", "balance": 1250.00},
            "C-200": {"name": "Bob Smith", "plan": "Starter", "balance": 49.99},
            "C-300": {"name": "Carol Lee", "plan": "Pro", "balance": 399.00},
        }

    def get_customer(self, customer_id: str) -> dict | None:
        return self._data.get(customer_id)

    def list_customers(self) -> list[str]:
        return list(self._data.keys())


@dataclass
class AppState:
    db: FakeDB
    current_user: str
    permissions: list[str]


# --- Define tools that use RunContext ---


@tool(name="get_customer")
def get_customer(ctx: RunContext[AppState], customer_id: str) -> str:
    """Look up a customer by ID."""
    if "customer:read" not in ctx.state.permissions:
        return "Error: You don't have permission to read customer data."
    customer = ctx.state.db.get_customer(customer_id)
    if customer is None:
        return f"Customer {customer_id} not found."
    return f"Customer {customer_id}: {customer}"


@tool(name="list_customers")
def list_customers(ctx: RunContext[AppState]) -> str:
    """List all customer IDs."""
    if "customer:read" not in ctx.state.permissions:
        return "Error: You don't have permission to list customers."
    ids = ctx.state.db.list_customers()
    return f"Customer IDs: {', '.join(ids)}"


@tool(name="whoami")
def whoami(ctx: RunContext[AppState]) -> str:
    """Get the current user's identity."""
    return f"You are: {ctx.state.current_user} (permissions: {ctx.state.permissions})"


# --- A plain tool (no context needed) ---


@tool(name="calculate")
def calculate(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: Only basic arithmetic is allowed."
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# --- Build the agent ---

agent = Agent(
    name="support-agent",
    system_prompt=(
        "You are a customer support agent. "
        "Use the available tools to help the user. Be concise."
    ),
    llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    tools=[get_customer, list_customers, whoami, calculate],
)


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping: OPENAI_API_KEY not set")
        print("Run: export OPENAI_API_KEY=sk-... && python examples/14_context_di.py")
        raise SystemExit(0)

    # Create context with runtime deps — each request gets its own
    ctx = RunContext(
        state=AppState(
            db=FakeDB(),
            current_user="agent@company.com",
            permissions=["customer:read", "customer:update"],
        )
    )

    print("=" * 60)
    print("Example 1: Tool with context + LLM args")
    print("=" * 60)
    result = agent.run("Look up customer C-100", context=ctx)
    print(f"Output: {result.output}")
    print(f"Tool calls: {[tc['tool_name'] for tc in result.tool_calls]}")

    print()
    print("=" * 60)
    print("Example 2: Context-only tool (no LLM args)")
    print("=" * 60)
    result = agent.run("Who am I?", context=ctx)
    print(f"Output: {result.output}")

    print()
    print("=" * 60)
    print("Example 3: Mix of context and non-context tools")
    print("=" * 60)
    result = agent.run(
        "List all customers and also calculate 1250 + 49.99 + 399",
        context=ctx,
    )
    print(f"Output: {result.output}")
    print(f"Tool calls: {[tc['tool_name'] for tc in result.tool_calls]}")

    print()
    print("=" * 60)
    print("Example 4: No context needed — backward compatible")
    print("=" * 60)
    result = agent.run("What is 17 * 23?")
    print(f"Output: {result.output}")

    print()
    print("=" * 60)
    print("Serialization check: RunContext never appears in to_dict()")
    print("=" * 60)
    d = agent.to_dict()
    print(f"Agent dict keys: {list(d.keys())}")
    print(f"Tool schemas: {[t['name'] for t in d['tools']]}")
    for t in d["tools"]:
        props = list(t.get("parameters", {}).get("properties", {}).keys())
        print(f"  {t['name']} params: {props}  (no 'ctx' — correct!)")
