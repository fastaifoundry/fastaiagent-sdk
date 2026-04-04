"""Example 23: Tool-Position Guardrails.

Demonstrates guardrails at tool_call and tool_result positions —
validating tool arguments before execution and tool output after.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/23_tool_guardrails.py
"""

from fastaiagent import Agent, LLMClient
from fastaiagent._internal.errors import GuardrailBlockedError
from fastaiagent.guardrail import Guardrail, GuardrailPosition, allowed_domains
from fastaiagent.tool import tool


# --- Example tools ---


@tool()
def search_web(url: str) -> str:
    """Fetch data from a URL."""
    return f"Data from {url}: lorem ipsum..."


@tool()
def get_user_data(user_id: str) -> str:
    """Look up user data by ID."""
    # Simulated — in reality this might return sensitive data
    return f"User {user_id}: name=Alice, email=alice@example.com, SSN=123-45-6789"


# --- Example 1: allowed_domains() built-in ---


def example_allowed_domains():
    """Restrict tool calls to approved domains only."""
    print("=== Example 1: allowed_domains() guardrail ===\n")

    agent = Agent(
        name="safe-agent",
        system_prompt="You can search the web for information.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[search_web],
        guardrails=[
            allowed_domains(["api.mycompany.com", "docs.mycompany.com"]),
        ],
    )

    # This would be blocked if the LLM tries to call search_web
    # with a URL outside the allowed domains
    try:
        result = agent.run("Search https://api.mycompany.com/data for user info")
        print(f"Result: {result.output}")
    except GuardrailBlockedError as e:
        print(f"Blocked: {e}")
    print()


# --- Example 2: Custom tool_call guardrail ---


def example_tool_call_guardrail():
    """Validate tool arguments before execution."""
    print("=== Example 2: Custom tool_call guardrail ===\n")

    # Block tool calls that contain PII patterns
    no_pii_in_args = Guardrail(
        name="no-pii-in-tool-args",
        position=GuardrailPosition.tool_call,
        blocking=True,
        fn=lambda text: "ssn" not in text.lower() and "password" not in text.lower(),
    )

    agent = Agent(
        name="data-agent",
        system_prompt="Help the user look up information.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[get_user_data],
        guardrails=[no_pii_in_args],
    )

    try:
        result = agent.run("Look up user U-123")
        print(f"Result: {result.output}")
    except GuardrailBlockedError as e:
        print(f"Tool call blocked: {e}")
    print()


# --- Example 3: Custom tool_result guardrail ---


def example_tool_result_guardrail():
    """Validate tool output after execution — catch sensitive data."""
    print("=== Example 3: Custom tool_result guardrail ===\n")

    # Block tool results that contain SSN patterns
    no_ssn_in_results = Guardrail(
        name="no-ssn-in-results",
        position=GuardrailPosition.tool_result,
        blocking=True,
        fn=lambda text: "SSN" not in text and "ssn" not in text.lower(),
    )

    agent = Agent(
        name="data-agent",
        system_prompt="Help the user look up information.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[get_user_data],
        guardrails=[no_ssn_in_results],
    )

    try:
        result = agent.run("Look up user U-123")
        print(f"Result: {result.output}")
    except GuardrailBlockedError as e:
        print(f"Tool result blocked: {e}")
        print("(The tool returned sensitive data that was caught by the guardrail)")
    print()


# --- Example 4: All four positions ---


def example_all_positions():
    """Guardrails at input, tool_call, tool_result, and output positions."""
    print("=== Example 4: All four guardrail positions ===\n")

    guards = [
        Guardrail(
            name="input-check",
            position=GuardrailPosition.input,
            blocking=True,
            fn=lambda text: len(text) < 1000,
            description="Block overly long inputs",
        ),
        Guardrail(
            name="tool-call-audit",
            position=GuardrailPosition.tool_call,
            blocking=False,  # Non-blocking — just logs
            fn=lambda text: (print(f"  [audit] Tool call: {text[:80]}..."), True)[1],
        ),
        Guardrail(
            name="tool-result-check",
            position=GuardrailPosition.tool_result,
            blocking=True,
            fn=lambda text: "ERROR" not in text,
            description="Block error responses from tools",
        ),
        Guardrail(
            name="output-length",
            position=GuardrailPosition.output,
            blocking=True,
            fn=lambda text: len(text) < 5000,
            description="Block overly long outputs",
        ),
    ]

    agent = Agent(
        name="guarded-agent",
        system_prompt="Help the user with searches.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[search_web],
        guardrails=guards,
    )

    try:
        result = agent.run("Search https://docs.example.com for Python tutorials")
        print(f"Result: {result.output}")
    except GuardrailBlockedError as e:
        print(f"Blocked by {e.guardrail_name}: {e}")
    print()


# --- Main ---


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
    else:
        example_allowed_domains()
        example_tool_call_guardrail()
        example_tool_result_guardrail()
        example_all_positions()
