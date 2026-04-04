"""Example 20: Output Type — Pydantic-parsed structured output on Agent.

Demonstrates using output_type to automatically parse LLM responses
into typed Pydantic models. No manual JSON parsing needed.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/20_output_type.py
"""

from pydantic import BaseModel

from fastaiagent import Agent, LLMClient


# --- Define your output models ---


class Person(BaseModel):
    name: str
    age: int
    occupation: str
    city: str


class Address(BaseModel):
    street: str
    city: str
    country: str


class Customer(BaseModel):
    name: str
    email: str
    address: Address
    is_active: bool


# --- Example 1: Simple output type ---


def example_simple():
    """Extract structured data with automatic parsing."""
    print("=== Example 1: Simple output_type ===\n")

    agent = Agent(
        name="extractor",
        system_prompt="Extract person information from the user's message.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        output_type=Person,
    )

    result = agent.run("Alice is a 30-year-old software engineer living in Tokyo.")

    # result.parsed is a typed Person instance
    print(f"Name: {result.parsed.name}")
    print(f"Age: {result.parsed.age}")
    print(f"Occupation: {result.parsed.occupation}")
    print(f"City: {result.parsed.city}")
    print(f"\nRaw JSON: {result.output}")
    print()


# --- Example 2: Nested models ---


def example_nested():
    """Nested Pydantic models work automatically."""
    print("=== Example 2: Nested models ===\n")

    agent = Agent(
        name="customer-extractor",
        system_prompt="Extract customer information from the user's message.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        output_type=Customer,
    )

    result = agent.run(
        "John Doe (john@example.com) is an active customer. "
        "He lives at 123 Main Street, San Francisco, USA."
    )

    print(f"Name: {result.parsed.name}")
    print(f"Email: {result.parsed.email}")
    print(f"Active: {result.parsed.is_active}")
    print(f"Address: {result.parsed.address.street}, "
          f"{result.parsed.address.city}, {result.parsed.address.country}")
    print()


# --- Example 3: Streaming with output_type ---


def example_streaming():
    """stream() collects tokens and parses at the end."""
    print("=== Example 3: Streaming with output_type ===\n")

    agent = Agent(
        name="extractor",
        system_prompt="Extract person information.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        output_type=Person,
    )

    result = agent.stream("Bob is a 45-year-old doctor from London.")

    print(f"Parsed: {result.parsed}")
    print(f"Type: {type(result.parsed).__name__}")
    print()


# --- Main ---


if __name__ == "__main__":
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
    else:
        example_simple()
        example_nested()
        example_streaming()
