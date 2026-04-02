"""Example 13: Structured output (response_format).

Demonstrates three response_format types across providers:
  1. json_object  — any valid JSON
  2. json_schema  — JSON matching a specific schema
  3. Multi-provider — same code works with OpenAI, Anthropic, Ollama

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/13_structured_output.py
"""

import json

from fastaiagent import LLMClient
from fastaiagent.llm import UserMessage


# --- Example 1: JSON Object mode ---


def example_json_object():
    """Force the LLM to respond with valid JSON."""
    print("=== Example 1: json_object mode ===\n")

    llm = LLMClient(provider="openai", model="gpt-4.1")
    response = llm.complete(
        [UserMessage("List 3 programming languages with their year of creation. Respond as JSON.")],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.content)
    print(json.dumps(data, indent=2))
    print()


# --- Example 2: JSON Schema mode ---


def example_json_schema():
    """Force the LLM to respond with JSON matching a schema."""
    print("=== Example 2: json_schema mode ===\n")

    llm = LLMClient(provider="openai", model="gpt-4.1")
    response = llm.complete(
        [UserMessage("Describe the city of Tokyo")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "city_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "country": {"type": "string"},
                        "population": {"type": "integer"},
                        "famous_for": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "country", "population", "famous_for"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )

    city = json.loads(response.content)
    print(f"City: {city['name']}")
    print(f"Country: {city['country']}")
    print(f"Population: {city['population']:,}")
    print(f"Famous for: {', '.join(city['famous_for'])}")
    print()


# --- Example 3: Data extraction with Agent ---


def example_agent_structured():
    """Use structured output with an Agent for data extraction."""
    print("=== Example 3: Agent with structured output ===\n")

    from fastaiagent import Agent

    agent = Agent(
        name="extractor",
        system_prompt="Extract structured data from the user's message.",
        llm=LLMClient(provider="openai", model="gpt-4.1"),
    )

    result = agent.run(
        "Alice Smith is a 28-year-old software engineer from San Francisco. "
        "She speaks English and Japanese.",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "occupation": {"type": "string"},
                        "city": {"type": "string"},
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "age", "occupation", "city", "languages"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )

    person = json.loads(result.output)
    print(f"Name: {person['name']}")
    print(f"Age: {person['age']}")
    print(f"Occupation: {person['occupation']}")
    print(f"City: {person['city']}")
    print(f"Languages: {', '.join(person['languages'])}")
    print()


# --- Example 4: Anthropic JSON Object mode ---


def example_anthropic_json_object():
    """Force Anthropic to respond with valid JSON."""
    print("=== Example 4: Anthropic json_object mode ===\n")

    llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    response = llm.complete(
        [UserMessage("List 3 programming languages with their year of creation. Respond as JSON.")],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.content)
    print(json.dumps(data, indent=2))
    print()


# --- Example 5: Anthropic JSON Schema mode ---


def example_anthropic_json_schema():
    """Force Anthropic to respond with JSON matching a schema."""
    print("=== Example 5: Anthropic json_schema mode ===\n")

    llm = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    response = llm.complete(
        [UserMessage("Describe the city of Tokyo")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "city_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "country": {"type": "string"},
                        "population": {"type": "integer"},
                        "famous_for": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["name", "country", "population", "famous_for"],
                },
            },
        },
    )

    city = json.loads(response.content)
    print(f"City: {city['name']}")
    print(f"Country: {city['country']}")
    print(f"Population: {city['population']:,}")
    print(f"Famous for: {', '.join(city['famous_for'])}")
    print()


# --- Main ---


if __name__ == "__main__":
    import os

    # --- OpenAI tests ---
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping OpenAI examples: OPENAI_API_KEY not set\n")
    else:
        print("=" * 50)
        print("  OPENAI PROVIDER TESTS")
        print("=" * 50 + "\n")
        example_json_object()
        example_json_schema()
        example_agent_structured()

    # --- Anthropic tests ---
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Skipping Anthropic examples: ANTHROPIC_API_KEY not set\n")
    else:
        print("=" * 50)
        print("  ANTHROPIC PROVIDER TESTS")
        print("=" * 50 + "\n")
        example_anthropic_json_object()
        example_anthropic_json_schema()
