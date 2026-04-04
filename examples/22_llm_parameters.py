"""Example 22: Additional LLM Parameters.

Demonstrates top_p, stop, seed, frequency_penalty, presence_penalty,
and parallel_tool_calls — with per-provider mapping.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/22_llm_parameters.py
"""

from fastaiagent import LLMClient
from fastaiagent.llm import UserMessage


# --- Example 1: Sampling parameters ---


def example_sampling():
    """Control sampling with top_p, temperature, and seed."""
    print("=== Example 1: Sampling parameters ===\n")

    llm = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.3,
        top_p=0.9,
        seed=42,  # Reproducible outputs (best effort)
    )

    response = llm.complete([UserMessage("Write a haiku about coding.")])
    print(f"Response: {response.content}")
    print()


# --- Example 2: Stop sequences ---


def example_stop_sequences():
    """Stop generation at specific tokens."""
    print("=== Example 2: Stop sequences ===\n")

    llm = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
        stop=["END", "\n\n"],  # Stop at these sequences
    )

    response = llm.complete([UserMessage("List 5 colors, one per line.")])
    print(f"Response: {response.content}")
    print()


# --- Example 3: Penalty parameters ---


def example_penalties():
    """Reduce repetition with frequency and presence penalties."""
    print("=== Example 3: Frequency and presence penalties ===\n")

    llm = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
        frequency_penalty=0.5,   # Penalize repeated tokens
        presence_penalty=0.3,    # Encourage new topics
    )

    response = llm.complete(
        [UserMessage("Write a paragraph about the importance of diversity in tech.")]
    )
    print(f"Response: {response.content}")
    print()


# --- Example 4: Per-call override ---


def example_per_call_override():
    """Override constructor defaults on a per-call basis."""
    print("=== Example 4: Per-call override ===\n")

    llm = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
    )

    # Override top_p for this specific call
    response = llm.complete(
        [UserMessage("Give me a creative name for a pet cat.")],
        top_p=0.5,  # More focused sampling for this call
    )
    print(f"Response (top_p=0.5): {response.content}")

    # Use constructor defaults
    response2 = llm.complete(
        [UserMessage("Give me a creative name for a pet dog.")],
    )
    print(f"Response (top_p=0.9): {response2.content}")
    print()


# --- Example 5: Serialization roundtrip ---


def example_serialization():
    """Parameters survive to_dict()/from_dict() roundtrip."""
    print("=== Example 5: Serialization ===\n")

    llm = LLMClient(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.5,
        top_p=0.9,
        seed=42,
        stop=["END"],
        frequency_penalty=0.3,
        max_retries=2,
    )

    d = llm.to_dict()
    print(f"Serialized: {d}")

    llm2 = LLMClient.from_dict(d)
    print(f"Restored top_p: {llm2.top_p}")
    print(f"Restored seed: {llm2.seed}")
    print(f"Restored max_retries: {llm2.max_retries}")
    print()


# --- Main ---


if __name__ == "__main__":
    import os

    example_serialization()  # No API key needed

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run live examples")
    else:
        example_sampling()
        example_stop_sequences()
        example_penalties()
        example_per_call_override()
