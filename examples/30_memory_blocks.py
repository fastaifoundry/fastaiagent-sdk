"""Example 30: Composable long-term memory.

Demonstrates the four shipped memory blocks layered on top of a sliding-window
primary memory:

  - StaticBlock          — a persistent fact the agent always sees
  - SummaryBlock         — a rolling LLM-generated summary of older turns
  - VectorBlock          — semantic recall over past messages (FAISS)
  - FactExtractionBlock  — durable facts distilled by a fast LLM

A 6-turn conversation walks the agent through establishing facts, referencing
them later, and confirms the blocks pull their weight.

Install:
    pip install 'fastaiagent[kb]'   # for FAISS + fastembed

Usage:
    export OPENAI_API_KEY=sk-...     # or ANTHROPIC_API_KEY
    python examples/30_memory_blocks.py
"""

from __future__ import annotations

import os
import sys

from fastaiagent import (
    Agent,
    AgentMemory,
    ComposableMemory,
    FactExtractionBlock,
    LLMClient,
    StaticBlock,
    SummaryBlock,
    VectorBlock,
)
from fastaiagent.kb.backends.faiss import FaissVectorStore


def _pick_llm() -> LLMClient:
    if os.environ.get("OPENAI_API_KEY"):
        print("Using OpenAI gpt-4o-mini\n")
        return LLMClient(provider="openai", model="gpt-4o-mini")
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("Using Anthropic claude-haiku-4-5-20251001\n")
        return LLMClient(provider="anthropic", model="claude-haiku-4-5-20251001")
    print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this example.")
    sys.exit(1)


def main() -> None:
    llm = _pick_llm()

    # The vector store used by VectorBlock. FAISS in-process; zero setup.
    # Dimension matches the SimpleEmbedder's default output (128), which the
    # VectorBlock will use as the auto-selected embedder when no API key is
    # set for FastEmbed/OpenAI. In practice prefer a real embedder for real
    # semantic matching — see docs/knowledge-base/backends.md.
    vector_store = FaissVectorStore(dimension=384, index_type="flat")

    memory = ComposableMemory(
        blocks=[
            StaticBlock(
                "The current date is 2026-04-18. The user prefers concise answers."
            ),
            SummaryBlock(llm=llm, keep_last=4, summarize_every=3, max_chars=400),
            VectorBlock(store=vector_store, top_k=3, min_content_chars=15),
            FactExtractionBlock(llm=llm, max_facts=50, extract_every=1),
        ],
        primary=AgentMemory(max_messages=10),
    )

    agent = Agent(
        name="memory-demo",
        system_prompt=(
            "You are a helpful assistant with long-term memory. Use the "
            "pinned system-level facts, the running summary, and the known "
            "facts list when answering. Stay concise."
        ),
        llm=llm,
        memory=memory,
    )

    turns = [
        "Hi! My name is Casey and I live in Amsterdam.",
        "I work as a civil engineer specializing in bridge design.",
        "My dog's name is Pepper. She's a border collie who loves fetch.",
        "I'm planning a trip to Lisbon next month.",
        "Given what you know about me, what local activities might I enjoy?",
        "What have we discussed about my dog?",
    ]

    for i, user_input in enumerate(turns, start=1):
        print(f"--- Turn {i} ---")
        print(f"User: {user_input}")
        result = agent.run(user_input)
        print(f"Agent: {result.output}\n")

    # Inspect what the blocks captured.
    print("--- Block State ---")
    for block in memory.blocks:
        name = block.name or type(block).__name__
        if hasattr(block, "_facts") and block._facts:
            print(f"{name}: {len(block._facts)} facts — {block._facts[:3]}...")
        elif hasattr(block, "_summary") and block._summary:
            print(f"{name}: summary = {block._summary[:120]}...")
        elif hasattr(block, "text"):
            print(f"{name}: static = {block.text}")
        else:
            print(f"{name}: (stateful block — see store for content)")


if __name__ == "__main__":
    main()
