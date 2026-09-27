"""Example 30: Composable long-term memory.

Demonstrates the four shipped memory blocks layered on top of a sliding-window
primary memory:

  - StaticBlock          — a persistent fact the agent always sees
  - SummaryBlock         — a rolling LLM-generated summary of older turns
  - VectorBlock          — semantic recall over past messages (FAISS)
  - FactExtractionBlock  — durable facts distilled by a fast LLM

A 6-turn conversation walks the agent through establishing facts, referencing
them later, and confirms the blocks pull their weight: the primary window keeps
only the last exchange, so turns 5-6 can only be answered from the blocks.

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
from fastaiagent.kb.embedding import get_default_embedder


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

    # The vector store used by VectorBlock: FAISS in-process, zero setup. Size
    # it to the embedder actually in use (FastEmbed 384, OpenAI 1536, the
    # SimpleEmbedder fallback 128) — a mismatched index can't store anything.
    embedder = get_default_embedder()
    dimension = len(embedder.embed(["probe"])[0])
    vector_store = FaissVectorStore(dimension=dimension, index_type="flat")

    memory = ComposableMemory(
        blocks=[
            StaticBlock(
                "The current date is 2026-04-18. The user prefers concise answers."
            ),
            SummaryBlock(llm=llm, keep_last=4, summarize_every=3, max_chars=400),
            VectorBlock(store=vector_store, embedder=embedder, top_k=3, min_content_chars=15),
            # Facts come from what the user says, not from the model's replies.
            FactExtractionBlock(llm=llm, max_facts=50, extract_every=1, roles=("user",)),
        ],
        # Only the last exchange is kept verbatim. By turns 5-6 the name, the
        # job and the dog have left the window, so the answers have to come
        # from the blocks — which is what this example shows.
        primary=AgentMemory(max_messages=2),
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
        elif isinstance(block, VectorBlock):
            print(f"{name}: {vector_store.count()} past messages indexed for recall")
        else:
            print(f"{name}: (no state)")


if __name__ == "__main__":
    main()
