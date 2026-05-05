"""
Memory setup — wire all four memory-block types together.

The personal-assistant template is the canonical demo for ``ComposableMemory``.
Every turn the agent's prompt is composed of these layers (in order):

  1. ``StaticBlock``        — pinned identity. ``today is 2026-05-06; user
                              is Bob, software engineer, in PST``. Cheapest
                              possible memory layer.
  2. ``SummaryBlock``       — rolling executive summary of older turns.
                              Refreshed every N messages by an LLM call. Cuts
                              context-window cost in long sessions.
  3. ``VectorBlock``        — semantic recall over every prior message. On
                              each turn the user's query is embedded and the
                              top-k similar past turns are pulled in as
                              ``Relevant prior exchanges:`` system fragments.
  4. ``FactExtractionBlock`` — every turn an LLM extracts durable facts
                              ("user has a 6-month-old daughter named Mira",
                              "user prefers vim over emacs") and persists
                              them to a deduplicated list rendered as bullets.

Plus the primary sliding-window ``AgentMemory`` carries the literal last 20
messages.

All four block types are saved + loaded across REPL sessions so the
assistant genuinely *learns* about you over time.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import fastaiagent as fa
from fastaiagent.agent.memory import AgentMemory, ComposableMemory
from fastaiagent.agent.memory_blocks import (
    FactExtractionBlock,
    StaticBlock,
    SummaryBlock,
    VectorBlock,
)


def _make_vector_store():
    """Build the VectorBlock's backing store.

    Uses ``FaissVectorStore`` (declared in requirements.txt as ``faiss-cpu``).
    384 matches the default fastembed sentence-transformer model AND the
    SDK's ``SimpleEmbedder`` hash-based fallback — both produce 384-dim
    vectors so swapping embedders doesn't require changing the store.

    For long-running deployments swap to Chroma or Qdrant — same protocol,
    swap the import:
        from fastaiagent.kb.backends.qdrant import QdrantVectorStore
        return QdrantVectorStore(host="localhost", port=6333, collection="pa-memory")
    """
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    return FaissVectorStore(dimension=384, index_type="flat")


def build_identity_text() -> str:
    """Compose the StaticBlock text from .env settings + today's date.

    Rebuilt on every REPL startup so ``today is X`` actually advances. If
    you want the LLM to know the current time too, add a ``current_time``
    tool — pinning a string here that goes stale at midnight is worse
    than no answer.
    """
    return (
        f"User identity (pinned every turn):\n"
        f"  Name: {os.getenv('USER_NAME', 'Friend')}\n"
        f"  Role: {os.getenv('USER_ROLE', 'Software engineer')}\n"
        f"  Timezone: {os.getenv('USER_TZ', 'UTC')}\n"
        f"  Today's date: {date.today().isoformat()}\n"
        f"\nUse the user's name when natural; respect their timezone when "
        f"discussing schedules; default to their primary tools/skills based "
        f"on their role unless they say otherwise."
    )


def build_memory(
    *,
    memory_dir: Path | None = None,
    main_llm: fa.LLMClient | None = None,
    cheap_llm: fa.LLMClient | None = None,
) -> ComposableMemory:
    """Build a fresh ``ComposableMemory`` with all four block types wired.

    ``memory_dir`` (when provided AND existing) populates each block from
    its previously-saved state — so a restarted REPL picks up where it
    left off. The agent's caller is responsible for calling
    :meth:`ComposableMemory.save(memory_dir)` before exit; we don't
    register an atexit hook here because tests / scripts may not want it.
    """
    main_llm = main_llm or fa.LLMClient(
        provider="openai", model=os.getenv("LLM_MODEL", "gpt-4o")
    )
    cheap_llm = cheap_llm or fa.LLMClient(
        provider="openai", model=os.getenv("MEMORY_LLM_MODEL", "gpt-4o-mini")
    )

    static = StaticBlock(text=build_identity_text(), name="static")

    summary = SummaryBlock(
        llm=cheap_llm,
        keep_last=6,           # don't summarise the last 6 messages
        summarize_every=4,     # refresh summary every 4 messages seen
        max_chars=600,         # soft cap on summary length
    )

    vector = VectorBlock(
        store=_make_vector_store(),
        top_k=5,
        namespace="default",
        min_content_chars=12,  # skip "ok" / "thanks" / etc.
    )

    facts = FactExtractionBlock(
        llm=cheap_llm,
        max_facts=120,
        extract_every=1,
    )

    memory = ComposableMemory(
        blocks=[static, summary, vector, facts],
        primary=AgentMemory(max_messages=20),
    )

    if memory_dir is not None and memory_dir.exists():
        memory.load(memory_dir)

    return memory


def save_memory(memory: ComposableMemory, memory_dir: Path) -> None:
    """Persist the primary window AND every block's state to disk.

    Called by the CLI on graceful exit. Each block writes its own JSON
    inside ``memory_dir/blocks/``; the primary window goes to
    ``memory_dir/primary.json``.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    memory.save(memory_dir)
