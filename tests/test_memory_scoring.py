"""Tests for v1.9.0 memory scoring (recency + importance) on
``VectorBlock`` and ``PersistentFactBlock``.

Goals:
  - Default behaviour (both weights = 0) is byte-identical to today.
  - With ``recency_weight > 0``, newer chunks at the same similarity rank
    higher.
  - With ``importance_weight > 0``, high-importance chunks rank higher.
  - Validation guards reject negative weights and weights summing > 1.
"""

from __future__ import annotations

import time

import pytest

from fastaiagent.agent.memory_blocks import PersistentFactBlock, VectorBlock

# ---------------------------------------------------------------------------
# Tiny in-memory VectorStore + Embedder stand-ins
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Returns a deterministic 4-d vector based on the text's first letter.

    Used so tests can construct chunks with chosen similarity values
    without depending on a real embedding model.
    """

    def embed(self, texts):
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


class _FakeVectorStore:
    """Returns canned (chunk, similarity) hits in the order added."""

    def __init__(self, hits):
        self._hits = hits
        self.added = []

    def add(self, chunks, embeddings):
        self.added.append((chunks, embeddings))

    def search(self, embedding, top_k):
        return list(self._hits)[:top_k]


def _chunk(content: str, namespace: str, *, created_at=None, importance=None):
    from fastaiagent.kb.chunking import Chunk

    metadata = {"namespace": namespace}
    if created_at is not None:
        metadata["created_at"] = created_at
    if importance is not None:
        metadata["importance"] = importance
    return Chunk(
        id=content,
        content=content,
        metadata=metadata,
        index=0,
        start_char=0,
        end_char=len(content),
    )


# ---------------------------------------------------------------------------
# VectorBlock — defaults preserve today's behaviour
# ---------------------------------------------------------------------------


def test_vectorblock_defaults_preserve_input_order() -> None:
    """With both weights = 0, scoring must be a no-op — input order returned."""
    now = time.time()
    hits = [
        (_chunk("old-high-sim", "default", created_at=now - 7 * 86400), 0.9),
        (_chunk("recent-low-sim", "default", created_at=now - 60), 0.5),
    ]
    block = VectorBlock(
        store=_FakeVectorStore(hits),
        embedder=_FakeEmbedder(),
        top_k=2,
    )
    msgs = block.render("query")
    body = msgs[0].content
    # Old-high-sim must come first, just like today.
    assert body.find("old-high-sim") < body.find("recent-low-sim")


# ---------------------------------------------------------------------------
# VectorBlock — recency_weight reorders
# ---------------------------------------------------------------------------


def test_vectorblock_recency_weight_promotes_recent() -> None:
    now = time.time()
    hits = [
        # Same similarity, different ages.
        (_chunk("old", "default", created_at=now - 7 * 86400), 0.7),
        (_chunk("fresh", "default", created_at=now - 30), 0.7),
    ]
    block = VectorBlock(
        store=_FakeVectorStore(hits),
        embedder=_FakeEmbedder(),
        top_k=2,
        recency_weight=0.5,
        recency_half_life_seconds=3600.0,
    )
    body = block.render("q")[0].content
    assert body.find("fresh") < body.find("old")


# ---------------------------------------------------------------------------
# VectorBlock — importance_weight reorders
# ---------------------------------------------------------------------------


def test_vectorblock_importance_weight_promotes_important() -> None:
    now = time.time()
    hits = [
        (_chunk("low-imp", "default", created_at=now, importance=0.1), 0.7),
        (_chunk("high-imp", "default", created_at=now, importance=1.0), 0.7),
    ]
    block = VectorBlock(
        store=_FakeVectorStore(hits),
        embedder=_FakeEmbedder(),
        top_k=2,
        importance_weight=0.5,
    )
    body = block.render("q")[0].content
    assert body.find("high-imp") < body.find("low-imp")


# ---------------------------------------------------------------------------
# VectorBlock — old correct vs new correct (the canonical scenario)
# ---------------------------------------------------------------------------


def test_vectorblock_new_correct_outranks_old_correct() -> None:
    """The headline failure mode the scoring fix targets:
    same-similarity old vs. new, recency tilts the right way."""
    now = time.time()
    hits = [
        (_chunk("alice@old.com", "default", created_at=now - 7 * 86400), 0.85),
        (_chunk("alice@new.com", "default", created_at=now - 3600), 0.80),
    ]
    block = VectorBlock(
        store=_FakeVectorStore(hits),
        embedder=_FakeEmbedder(),
        top_k=2,
        recency_weight=0.3,
        recency_half_life_seconds=3600.0,
    )
    body = block.render("what's my email?")[0].content
    assert body.find("alice@new.com") < body.find("alice@old.com")


# ---------------------------------------------------------------------------
# VectorBlock — chunk metadata records created_at + optional importance
# ---------------------------------------------------------------------------


def test_vectorblock_make_chunk_records_created_at() -> None:
    from fastaiagent.llm.message import UserMessage

    block = VectorBlock(
        store=_FakeVectorStore([]),
        embedder=_FakeEmbedder(),
        top_k=1,
    )
    msg = UserMessage("This is a moderately long message for indexing.")
    chunk = block._make_chunk(msg)
    assert chunk is not None
    assert "created_at" in chunk.metadata
    assert isinstance(chunk.metadata["created_at"], float)


# ---------------------------------------------------------------------------
# VectorBlock — guards
# ---------------------------------------------------------------------------


def test_vectorblock_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        VectorBlock(
            store=_FakeVectorStore([]),
            embedder=_FakeEmbedder(),
            recency_weight=-0.1,
        )


def test_vectorblock_weights_summing_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="not exceed 1.0"):
        VectorBlock(
            store=_FakeVectorStore([]),
            embedder=_FakeEmbedder(),
            recency_weight=0.7,
            importance_weight=0.5,
        )


def test_vectorblock_zero_half_life_rejected() -> None:
    with pytest.raises(ValueError, match="> 0"):
        VectorBlock(
            store=_FakeVectorStore([]),
            embedder=_FakeEmbedder(),
            recency_half_life_seconds=0.0,
        )


# ---------------------------------------------------------------------------
# PersistentFactBlock — defaults preserve newest-first
# ---------------------------------------------------------------------------


class _StubMemoryStore:
    """Mimics MemoryStore.list_active — returns the canned facts list."""

    def __init__(self, facts):
        self._facts = facts

    def list_active(self, **_kwargs):
        return list(self._facts)


def _fact(text: str, *, created_at: float, confidence: float = 1.0):
    from fastaiagent.learn.store import Fact

    return Fact(
        scope="agent",
        scope_id="x",
        fact=text,
        created_at=created_at,
        confidence=confidence,
    )


def test_persistent_facts_default_preserves_newest_first() -> None:
    now = time.time()
    facts = [
        _fact("newest fact", created_at=now),
        _fact("oldest fact", created_at=now - 30 * 86400),
    ]
    block = PersistentFactBlock(
        scope="agent",
        scope_id="x",
        store=_StubMemoryStore(facts),
    )
    body = block.render("q")[0].content
    assert body.find("newest fact") < body.find("oldest fact")


def test_persistent_facts_recency_weight_demotes_old() -> None:
    now = time.time()
    facts = [
        # list_active normally returns newest-first. Build the list so
        # that without recency the order would be old-then-new — proving
        # the scorer reorders correctly.
        _fact("old-but-listed-first", created_at=now - 30 * 86400),
        _fact("recent", created_at=now - 60),
    ]
    block = PersistentFactBlock(
        scope="agent",
        scope_id="x",
        store=_StubMemoryStore(facts),
        recency_weight=0.6,
        recency_half_life_seconds=86400.0,
    )
    body = block.render("q")[0].content
    assert body.find("recent") < body.find("old-but-listed-first")


def test_persistent_facts_importance_weight_promotes_high_confidence() -> None:
    now = time.time()
    facts = [
        _fact("low-conf", created_at=now, confidence=0.2),
        _fact("high-conf", created_at=now, confidence=1.0),
    ]
    block = PersistentFactBlock(
        scope="agent",
        scope_id="x",
        store=_StubMemoryStore(facts),
        importance_weight=0.6,
    )
    body = block.render("q")[0].content
    assert body.find("high-conf") < body.find("low-conf")


# ---------------------------------------------------------------------------
# PersistentFactBlock — guards
# ---------------------------------------------------------------------------


def test_persistent_facts_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        PersistentFactBlock(scope="agent", scope_id="x", recency_weight=-0.1)


def test_persistent_facts_weights_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="not exceed 1.0"):
        PersistentFactBlock(
            scope="agent",
            scope_id="x",
            recency_weight=0.7,
            importance_weight=0.5,
        )
