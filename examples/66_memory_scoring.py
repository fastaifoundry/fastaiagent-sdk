"""Example 66: Memory recency + importance scoring on VectorBlock.

Long-running agents often want recent or important memories to outrank
stale-but-similar ones. This example shows the canonical "old correct
answer vs new correct answer" failure mode and how the scoring fix
flips it.

Runnable as pytest (no API keys, no network):
    pytest examples/66_memory_scoring.py -v
"""

import time

from fastaiagent.agent.memory_blocks import VectorBlock


class _FakeStore:
    """In-memory stand-in for fastaiagent.kb.protocols.VectorStore."""

    def __init__(self, hits):
        self._hits = hits

    def add(self, chunks, embeddings):
        return None

    def search(self, embedding, top_k):
        return list(self._hits)[:top_k]


class _FakeEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _chunk(content: str, *, created_at: float):
    from fastaiagent.kb.chunking import Chunk

    return Chunk(
        id=content,
        content=content,
        metadata={"namespace": "default", "created_at": created_at},
        index=0,
        start_char=0,
        end_char=len(content),
    )


def test_recency_promotes_newer_email_over_older_one() -> None:
    now = time.time()
    hits = [
        # Both messages match the query "what's my email?". The old one
        # has slightly higher cosine similarity, but it's stale.
        (_chunk("alice@old.com", created_at=now - 7 * 86400), 0.85),
        (_chunk("alice@new.com", created_at=now - 3600), 0.80),
    ]

    # Without recency_weight: similarity wins — wrong answer surfaces.
    similarity_only = VectorBlock(
        store=_FakeStore(hits), embedder=_FakeEmbedder(), top_k=2
    )
    body = similarity_only.render("what's my email?")[0].content
    assert body.find("alice@old.com") < body.find("alice@new.com")

    # With recency_weight=0.3 (1h half-life): the newer message wins.
    with_recency = VectorBlock(
        store=_FakeStore(hits),
        embedder=_FakeEmbedder(),
        top_k=2,
        recency_weight=0.3,
        recency_half_life_seconds=3600.0,
    )
    body = with_recency.render("what's my email?")[0].content
    assert body.find("alice@new.com") < body.find("alice@old.com")
