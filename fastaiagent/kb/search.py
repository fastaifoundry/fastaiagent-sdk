"""Cosine similarity search for knowledge base."""

from __future__ import annotations

import math

from pydantic import BaseModel

from fastaiagent.kb.chunking import Chunk


class SearchResult(BaseModel):
    """A search result with score."""

    chunk: Chunk
    score: float

    model_config = {"arbitrary_types_allowed": True}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[SearchResult]:
    """Search for the most similar chunks to the query."""
    scores = []
    for i, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]

    return [SearchResult(chunk=chunks[idx], score=score) for idx, score in top]
