"""BM25-backed ``KeywordStore`` implementation.

Wraps the existing ``BM25Index`` — a pure-Python, dependency-free inverted
index with standard BM25 scoring (k1=1.5, b=0.75 defaults).
"""

from __future__ import annotations

from fastaiagent.kb.bm25 import BM25Index
from fastaiagent.kb.chunking import Chunk


class BM25KeywordStore:
    """In-process BM25 keyword store. Zero external services required."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._index = BM25Index(k1=k1, b=b)
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for c in chunks:
            self._chunks[c.id] = c
        self._index.add(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        raw = self._index.search(query, top_k)
        return [
            (self._chunks[chunk_id], score)
            for chunk_id, score in raw
            if chunk_id in self._chunks
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        for cid in chunk_ids:
            self._chunks.pop(cid, None)
        self._index.remove(list(chunk_ids))

    def rebuild(self, chunks: list[Chunk]) -> None:
        self._chunks = {c.id: c for c in chunks}
        self._index.rebuild(chunks)

    def reset(self) -> None:
        self._chunks.clear()
        self._index.reset()
