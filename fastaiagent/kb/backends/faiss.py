"""FAISS-backed ``VectorStore`` implementation.

Wraps the existing ``FaissIndex`` (inner-product, flat/ivf/hnsw) and tracks
a parallel list of chunk ids so the protocol-level ``search()`` can return
``(Chunk, score)`` pairs instead of raw index positions.
"""

from __future__ import annotations

from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.search import FaissIndex, IndexType


class FaissVectorStore:
    """In-process FAISS vector store. Zero external services required.

    Args:
        dimension: Vector dimensionality. Required before the first ``add()``
            call since FAISS indexes are dimension-locked at construction.
        index_type: ``"flat"`` (exact, <100k vectors), ``"ivf"`` (100k–1M),
            or ``"hnsw"`` (speed/recall tradeoff).
    """

    def __init__(self, dimension: int, index_type: IndexType = "flat"):
        self._dimension = dimension
        self._index_type = index_type
        self._index = FaissIndex(dimension, index_type)
        self._chunks: list[Chunk] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must be aligned: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        self._chunks.extend(chunks)
        self._index.add(embeddings)

    def search(
        self, query_embedding: list[float], top_k: int
    ) -> list[tuple[Chunk, float]]:
        raw = self._index.search(query_embedding, top_k)
        return [
            (self._chunks[idx], score)
            for idx, score in raw
            if 0 <= idx < len(self._chunks)
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        id_set = set(chunk_ids)
        # FAISS has no efficient per-id delete — rebuild from the survivors.
        survivors = [c for c in self._chunks if c.id not in id_set]
        if len(survivors) == len(self._chunks):
            return
        # We don't retain raw embeddings, so a full delete requires a rebuild
        # initiated by the caller with fresh embeddings. Reset here so the
        # next add() starts clean; LocalKB's rebuild() path handles this.
        self._chunks = survivors
        self._index.reset()

    def rebuild(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self._chunks = []
        self._index.reset()
        self.add(chunks, embeddings)

    def reset(self) -> None:
        self._chunks = []
        self._index.reset()

    def count(self) -> int:
        return len(self._chunks)
