"""FAISS-based vector search for knowledge base."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from fastaiagent.kb.chunking import Chunk


class SearchResult(BaseModel):
    """A search result with score."""

    chunk: Chunk
    score: float

    model_config = {"arbitrary_types_allowed": True}


IndexType = Literal["flat", "ivf", "hnsw"]


class FaissIndex:
    """FAISS vector index with configurable index type.

    Supported index types:
        - "flat": Exact brute-force inner product (IndexFlatIP). Best for < 100K vectors.
        - "ivf": Inverted file index (IndexIVFFlat). Best for 100K-1M vectors.
        - "hnsw": Hierarchical Navigable Small World graph (IndexHNSWFlat). Best for speed + recall.
    """

    def __init__(self, dimension: int, index_type: IndexType = "flat"):
        import faiss

        self._dimension = dimension
        self._index_type = index_type
        self._faiss = faiss
        self._index = self._create_index(dimension, index_type)
        # IVF requires training before adding vectors
        self._trained = index_type != "ivf"

    def _create_index(self, dimension: int, index_type: IndexType):
        """Create the underlying FAISS index."""
        faiss = self._faiss
        if index_type == "flat":
            return faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            # nlist=100 is a good default; auto-adjusted in _train_if_needed
            index = faiss.IndexIVFFlat(
                quantizer, dimension, min(100, 1), faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = 10
            return index
        elif index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
            return index
        else:
            raise ValueError(f"Unknown index_type: {index_type!r}. Use 'flat', 'ivf', or 'hnsw'.")

    @property
    def count(self) -> int:
        return self._index.ntotal

    def add(self, embeddings: list[list[float]]) -> None:
        """Add embeddings to the index."""
        if not embeddings:
            return
        import numpy as np

        arr = np.array(embeddings, dtype=np.float32)
        if self._index_type == "ivf" and not self._trained:
            self._train_if_needed(arr)
        self._index.add(arr)

    def _train_if_needed(self, arr) -> None:
        """Train IVF index if not yet trained."""
        if self._trained:
            return
        n = arr.shape[0]
        # Rebuild with appropriate nlist
        nlist = max(1, min(100, n // 10))
        quantizer = self._faiss.IndexFlatIP(self._dimension)
        self._index = self._faiss.IndexIVFFlat(
            quantizer, self._dimension, nlist, self._faiss.METRIC_INNER_PRODUCT
        )
        self._index.nprobe = min(10, nlist)
        self._index.train(arr)
        self._trained = True

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[int, float]]:
        """Search and return list of (index, score) pairs."""
        if self._index.ntotal == 0:
            return []
        import numpy as np

        q = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx >= 0:  # FAISS returns -1 for padding
                results.append((idx, float(scores[0][i])))
        return results

    def rebuild(self, embeddings: list[list[float]]) -> None:
        """Rebuild the index from scratch."""
        self._index = self._create_index(self._dimension, self._index_type)
        self._trained = self._index_type != "ivf"
        if embeddings:
            self.add(embeddings)

    def reset(self) -> None:
        """Clear the index."""
        self._index = self._create_index(self._dimension, self._index_type)
        self._trained = self._index_type != "ivf"
