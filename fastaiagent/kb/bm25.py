"""Lightweight BM25 keyword search for knowledge base. No external dependencies."""

from __future__ import annotations

import math
import re
from collections import defaultdict

from fastaiagent.kb.chunking import Chunk


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 index over chunk texts.

    Parameters:
        k1: Term frequency saturation parameter (default 1.5).
        b: Length normalization parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # chunk_id -> tokenized doc
        self._docs: dict[str, list[str]] = {}
        # term -> set of chunk_ids containing it
        self._inverted: dict[str, set[str]] = defaultdict(set)
        # chunk_id -> doc length
        self._doc_lens: dict[str, int] = {}
        self._avg_dl: float = 0.0

    def _update_avg_dl(self) -> None:
        if self._doc_lens:
            self._avg_dl = sum(self._doc_lens.values()) / len(self._doc_lens)
        else:
            self._avg_dl = 0.0

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index."""
        for chunk in chunks:
            tokens = _tokenize(chunk.content)
            self._docs[chunk.id] = tokens
            self._doc_lens[chunk.id] = len(tokens)
            for token in set(tokens):
                self._inverted[token].add(chunk.id)
        self._update_avg_dl()

    def remove(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID from the index."""
        for cid in chunk_ids:
            tokens = self._docs.pop(cid, None)
            if tokens is None:
                continue
            self._doc_lens.pop(cid, None)
            for token in set(tokens):
                self._inverted[token].discard(cid)
                if not self._inverted[token]:
                    del self._inverted[token]
        self._update_avg_dl()

    def rebuild(self, chunks: list[Chunk]) -> None:
        """Rebuild the entire index from a list of chunks."""
        self.reset()
        self.add(chunks)

    def reset(self) -> None:
        """Clear the index."""
        self._docs.clear()
        self._inverted.clear()
        self._doc_lens.clear()
        self._avg_dl = 0.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search and return list of (chunk_id, bm25_score) pairs, sorted descending."""
        if not self._docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(self._docs)
        scores: dict[str, float] = defaultdict(float)

        for token in query_tokens:
            if token not in self._inverted:
                continue
            df = len(self._inverted[token])
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for cid in self._inverted[token]:
                tf = self._docs[cid].count(token)
                dl = self._doc_lens[cid]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1.0))
                scores[cid] += idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
