"""Pluggable storage backends for LocalKB.

Default in-process backends (``FaissVectorStore``, ``BM25KeywordStore``,
``SqliteMetadataStore``) are always importable. External-service backends
(``QdrantVectorStore``, ``ChromaVectorStore``) are gated behind optional
extras — importing them without the upstream SDK installed raises
``ImportError`` with install instructions.
"""

from __future__ import annotations

from fastaiagent.kb.backends.bm25 import BM25KeywordStore
from fastaiagent.kb.backends.faiss import FaissVectorStore
from fastaiagent.kb.backends.sqlite import SqliteMetadataStore

__all__ = [
    "BM25KeywordStore",
    "FaissVectorStore",
    "SqliteMetadataStore",
]
