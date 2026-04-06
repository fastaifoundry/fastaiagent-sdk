"""Local knowledge base with FAISS vector search, BM25 keyword search, and SQLite persistence."""

from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.local import LocalKB
from fastaiagent.kb.search import SearchResult

__all__ = ["LocalKB", "Chunk", "SearchResult"]
