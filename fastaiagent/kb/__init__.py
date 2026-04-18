"""Knowledge base with pluggable vector, keyword, and metadata backends."""

from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.document import Document
from fastaiagent.kb.local import LocalKB
from fastaiagent.kb.platform import PlatformKB
from fastaiagent.kb.protocols import KeywordStore, MetadataStore, VectorStore
from fastaiagent.kb.search import SearchResult

__all__ = [
    "Chunk",
    "Document",
    "KeywordStore",
    "LocalKB",
    "MetadataStore",
    "PlatformKB",
    "SearchResult",
    "VectorStore",
]
