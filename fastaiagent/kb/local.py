"""LocalKB — file-based knowledge base with embedding search."""

from __future__ import annotations

from pathlib import Path

from fastaiagent.kb.chunking import Chunk, chunk_text
from fastaiagent.kb.document import Document, ingest_file
from fastaiagent.kb.embedding import Embedder, get_default_embedder
from fastaiagent.kb.search import SearchResult, search


class LocalKB:
    """Local file-based knowledge base.

    Example:
        kb = LocalKB("docs")
        kb.add("readme.md")
        results = kb.search("How to install?")
        tool = kb.as_tool()
    """

    def __init__(
        self,
        name: str = "default",
        path: str = ".fastaiagent/kb/",
        embedder: Embedder | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.name = name
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder or get_default_embedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, path_or_text: str, metadata: dict | None = None) -> int:
        """Add a file or raw text to the knowledge base. Returns chunk count."""
        p = Path(path_or_text)
        if p.exists():
            docs = ingest_file(p)
        else:
            docs = [Document(content=path_or_text, metadata=metadata or {})]

        return self.add_documents(docs)

    def add_documents(self, docs: list[Document]) -> int:
        """Add documents to the KB. Returns number of chunks created."""
        new_chunks = []
        for doc in docs:
            chunks = chunk_text(
                doc.content,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                metadata={**doc.metadata, "source": doc.source},
            )
            new_chunks.extend(chunks)

        if new_chunks:
            texts = [c.content for c in new_chunks]
            embeddings = self._embedder.embed(texts)
            self._chunks.extend(new_chunks)
            self._embeddings.extend(embeddings)

        return len(new_chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the knowledge base."""
        if not self._chunks:
            return []

        query_emb = self._embedder.embed([query])[0]
        return search(query_emb, self._embeddings, self._chunks, top_k=top_k)

    def as_tool(self):
        """Create a FunctionTool that wraps this KB for agent use."""
        from fastaiagent.tool.function import FunctionTool

        def kb_search(query: str, top_k: int = 5) -> str:
            results = self.search(query, top_k=top_k)
            if not results:
                return "No results found."
            parts = []
            for r in results:
                parts.append(f"[Score: {r.score:.3f}] {r.chunk.content[:200]}")
            return "\n\n".join(parts)

        return FunctionTool(
            name=f"search_{self.name}",
            fn=kb_search,
            description=f"Search the '{self.name}' knowledge base",
        )

    def status(self) -> dict:
        """Get KB status."""
        return {
            "name": self.name,
            "chunk_count": len(self._chunks),
            "path": str(self.path),
        }
