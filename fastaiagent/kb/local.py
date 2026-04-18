"""LocalKB — a persistent knowledge base with pluggable storage backends.

Default configuration keeps the historical behavior: FAISS for vector search,
BM25 for keyword search, SQLite for durable metadata. Swap any of the three
by passing a ``VectorStore`` / ``KeywordStore`` / ``MetadataStore`` instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastaiagent.kb.chunking import Chunk, chunk_text
from fastaiagent.kb.document import Document, ingest_file
from fastaiagent.kb.embedding import Embedder, get_default_embedder
from fastaiagent.kb.protocols import KeywordStore, MetadataStore, VectorStore
from fastaiagent.kb.search import IndexType, SearchResult
from fastaiagent.tool.base import Tool

SearchType = Literal["vector", "keyword", "hybrid"]

# Supported file extensions for directory ingestion
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class LocalKB:
    """Knowledge base with pluggable vector, keyword, and metadata backends.

    Example — default (FAISS + BM25 + SQLite), identical to pre-0.3.0 behavior::

        kb = LocalKB(name="docs")
        kb.add("docs/")
        results = kb.search("refund policy", top_k=3)

    Example — remote Qdrant for vectors, default BM25 for keywords::

        from fastaiagent.kb.backends.qdrant import QdrantVectorStore

        kb = LocalKB(
            name="docs",
            vector_store=QdrantVectorStore(
                url="http://localhost:6333",
                collection="docs",
                dimension=384,
            ),
        )

    Example — vector-only, no keyword or persistent metadata::

        kb = LocalKB(name="scratch", search_type="vector", persist=False)
    """

    def __init__(
        self,
        name: str = "default",
        path: str = ".fastaiagent/kb/",
        embedder: Embedder | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        persist: bool = True,
        search_type: SearchType = "hybrid",
        index_type: IndexType = "flat",
        alpha: float = 0.7,
        vector_store: VectorStore | None = None,
        keyword_store: KeywordStore | None = None,
        metadata_store: MetadataStore | None = None,
    ):
        self.name = name
        self.path = Path(path) / name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist = persist
        self.search_type: SearchType = search_type
        self.index_type: IndexType = index_type
        self.alpha = alpha

        # Embedder only needed for vector/hybrid
        self._embedder: Embedder | None = None
        if search_type in ("vector", "hybrid"):
            self._embedder = embedder or get_default_embedder()

        # Set up default backends when not supplied.
        self._vector: VectorStore | None = vector_store
        self._keyword: KeywordStore | None = keyword_store
        self._metadata: MetadataStore | None = metadata_store

        if self._metadata is None and persist:
            self.path.mkdir(parents=True, exist_ok=True)
            from fastaiagent.kb.backends.sqlite import SqliteMetadataStore

            self._metadata = SqliteMetadataStore(self.path / "kb.sqlite")

        if self._keyword is None and search_type in ("keyword", "hybrid"):
            from fastaiagent.kb.backends.bm25 import BM25KeywordStore

            self._keyword = BM25KeywordStore()

        # Eagerly construct the default vector store by probing the embedder
        # for its output dimension. This keeps ``status()`` accurate right
        # after construction (matches pre-0.3.0 user expectations).
        self._chunks: list[Chunk] = []
        if (
            self._vector is None
            and self._embedder is not None
            and search_type in ("vector", "hybrid")
        ):
            from fastaiagent.kb.backends.faiss import FaissVectorStore

            probe_dim = len(self._embedder.embed(["_probe_"])[0])
            self._vector = FaissVectorStore(dimension=probe_dim, index_type=index_type)

        if self._metadata is not None:
            self._load_from_metadata()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _ensure_vector_store(self, dim: int) -> VectorStore | None:
        """Construct the default FAISS vector store lazily once we know the dim."""
        if self._vector is not None or self.search_type not in ("vector", "hybrid"):
            return self._vector
        from fastaiagent.kb.backends.faiss import FaissVectorStore

        self._vector = FaissVectorStore(dimension=dim, index_type=self.index_type)
        return self._vector

    def _load_from_metadata(self) -> None:
        """Replay persisted chunks and embeddings into the active backends."""
        if self._metadata is None:
            return
        chunks, embeddings = self._metadata.get_chunks()
        if not chunks:
            return

        self._chunks = list(chunks)

        # Sanity-check embedding dimensions against the configured embedder.
        non_empty = [e for e in embeddings if e]
        if self._embedder and non_empty:
            stored_dim = len(non_empty[0])
            test_emb = self._embedder.embed(["test"])[0]
            if len(test_emb) != stored_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: stored={stored_dim}, "
                    f"current embedder={len(test_emb)}. "
                    "Use the same embedder that created this KB."
                )

        # Populate vector store
        if self.search_type in ("vector", "hybrid") and non_empty:
            dim = len(non_empty[0])
            vstore = self._ensure_vector_store(dim)
            if vstore is not None:
                # Pair chunks with their embeddings, skipping any without.
                paired_chunks: list[Chunk] = []
                paired_embs: list[list[float]] = []
                for c, e in zip(chunks, embeddings):
                    if e:
                        paired_chunks.append(c)
                        paired_embs.append(e)
                vstore.rebuild(paired_chunks, paired_embs)

        # Populate keyword store
        if self._keyword is not None:
            self._keyword.rebuild(list(chunks))

    # ------------------------------------------------------------------
    # Add content
    # ------------------------------------------------------------------

    def add(self, path_or_text: str, metadata: dict[str, Any] | None = None) -> int:
        """Add a file, directory, or raw text. Returns chunk count.

        - If path_or_text is a directory, all supported files (.txt, .md, .pdf)
          are ingested recursively.
        - If path_or_text is a file path, the file is ingested.
        - Otherwise it is treated as raw text content.
        """
        if len(path_or_text) <= 255:
            try:
                p = Path(path_or_text)
                if p.exists() and p.is_dir():
                    return self._add_directory(p, metadata)
                if p.exists() and p.is_file():
                    docs = ingest_file(p)
                    return self.add_documents(docs)
            except OSError:
                pass

        docs = [Document(content=path_or_text, metadata=metadata or {})]
        return self.add_documents(docs)

    def _add_directory(self, directory: Path, metadata: dict[str, Any] | None = None) -> int:
        total = 0
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                docs = ingest_file(file_path)
                if metadata:
                    for doc in docs:
                        doc.metadata.update(metadata)
                total += self.add_documents(docs)
        return total

    def add_documents(self, docs: list[Document]) -> int:
        """Add documents to the KB. Returns number of chunks created."""
        new_chunks: list[Chunk] = []
        for doc in docs:
            chunks = chunk_text(
                doc.content,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
                metadata={**doc.metadata, "source": doc.source},
            )
            new_chunks.extend(chunks)

        if not new_chunks:
            return 0

        # Compute embeddings if needed
        new_embeddings: list[list[float]] | None = None
        if self._embedder and self.search_type in ("vector", "hybrid"):
            texts = [c.content for c in new_chunks]
            new_embeddings = self._embedder.embed(texts)

            vstore = self._ensure_vector_store(len(new_embeddings[0]))
            if vstore is not None:
                vstore.add(new_chunks, new_embeddings)

        # Update in-memory chunks (kept for search result ordering stability).
        self._chunks.extend(new_chunks)

        # Update keyword store
        if self._keyword is not None:
            self._keyword.add(new_chunks)

        # Persist
        if self._metadata is not None:
            for doc in docs:
                self._metadata.put_document(doc)
            self._metadata.put_chunks(new_chunks, new_embeddings)

        return len(new_chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the knowledge base."""
        if not self._chunks and (self._vector is None or self._vector.count() == 0):
            return []

        if self.search_type == "vector":
            return self._vector_search(query, top_k)
        elif self.search_type == "keyword":
            return self._keyword_search(query, top_k)
        else:
            return self._hybrid_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[SearchResult]:
        if not self._embedder or self._vector is None:
            return []
        query_emb = self._embedder.embed([query])[0]
        raw = self._vector.search(query_emb, top_k)
        return [SearchResult(chunk=c, score=score) for c, score in raw]

    def _keyword_search(self, query: str, top_k: int) -> list[SearchResult]:
        if self._keyword is None:
            return []
        raw = self._keyword.search(query, top_k)
        return [SearchResult(chunk=c, score=score) for c, score in raw]

    def _hybrid_search(self, query: str, top_k: int) -> list[SearchResult]:
        fetch_k = top_k * 3
        vector_results = self._vector_search(query, fetch_k)
        keyword_results = self._keyword_search(query, fetch_k)

        if not vector_results and not keyword_results:
            return []
        if not vector_results:
            return keyword_results[:top_k]
        if not keyword_results:
            return vector_results[:top_k]

        def _normalize(results: list[SearchResult]) -> dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            spread = max_s - min_s if max_s > min_s else 1.0
            return {r.chunk.id: (r.score - min_s) / spread for r in results}

        vec_scores = _normalize(vector_results)
        kw_scores = _normalize(keyword_results)

        chunk_map: dict[str, Chunk] = {}
        for r in vector_results:
            chunk_map[r.chunk.id] = r.chunk
        for r in keyword_results:
            chunk_map.setdefault(r.chunk.id, r.chunk)

        all_ids = set(vec_scores.keys()) | set(kw_scores.keys())
        combined: list[tuple[str, float]] = []
        for cid in all_ids:
            vs = vec_scores.get(cid, 0.0)
            ks = kw_scores.get(cid, 0.0)
            score = self.alpha * vs + (1 - self.alpha) * ks
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return [
            SearchResult(chunk=chunk_map[cid], score=score)
            for cid, score in combined[:top_k]
            if cid in chunk_map
        ]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def delete(self, chunk_id: str) -> bool:
        """Delete a chunk by id. Returns True if found and deleted."""
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.id != chunk_id]
        if len(self._chunks) == before:
            return False
        self._rebuild_indexes_after_delete()
        if self._keyword is not None:
            self._keyword.delete([chunk_id])
        if self._metadata is not None:
            self._metadata.delete_chunks([chunk_id])
        return True

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source. Returns count deleted."""
        ids_to_delete = [c.id for c in self._chunks if c.metadata.get("source") == source]
        if not ids_to_delete:
            return 0
        id_set = set(ids_to_delete)
        self._chunks = [c for c in self._chunks if c.id not in id_set]
        self._rebuild_indexes_after_delete()
        if self._keyword is not None:
            self._keyword.delete(ids_to_delete)
        if self._metadata is not None:
            self._metadata.delete_chunks(ids_to_delete)
        return len(ids_to_delete)

    def _rebuild_indexes_after_delete(self) -> None:
        """Rebuild the vector store from surviving chunks after a delete."""
        if self._vector is None or self._embedder is None:
            return
        if not self._chunks:
            self._vector.reset()
            return
        # Re-embed survivors. Cheap for small corpora; adapters with true
        # per-id delete (Qdrant, Chroma) override this via their own delete().
        texts = [c.content for c in self._chunks]
        embeddings = self._embedder.embed(texts)
        self._vector.rebuild(self._chunks, embeddings)

    def update(self, chunk_id: str, content: str) -> bool:
        """Update a chunk's content and re-embed it. Returns True if found."""
        idx = None
        for i, chunk in enumerate(self._chunks):
            if chunk.id == chunk_id:
                idx = i
                break
        if idx is None:
            return False

        self._chunks[idx].content = content

        # Re-embed if needed
        new_embedding: list[float] | None = None
        if self._embedder and self.search_type in ("vector", "hybrid"):
            new_embedding = self._embedder.embed([content])[0]
            if self._vector is not None:
                # Rebuild is the safe default that works for all adapters.
                texts = [c.content for c in self._chunks]
                embeddings = self._embedder.embed(texts)
                self._vector.rebuild(self._chunks, embeddings)

        # Update BM25
        if self._keyword is not None:
            self._keyword.delete([chunk_id])
            self._keyword.add([self._chunks[idx]])

        # Persist
        if self._metadata is not None:
            embs_to_persist: list[list[float]] | None = (
                [new_embedding] if new_embedding else None
            )
            self._metadata.put_chunks([self._chunks[idx]], embs_to_persist)

        return True

    def clear(self) -> None:
        """Remove all chunks and embeddings."""
        self._chunks.clear()
        if self._vector is not None:
            self._vector.reset()
        if self._keyword is not None:
            self._keyword.reset()
        if self._metadata is not None:
            self._metadata.reset()

    # ------------------------------------------------------------------
    # Tool integration
    # ------------------------------------------------------------------

    def as_tool(self) -> Tool:
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

    # ------------------------------------------------------------------
    # Status & lifecycle
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Get KB status."""
        return {
            "name": self.name,
            "chunk_count": len(self._chunks),
            "path": str(self.path),
            "persist": self.persist,
            "search_type": self.search_type,
            "index_type": self.index_type,
            "vector_backend": type(self._vector).__name__ if self._vector else None,
            "keyword_backend": type(self._keyword).__name__ if self._keyword else None,
            "metadata_backend": type(self._metadata).__name__ if self._metadata else None,
        }

    def close(self) -> None:
        """Close the metadata store's underlying connection, if any."""
        if self._metadata is not None:
            self._metadata.close()

    def __enter__(self) -> LocalKB:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
