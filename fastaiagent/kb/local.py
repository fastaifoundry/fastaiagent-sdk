"""LocalKB — persistent knowledge base with FAISS vector search and BM25 keyword search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastaiagent.kb.chunking import Chunk, chunk_text
from fastaiagent.kb.document import Document, ingest_file
from fastaiagent.kb.embedding import Embedder, get_default_embedder
from fastaiagent.kb.search import FaissIndex, IndexType, SearchResult
from fastaiagent.kb.bm25 import BM25Index
from fastaiagent.tool.base import Tool

SearchType = Literal["vector", "keyword", "hybrid"]

_KB_SCHEMA = """\
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    index_pos INTEGER DEFAULT 0,
    start_char INTEGER DEFAULT 0,
    end_char INTEGER DEFAULT 0,
    embedding TEXT
)"""

# Supported file extensions for directory ingestion
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


class LocalKB:
    """Local knowledge base with persistent storage, FAISS vector search, and BM25 keyword search.

    Example::

        # Persistent (default) — survives restarts
        kb = LocalKB(name="docs")
        kb.add("docs/")
        results = kb.search("refund policy", top_k=3)

        # Temporary — in-memory only
        kb = LocalKB(name="scratch", persist=False)
        kb.add("some dynamic content")
        results = kb.search("keyword")
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

        # In-memory state
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []

        # Indexes — only create what's needed
        self._faiss_index: FaissIndex | None = None
        self._bm25_index: BM25Index | None = None
        if search_type in ("keyword", "hybrid"):
            self._bm25_index = BM25Index()

        # Persistence
        self._db = None
        if persist:
            self.path.mkdir(parents=True, exist_ok=True)
            from fastaiagent._internal.storage import SQLiteHelper

            self._db = SQLiteHelper(self.path / "kb.sqlite")
            self._db.execute(_KB_SCHEMA)
            self._load()
        else:
            # Ensure path dir is not created for non-persistent KBs
            pass

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load all chunks and embeddings from SQLite into memory."""
        if self._db is None:
            return

        rows = self._db.fetchall("SELECT * FROM chunks ORDER BY rowid")
        if not rows:
            return

        for row in rows:
            chunk = Chunk(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row["metadata"]),
                index=row["index_pos"],
                start_char=row["start_char"],
                end_char=row["end_char"],
            )
            self._chunks.append(chunk)
            if row["embedding"]:
                self._embeddings.append(json.loads(row["embedding"]))

        # Validate embedding dimensions match current embedder
        if self._embedder and self._embeddings:
            stored_dim = len(self._embeddings[0])
            test_emb = self._embedder.embed(["test"])[0]
            if len(test_emb) != stored_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: stored={stored_dim}, "
                    f"current embedder={len(test_emb)}. "
                    "Use the same embedder that created this KB."
                )

        # Rebuild FAISS index from loaded embeddings
        if self.search_type in ("vector", "hybrid") and self._embeddings:
            dim = len(self._embeddings[0])
            self._faiss_index = FaissIndex(dim, self.index_type)
            self._faiss_index.add(self._embeddings)

        # Rebuild BM25 index from loaded chunks
        if self._bm25_index and self._chunks:
            self._bm25_index.rebuild(self._chunks)

    @staticmethod
    def _to_python_floats(embedding: list) -> list[float]:
        """Convert numpy float32 values to Python floats for JSON serialization."""
        return [float(v) for v in embedding]

    def _persist_chunks(self, chunks: list[Chunk], embeddings: list[list[float]] | None) -> None:
        """Insert new chunks into SQLite."""
        if self._db is None:
            return

        rows = []
        for i, chunk in enumerate(chunks):
            emb_json = (
                json.dumps(self._to_python_floats(embeddings[i]))
                if embeddings and i < len(embeddings) else None
            )
            rows.append((
                chunk.id,
                chunk.content,
                json.dumps(chunk.metadata),
                chunk.index,
                chunk.start_char,
                chunk.end_char,
                emb_json,
            ))

        self._db.executemany(
            """INSERT OR REPLACE INTO chunks
               (id, content, metadata, index_pos, start_char, end_char, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

    # ------------------------------------------------------------------
    # Add content
    # ------------------------------------------------------------------

    def add(self, path_or_text: str, metadata: dict[str, Any] | None = None) -> int:
        """Add a file, directory, or raw text to the knowledge base. Returns chunk count.

        - If path_or_text is a directory, all supported files (.txt, .md, .pdf) are ingested recursively.
        - If path_or_text is a file path, the file is ingested.
        - Otherwise it is treated as raw text content.
        """
        # Check for directory
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
        """Recursively ingest all supported files from a directory."""
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
        new_chunks = []
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
            self._embeddings.extend(new_embeddings)

            # Update FAISS index
            if self._faiss_index is None:
                dim = len(new_embeddings[0])
                self._faiss_index = FaissIndex(dim, self.index_type)
            self._faiss_index.add(new_embeddings)

        # Update in-memory chunks
        self._chunks.extend(new_chunks)

        # Update BM25 index
        if self._bm25_index:
            self._bm25_index.add(new_chunks)

        # Persist
        self._persist_chunks(new_chunks, new_embeddings)

        return len(new_chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search the knowledge base."""
        if not self._chunks:
            return []

        if self.search_type == "vector":
            return self._vector_search(query, top_k)
        elif self.search_type == "keyword":
            return self._keyword_search(query, top_k)
        else:
            return self._hybrid_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[SearchResult]:
        """FAISS semantic vector search."""
        if not self._embedder or not self._faiss_index:
            return []
        query_emb = self._embedder.embed([query])[0]
        results_raw = self._faiss_index.search(query_emb, top_k)
        return [
            SearchResult(chunk=self._chunks[idx], score=score)
            for idx, score in results_raw
            if 0 <= idx < len(self._chunks)
        ]

    def _keyword_search(self, query: str, top_k: int) -> list[SearchResult]:
        """BM25 keyword search."""
        if not self._bm25_index:
            return []
        results_raw = self._bm25_index.search(query, top_k)
        # Map chunk_id -> chunk
        chunk_map = {c.id: c for c in self._chunks}
        return [
            SearchResult(chunk=chunk_map[cid], score=score)
            for cid, score in results_raw
            if cid in chunk_map
        ]

    def _hybrid_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Combined FAISS + BM25 search with score normalization."""
        # Fetch more than top_k from each to improve merge quality
        fetch_k = top_k * 3

        vector_results = self._vector_search(query, fetch_k)
        keyword_results = self._keyword_search(query, fetch_k)

        if not vector_results and not keyword_results:
            return []
        if not vector_results:
            return keyword_results[:top_k]
        if not keyword_results:
            return vector_results[:top_k]

        # Normalize scores to [0, 1]
        def _normalize(results: list[SearchResult]) -> dict[str, float]:
            if not results:
                return {}
            scores = [r.score for r in results]
            min_s, max_s = min(scores), max(scores)
            spread = max_s - min_s if max_s > min_s else 1.0
            return {r.chunk.id: (r.score - min_s) / spread for r in results}

        vec_scores = _normalize(vector_results)
        kw_scores = _normalize(keyword_results)

        # Merge scores
        all_ids = set(vec_scores.keys()) | set(kw_scores.keys())
        chunk_map = {c.id: c for c in self._chunks}
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
        """Delete a chunk by ID. Returns True if found and deleted."""
        idx = None
        for i, chunk in enumerate(self._chunks):
            if chunk.id == chunk_id:
                idx = i
                break
        if idx is None:
            return False

        self._chunks.pop(idx)
        if self._embeddings and idx < len(self._embeddings):
            self._embeddings.pop(idx)

        # Rebuild active indexes
        if self._faiss_index and self._embeddings:
            self._faiss_index.rebuild(self._embeddings)
        elif self._faiss_index:
            self._faiss_index.reset()

        if self._bm25_index:
            self._bm25_index.remove([chunk_id])

        # Persist
        if self._db:
            self._db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))

        return True

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source. Returns count deleted."""
        ids_to_delete = [c.id for c in self._chunks if c.metadata.get("source") == source]
        if not ids_to_delete:
            return 0

        id_set = set(ids_to_delete)
        paired = [
            (c, e) for c, e in zip(self._chunks, self._embeddings or [None] * len(self._chunks))
            if c.id not in id_set
        ]
        if paired:
            self._chunks = [c for c, _ in paired]
            if self._embeddings:
                self._embeddings = [e for _, e in paired if e is not None]
        else:
            self._chunks = []
            self._embeddings = []

        # Rebuild active indexes
        if self._faiss_index:
            self._faiss_index.rebuild(self._embeddings) if self._embeddings else self._faiss_index.reset()

        if self._bm25_index:
            self._bm25_index.remove(ids_to_delete)

        # Persist
        if self._db:
            placeholders = ",".join("?" for _ in ids_to_delete)
            self._db.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders})",
                tuple(ids_to_delete),
            )

        return len(ids_to_delete)

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
        if self._embedder and self.search_type in ("vector", "hybrid"):
            new_embedding = self._embedder.embed([content])[0]
            if idx < len(self._embeddings):
                self._embeddings[idx] = new_embedding
            # Rebuild FAISS
            if self._faiss_index:
                self._faiss_index.rebuild(self._embeddings)

        # Update BM25
        if self._bm25_index:
            self._bm25_index.remove([chunk_id])
            self._bm25_index.add([self._chunks[idx]])

        # Persist
        if self._db:
            emb_json = None
            if self._embedder and idx < len(self._embeddings):
                emb_json = json.dumps(self._to_python_floats(self._embeddings[idx]))
            self._db.execute(
                "UPDATE chunks SET content = ?, embedding = ? WHERE id = ?",
                (content, emb_json, chunk_id),
            )

        return True

    def clear(self) -> None:
        """Remove all chunks and embeddings."""
        self._chunks.clear()
        self._embeddings.clear()
        if self._faiss_index:
            self._faiss_index.reset()
        if self._bm25_index:
            self._bm25_index.reset()
        if self._db:
            self._db.execute("DELETE FROM chunks")

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
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._db:
            self._db.close()

    def __enter__(self) -> LocalKB:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
