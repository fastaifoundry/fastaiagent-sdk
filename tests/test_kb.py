"""Tests for fastaiagent.kb module."""

from __future__ import annotations

import math
import uuid

import pytest

from fastaiagent.kb import Chunk, LocalKB, SearchResult
from fastaiagent.kb.bm25 import BM25Index
from fastaiagent.kb.chunking import chunk_text
from fastaiagent.kb.document import ingest_file
from fastaiagent.kb.embedding import SimpleEmbedder
from fastaiagent.kb.search import FaissIndex

try:
    import faiss  # noqa: F401

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

_skip_no_faiss = pytest.mark.skipif(not _HAS_FAISS, reason="faiss-cpu not installed")


# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------


class TestDocument:
    def test_ingest_text_file(self, temp_dir):
        path = temp_dir / "test.txt"
        path.write_text("Hello world\nThis is a test document.")
        docs = ingest_file(path)
        assert len(docs) == 1
        assert "Hello world" in docs[0].content

    def test_ingest_md_file(self, temp_dir):
        path = temp_dir / "test.md"
        path.write_text("# Title\n\nSome content here.")
        docs = ingest_file(path)
        assert len(docs) == 1

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest_file("/nonexistent/file.txt")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_text(self):
        chunks = chunk_text("Short text", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text"

    def test_long_text_splits(self):
        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=200)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.content) <= 200

    def test_paragraph_splitting(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text(text, chunk_size=30)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_text("")
        assert len(chunks) == 0

    def test_chunk_has_index(self):
        text = "A" * 500
        chunks = chunk_text(text, chunk_size=100)
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_chunk_has_uuid_id(self):
        chunks = chunk_text("Some text for chunking", chunk_size=100)
        assert len(chunks) >= 1
        # Validate UUID format
        uuid.UUID(chunks[0].id)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    def test_simple_embedder(self):
        embedder = SimpleEmbedder(dimensions=64)
        vecs = embedder.embed(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 64

    def test_embeddings_are_normalized(self):
        embedder = SimpleEmbedder(dimensions=64)
        vec = embedder.embed(["test"])[0]
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------


@_skip_no_faiss
class TestFaissIndex:
    def test_flat_index_add_and_search(self):
        embedder = SimpleEmbedder(dimensions=32)
        vecs = embedder.embed(["hello", "world", "python"])
        index = FaissIndex(dimension=32, index_type="flat")
        index.add(vecs)
        assert index.count == 3

        query = embedder.embed(["hello"])[0]
        results = index.search(query, top_k=2)
        assert len(results) == 2
        # First result should be index 0 (identical to query)
        assert results[0][0] == 0

    def test_empty_index_search(self):
        index = FaissIndex(dimension=32, index_type="flat")
        results = index.search([0.0] * 32, top_k=5)
        assert results == []

    def test_rebuild(self):
        embedder = SimpleEmbedder(dimensions=32)
        vecs = embedder.embed(["a", "b", "c"])
        index = FaissIndex(dimension=32, index_type="flat")
        index.add(vecs)
        assert index.count == 3

        new_vecs = embedder.embed(["x", "y"])
        index.rebuild(new_vecs)
        assert index.count == 2

    def test_reset(self):
        embedder = SimpleEmbedder(dimensions=32)
        vecs = embedder.embed(["a", "b"])
        index = FaissIndex(dimension=32, index_type="flat")
        index.add(vecs)
        index.reset()
        assert index.count == 0

    def test_hnsw_index(self):
        embedder = SimpleEmbedder(dimensions=32)
        vecs = embedder.embed(["alpha", "beta", "gamma"])
        index = FaissIndex(dimension=32, index_type="hnsw")
        index.add(vecs)
        assert index.count == 3

        query = embedder.embed(["alpha"])[0]
        results = index.search(query, top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_add_and_search(self):
        chunks = [
            Chunk(content="Python is a programming language", index=0),
            Chunk(content="JavaScript is used for web development", index=1),
            Chunk(content="Error code ERR-4012 payment timeout", index=2),
        ]
        idx = BM25Index()
        idx.add(chunks)
        results = idx.search("Python programming", top_k=2)
        assert len(results) >= 1
        # First result should be the Python chunk
        assert results[0][0] == chunks[0].id

    def test_exact_term_match(self):
        chunks = [
            Chunk(content="Refund policy returns within 30 days", index=0),
            Chunk(content="Error code ERR-4012 payment gateway timeout", index=1),
            Chunk(content="Customer satisfaction guarantee", index=2),
        ]
        idx = BM25Index()
        idx.add(chunks)
        results = idx.search("ERR-4012", top_k=1)
        assert len(results) == 1
        assert results[0][0] == chunks[1].id

    def test_empty_index(self):
        idx = BM25Index()
        results = idx.search("anything", top_k=5)
        assert results == []

    def test_remove(self):
        chunks = [
            Chunk(content="chunk one", index=0),
            Chunk(content="chunk two", index=1),
        ]
        idx = BM25Index()
        idx.add(chunks)
        idx.remove([chunks[0].id])
        results = idx.search("chunk", top_k=5)
        assert len(results) == 1
        assert results[0][0] == chunks[1].id

    def test_rebuild(self):
        chunks = [Chunk(content="original content", index=0)]
        idx = BM25Index()
        idx.add(chunks)

        new_chunks = [Chunk(content="new content", index=0)]
        idx.rebuild(new_chunks)
        results = idx.search("original", top_k=5)
        assert len(results) == 0
        results = idx.search("new", top_k=5)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# LocalKB — core operations
# ---------------------------------------------------------------------------


@_skip_no_faiss
class TestLocalKB:
    def test_add_text_and_search(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        kb.add("Python is a programming language.")
        kb.add("JavaScript is used for web development.")
        kb.add("Machine learning uses neural networks.")

        results = kb.search("programming", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0

    def test_add_file(self, temp_dir):
        path = temp_dir / "doc.txt"
        path.write_text("FastAIAgent SDK builds AI agents.")
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(), persist=False,
        )
        count = kb.add(str(path))
        assert count >= 1

    def test_status(self, temp_dir):
        kb = LocalKB(
            name="my-kb", path=str(temp_dir),
            embedder=SimpleEmbedder(), persist=False,
        )
        kb.add("test content")
        status = kb.status()
        assert status["name"] == "my-kb"
        assert status["chunk_count"] >= 1
        assert status["persist"] is False
        assert status["search_type"] == "hybrid"

    def test_as_tool(self, temp_dir):
        kb = LocalKB(
            name="docs", path=str(temp_dir),
            embedder=SimpleEmbedder(), persist=False,
        )
        kb.add("Python programming language.")
        tool = kb.as_tool()
        assert tool.name == "search_docs"
        result = tool.execute({"query": "Python"})
        assert result.success

    def test_empty_search(self, temp_dir):
        kb = LocalKB(
            name="empty", path=str(temp_dir),
            embedder=SimpleEmbedder(), persist=False,
        )
        results = kb.search("anything")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# LocalKB — persistence
# ---------------------------------------------------------------------------


@_skip_no_faiss
class TestPersistence:
    def test_save_and_reload(self, temp_dir):
        """Data persists across LocalKB instances."""
        kb1 = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        kb1.add("Python is great.")
        kb1.add("JavaScript is cool.")
        count = kb1.status()["chunk_count"]
        kb1.close()

        kb2 = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        assert kb2.status()["chunk_count"] == count
        results = kb2.search("Python", top_k=1)
        assert len(results) > 0
        kb2.close()

    def test_auto_save_on_add(self, temp_dir):
        """Each add() is immediately persisted."""
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        kb.add("First chunk")

        from fastaiagent._internal.storage import SQLiteHelper

        db = SQLiteHelper(temp_dir / "test" / "kb.sqlite")
        rows = db.fetchall("SELECT * FROM chunks")
        assert len(rows) >= 1
        db.close()
        kb.close()

    def test_persist_false_no_files(self, temp_dir):
        """persist=False creates no files on disk."""
        kb = LocalKB(
            name="temp-kb", path=str(temp_dir),
            embedder=SimpleEmbedder(), persist=False,
        )
        kb.add("Some content")
        assert not (temp_dir / "temp-kb" / "kb.sqlite").exists()
        kb.close()

    def test_context_manager(self, temp_dir):
        """Context manager closes the DB."""
        with LocalKB(
            name="ctx", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        ) as kb:
            kb.add("context manager test")
            assert kb.status()["chunk_count"] >= 1
        # After exit, DB is closed — reload works
        kb2 = LocalKB(
            name="ctx", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        assert kb2.status()["chunk_count"] >= 1
        kb2.close()


# ---------------------------------------------------------------------------
# LocalKB — CRUD
# ---------------------------------------------------------------------------


@_skip_no_faiss
class TestCRUD:
    def test_delete_by_id(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        kb.add("Chunk to delete")
        chunk_id = kb._chunks[0].id
        assert kb.delete(chunk_id) is True
        assert kb.status()["chunk_count"] == 0

    def test_delete_nonexistent(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        assert kb.delete("nonexistent-id") is False

    def test_delete_by_source(self, temp_dir):
        path = temp_dir / "doc.txt"
        path.write_text("Some content for deletion test.")
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        kb.add(str(path))
        initial_count = kb.status()["chunk_count"]
        deleted = kb.delete_by_source(str(path))
        assert deleted == initial_count
        assert kb.status()["chunk_count"] == 0

    def test_delete_persists(self, temp_dir):
        """Deleting a chunk persists to SQLite."""
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        kb.add("Chunk to keep")
        kb.add("Chunk to delete")
        assert kb.status()["chunk_count"] == 2
        chunk_id = kb._chunks[1].id
        kb.delete(chunk_id)
        kb.close()

        kb2 = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        assert kb2.status()["chunk_count"] == 1
        kb2.close()

    def test_update_chunk(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        kb.add("Original content")
        chunk_id = kb._chunks[0].id
        assert kb.update(chunk_id, "Updated content") is True
        assert kb._chunks[0].content == "Updated content"

    def test_update_persists(self, temp_dir):
        kb1 = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        kb1.add("Original")
        chunk_id = kb1._chunks[0].id
        kb1.update(chunk_id, "Updated")
        kb1.close()

        kb2 = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        assert kb2._chunks[0].content == "Updated"
        kb2.close()

    def test_clear(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        kb.add("content 1")
        kb.add("content 2")
        kb.clear()
        assert kb.status()["chunk_count"] == 0

    def test_clear_persists(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        kb.add("content")
        kb.clear()
        kb.close()

        kb2 = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
        )
        assert kb2.status()["chunk_count"] == 0
        kb2.close()


# ---------------------------------------------------------------------------
# LocalKB — search types
# ---------------------------------------------------------------------------


@_skip_no_faiss
class TestSearchTypes:
    def test_vector_only(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
            search_type="vector", persist=False,
        )
        kb.add("Python programming language.")
        kb.add("JavaScript web development.")
        results = kb.search("Python", top_k=1)
        assert len(results) == 1
        assert kb._bm25_index is None

    def test_keyword_only(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            search_type="keyword", persist=False,
        )
        kb.add("Error code ERR-4012 payment gateway timeout.")
        kb.add("Refund policy returns within 30 days.")
        results = kb.search("ERR-4012", top_k=1)
        assert len(results) == 1
        assert "ERR-4012" in results[0].chunk.content
        # No embedder should be initialized
        assert kb._embedder is None
        assert kb._faiss_index is None

    def test_keyword_no_embeddings_stored(self, temp_dir):
        """keyword mode should not compute or store embeddings."""
        kb = LocalKB(
            name="test", path=str(temp_dir),
            search_type="keyword",
        )
        kb.add("Some text content")
        assert len(kb._embeddings) == 0

        # SQLite should have NULL embeddings
        from fastaiagent._internal.storage import SQLiteHelper

        db = SQLiteHelper(temp_dir / "test" / "kb.sqlite")
        rows = db.fetchall("SELECT embedding FROM chunks")
        assert all(row["embedding"] is None for row in rows)
        db.close()
        kb.close()

    def test_hybrid_default(self, temp_dir):
        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
            persist=False,
        )
        kb.add("Error code ERR-4012 payment gateway timeout.")
        kb.add("Refund policy returns within 30 days.")
        kb.add("Customer satisfaction guarantee program.")

        # Hybrid should find ERR-4012 via BM25 boost
        results = kb.search("ERR-4012 payment", top_k=2)
        assert len(results) >= 1

    def test_hybrid_alpha(self, temp_dir):
        """Different alpha values shift ranking."""
        kb_semantic = LocalKB(
            name="sem", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
            alpha=0.9, persist=False,
        )
        kb_keyword = LocalKB(
            name="kw", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64),
            alpha=0.1, persist=False,
        )
        texts = [
            "Error code ERR-4012 payment gateway timeout.",
            "Payment processing and billing information.",
            "Customer satisfaction guarantee.",
        ]
        for t in texts:
            kb_semantic.add(t)
            kb_keyword.add(t)

        # With alpha=0.1 (keyword-heavy), ERR-4012 chunk should rank higher for exact match
        results_kw = kb_keyword.search("ERR-4012", top_k=1)
        assert "ERR-4012" in results_kw[0].chunk.content


# ---------------------------------------------------------------------------
# LocalKB — directory ingestion
# ---------------------------------------------------------------------------


@_skip_no_faiss
class TestDirectoryIngestion:
    def test_add_directory(self, temp_dir):
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.txt").write_text("First document content.")
        (docs_dir / "b.md").write_text("# Second document\n\nContent here.")
        # Unsupported file should be ignored
        (docs_dir / "c.json").write_text('{"key": "value"}')

        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        count = kb.add(str(docs_dir))
        assert count >= 2  # At least 2 chunks from 2 files

    def test_add_nested_directory(self, temp_dir):
        docs_dir = temp_dir / "docs"
        sub_dir = docs_dir / "sub"
        sub_dir.mkdir(parents=True)
        (docs_dir / "top.txt").write_text("Top level doc.")
        (sub_dir / "nested.txt").write_text("Nested doc.")

        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        count = kb.add(str(docs_dir))
        assert count >= 2

    def test_add_empty_directory(self, temp_dir):
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        kb = LocalKB(
            name="test", path=str(temp_dir),
            embedder=SimpleEmbedder(dimensions=64), persist=False,
        )
        count = kb.add(str(empty_dir))
        assert count == 0
