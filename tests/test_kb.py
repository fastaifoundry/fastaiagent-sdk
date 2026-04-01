"""Tests for fastaiagent.kb module."""

from __future__ import annotations

import pytest

from fastaiagent.kb import LocalKB
from fastaiagent.kb.chunking import Chunk, chunk_text
from fastaiagent.kb.document import Document, ingest_file
from fastaiagent.kb.embedding import SimpleEmbedder
from fastaiagent.kb.search import SearchResult, cosine_similarity, search


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


class TestEmbedding:
    def test_simple_embedder(self):
        embedder = SimpleEmbedder(dimensions=64)
        vecs = embedder.embed(["hello", "world"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 64

    def test_embeddings_are_normalized(self):
        import math

        embedder = SimpleEmbedder(dimensions=64)
        vec = embedder.embed(["test"])[0]
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.01


class TestSearch:
    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_search_returns_top_k(self):
        chunks = [Chunk(content=f"chunk {i}", index=i) for i in range(10)]
        embedder = SimpleEmbedder(dimensions=32)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed(texts)
        query_emb = embedder.embed(["chunk 5"])[0]
        results = search(query_emb, embeddings, chunks, top_k=3)
        assert len(results) == 3
        assert results[0].score >= results[1].score


class TestLocalKB:
    def test_add_text_and_search(self):
        kb = LocalKB(name="test", embedder=SimpleEmbedder(dimensions=64))
        kb.add("Python is a programming language.")
        kb.add("JavaScript is used for web development.")
        kb.add("Machine learning uses neural networks.")

        results = kb.search("programming", top_k=2)
        assert len(results) <= 2
        assert len(results) > 0

    def test_add_file(self, temp_dir):
        path = temp_dir / "doc.txt"
        path.write_text("FastAIAgent SDK builds AI agents.")
        kb = LocalKB(name="test", embedder=SimpleEmbedder())
        count = kb.add(str(path))
        assert count >= 1

    def test_status(self):
        kb = LocalKB(name="my-kb", embedder=SimpleEmbedder())
        kb.add("test content")
        status = kb.status()
        assert status["name"] == "my-kb"
        assert status["chunk_count"] >= 1

    def test_as_tool(self):
        kb = LocalKB(name="docs", embedder=SimpleEmbedder())
        kb.add("Python programming language.")
        tool = kb.as_tool()
        assert tool.name == "search_docs"
        result = tool.execute({"query": "Python"})
        assert result.success

    def test_empty_search(self):
        kb = LocalKB(name="empty", embedder=SimpleEmbedder())
        results = kb.search("anything")
        assert len(results) == 0
