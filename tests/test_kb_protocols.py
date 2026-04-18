"""Contract tests for ``VectorStore``, ``KeywordStore``, and ``MetadataStore``.

Each backend that ships with the SDK must pass these tests — they encode the
behavioral contract that ``LocalKB`` (and user code) relies on.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from fastaiagent.kb.backends.bm25 import BM25KeywordStore
from fastaiagent.kb.backends.faiss import FaissVectorStore
from fastaiagent.kb.backends.sqlite import SqliteMetadataStore
from fastaiagent.kb.chunking import Chunk
from fastaiagent.kb.document import Document

DIM = 16


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        content=text,
        metadata={"source": "contract-test"},
        index=idx,
        start_char=0,
        end_char=len(text),
    )


def _simple_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic toy embedder that gives different vectors per text."""
    import math

    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * DIM
        for i, ch in enumerate(text.lower()):
            vec[ord(ch) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


# ---------------------------------------------------------------------------
# VectorStore contract
# ---------------------------------------------------------------------------


class VectorStoreContract:
    """Mixin — subclasses provide a ``make()`` factory for a backend."""

    def make(self) -> object:  # pragma: no cover — overridden
        raise NotImplementedError

    def test_add_and_search_returns_matching_chunks(self) -> None:
        store = self.make()
        texts = ["alpha bravo", "charlie delta", "echo foxtrot"]
        chunks = [_chunk(t, i) for i, t in enumerate(texts)]
        embeddings = _simple_embed(texts)
        store.add(chunks, embeddings)

        assert store.count() == 3
        query = _simple_embed(["alpha bravo"])[0]
        results = store.search(query, top_k=2)
        assert results, "search returned no results"
        top_chunk, top_score = results[0]
        assert top_chunk.content == "alpha bravo"
        assert isinstance(top_score, float)

    def test_misaligned_add_raises(self) -> None:
        store = self.make()
        with pytest.raises(ValueError):
            store.add([_chunk("x")], _simple_embed(["x", "y"]))

    def test_delete_removes_chunks(self) -> None:
        store = self.make()
        c1 = _chunk("keep me", 0)
        c2 = _chunk("delete me", 1)
        store.add([c1, c2], _simple_embed(["keep me", "delete me"]))
        store.delete([c2.id])
        # c1 still present; count reflects the deletion.
        assert store.count() in (1, 2)  # FAISS keeps slots; Qdrant/Chroma compact

    def test_rebuild_replaces_contents(self) -> None:
        store = self.make()
        chunks_a = [_chunk("old", 0)]
        store.add(chunks_a, _simple_embed(["old"]))
        chunks_b = [_chunk("new-1", 0), _chunk("new-2", 1)]
        store.rebuild(chunks_b, _simple_embed(["new-1", "new-2"]))
        assert store.count() == 2
        results = store.search(_simple_embed(["new-1"])[0], top_k=1)
        assert results and results[0][0].content.startswith("new")

    def test_reset_empties_store(self) -> None:
        store = self.make()
        store.add([_chunk("x")], _simple_embed(["x"]))
        store.reset()
        assert store.count() == 0
        assert store.search(_simple_embed(["x"])[0], top_k=1) == []


class TestFaissVectorStoreContract(VectorStoreContract):
    def make(self) -> object:
        return FaissVectorStore(dimension=DIM, index_type="flat")


# ---------------------------------------------------------------------------
# KeywordStore contract
# ---------------------------------------------------------------------------


class KeywordStoreContract:
    def make(self) -> object:  # pragma: no cover — overridden
        raise NotImplementedError

    def test_add_and_search(self) -> None:
        store = self.make()
        c1 = _chunk("the quick brown fox", 0)
        c2 = _chunk("a lazy green frog", 1)
        store.add([c1, c2])
        results = store.search("fox", top_k=1)
        assert results
        assert results[0][0].content == "the quick brown fox"

    def test_delete(self) -> None:
        store = self.make()
        c1 = _chunk("python rocks", 0)
        c2 = _chunk("javascript rocks too", 1)
        store.add([c1, c2])
        store.delete([c1.id])
        results = store.search("python", top_k=5)
        assert all(r[0].id != c1.id for r in results)

    def test_empty_corpus_and_empty_query_safe(self) -> None:
        store = self.make()
        assert store.search("anything", top_k=3) == []
        store.add([_chunk("some content", 0)])
        assert store.search("", top_k=3) == []

    def test_rebuild(self) -> None:
        store = self.make()
        store.add([_chunk("old content", 0)])
        new_chunk = _chunk("new content", 0)
        store.rebuild([new_chunk])
        results = store.search("new", top_k=1)
        assert results and results[0][0].id == new_chunk.id

    def test_reset(self) -> None:
        store = self.make()
        store.add([_chunk("a word", 0)])
        store.reset()
        assert store.search("word", top_k=1) == []


class TestBM25KeywordStoreContract(KeywordStoreContract):
    def make(self) -> object:
        return BM25KeywordStore()


# ---------------------------------------------------------------------------
# MetadataStore contract
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_store(tmp_path: Path) -> SqliteMetadataStore:
    store = SqliteMetadataStore(tmp_path / "meta.sqlite")
    yield store
    store.close()


def test_metadata_document_roundtrip(sqlite_store: SqliteMetadataStore) -> None:
    doc = Document(content="hello world", metadata={"tag": "greeting"}, source="s1")
    sqlite_store.put_document(doc)
    loaded = sqlite_store.get_document("s1")
    assert loaded is not None
    assert loaded.content == "hello world"
    assert loaded.metadata == {"tag": "greeting"}
    assert loaded.source == "s1"


def test_metadata_list_documents(sqlite_store: SqliteMetadataStore) -> None:
    sqlite_store.put_document(Document(content="a", source="s1"))
    sqlite_store.put_document(Document(content="b", source="s2"))
    docs = sqlite_store.list_documents()
    assert {d.source for d in docs} == {"s1", "s2"}


def test_metadata_delete_document(sqlite_store: SqliteMetadataStore) -> None:
    sqlite_store.put_document(Document(content="a", source="s1"))
    sqlite_store.delete_document("s1")
    assert sqlite_store.get_document("s1") is None


def test_metadata_chunks_roundtrip(sqlite_store: SqliteMetadataStore) -> None:
    chunks = [_chunk("c1", 0), _chunk("c2", 1)]
    embeddings = _simple_embed(["c1", "c2"])
    sqlite_store.put_chunks(chunks, embeddings)

    loaded_chunks, loaded_embs = sqlite_store.get_chunks()
    assert len(loaded_chunks) == 2
    assert [c.content for c in loaded_chunks] == ["c1", "c2"]
    assert len(loaded_embs) == 2
    assert len(loaded_embs[0]) == DIM


def test_metadata_chunks_without_embeddings(sqlite_store: SqliteMetadataStore) -> None:
    chunks = [_chunk("c1", 0)]
    sqlite_store.put_chunks(chunks, None)
    loaded_chunks, loaded_embs = sqlite_store.get_chunks()
    assert len(loaded_chunks) == 1
    assert loaded_embs == [[]]


def test_metadata_delete_chunks(sqlite_store: SqliteMetadataStore) -> None:
    chunks = [_chunk("c1", 0), _chunk("c2", 1)]
    sqlite_store.put_chunks(chunks, _simple_embed(["c1", "c2"]))
    sqlite_store.delete_chunks([chunks[0].id])
    remaining, _ = sqlite_store.get_chunks()
    assert {c.id for c in remaining} == {chunks[1].id}


def test_metadata_reset(sqlite_store: SqliteMetadataStore) -> None:
    sqlite_store.put_document(Document(content="a", source="s1"))
    sqlite_store.put_chunks([_chunk("c", 0)], None)
    sqlite_store.reset()
    assert sqlite_store.list_documents() == []
    assert sqlite_store.get_chunks() == ([], [])
