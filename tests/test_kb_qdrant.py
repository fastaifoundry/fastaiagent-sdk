"""Live end-to-end tests for ``QdrantVectorStore``.

Uses qdrant-client's ``location=":memory:"`` mode — an in-process ephemeral
Qdrant. No external Qdrant service required. Gated by ``@pytest.mark.qdrant``
and skipped if ``qdrant-client`` is not installed.
"""

from __future__ import annotations

import uuid

import pytest

qdrant_client = pytest.importorskip("qdrant_client")

from fastaiagent.kb import LocalKB  # noqa: E402 — after importorskip
from fastaiagent.kb.backends.qdrant import QdrantVectorStore  # noqa: E402
from fastaiagent.kb.chunking import Chunk  # noqa: E402
from fastaiagent.kb.embedding import SimpleEmbedder  # noqa: E402

pytestmark = pytest.mark.qdrant

DIM = 16


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        content=text,
        metadata={"src": "qdrant-live"},
        index=idx,
        start_char=0,
        end_char=len(text),
    )


def _embed(texts: list[str]) -> list[list[float]]:
    import math

    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * DIM
        for ch in text.lower():
            vec[ord(ch) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


@pytest.fixture
def qdrant_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        collection=f"test_{uuid.uuid4().hex[:8]}",
        dimension=DIM,
        location=":memory:",
    )


def test_qdrant_add_and_search_roundtrip(qdrant_store: QdrantVectorStore) -> None:
    texts = ["alpha doc", "bravo doc", "charlie doc"]
    chunks = [_chunk(t, i) for i, t in enumerate(texts)]
    qdrant_store.add(chunks, _embed(texts))

    assert qdrant_store.count() == 3

    q = _embed(["alpha doc"])[0]
    hits = qdrant_store.search(q, top_k=2)
    assert hits
    assert hits[0][0].content == "alpha doc"


def test_qdrant_delete(qdrant_store: QdrantVectorStore) -> None:
    c1 = _chunk("keep", 0)
    c2 = _chunk("delete", 1)
    qdrant_store.add([c1, c2], _embed(["keep", "delete"]))
    qdrant_store.delete([c2.id])
    assert qdrant_store.count() == 1


def test_qdrant_rebuild(qdrant_store: QdrantVectorStore) -> None:
    qdrant_store.add([_chunk("old", 0)], _embed(["old"]))
    new_chunks = [_chunk("n1", 0), _chunk("n2", 1)]
    qdrant_store.rebuild(new_chunks, _embed(["n1", "n2"]))
    assert qdrant_store.count() == 2


def test_qdrant_reset(qdrant_store: QdrantVectorStore) -> None:
    qdrant_store.add([_chunk("x", 0)], _embed(["x"]))
    qdrant_store.reset()
    assert qdrant_store.count() == 0


def test_qdrant_metadata_roundtrip(qdrant_store: QdrantVectorStore) -> None:
    c = Chunk(
        id=str(uuid.uuid4()),
        content="metadata check",
        metadata={"tag": "a", "nested": {"deep": True}},
        index=0,
        start_char=0,
        end_char=14,
    )
    qdrant_store.add([c], _embed(["metadata check"]))
    hits = qdrant_store.search(_embed(["metadata check"])[0], top_k=1)
    assert hits
    chunk, _ = hits[0]
    assert chunk.metadata.get("tag") == "a"
    assert chunk.metadata.get("nested") == {"deep": True}


def test_localkb_with_qdrant_backend(tmp_path: object) -> None:
    """LocalKB pointed at a Qdrant backend end-to-end."""
    tmp_path_str = str(tmp_path)
    collection = f"kb_{uuid.uuid4().hex[:8]}"
    kb = LocalKB(
        name="qdrant-kb",
        path=tmp_path_str,
        embedder=SimpleEmbedder(dimensions=DIM),
        search_type="vector",
        chunk_size=128,
        chunk_overlap=16,
        vector_store=QdrantVectorStore(
            collection=collection,
            dimension=DIM,
            location=":memory:",
        ),
        persist=False,
    )
    kb.add("The mitochondria is the powerhouse of the cell.")
    kb.add("Octopuses have three hearts and blue blood.")
    results = kb.search("octopus", top_k=1)
    assert results
    assert results[0].chunk.content
