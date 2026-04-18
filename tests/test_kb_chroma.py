"""Live end-to-end tests for ``ChromaVectorStore``.

Uses Chroma's in-process ``EphemeralClient`` — no external service required.
Gated by ``@pytest.mark.chroma`` and skipped if ``chromadb`` is not installed.
"""

from __future__ import annotations

import uuid

import pytest

chromadb = pytest.importorskip("chromadb")

from fastaiagent.kb import LocalKB  # noqa: E402 — after importorskip
from fastaiagent.kb.backends.chroma import ChromaVectorStore  # noqa: E402
from fastaiagent.kb.chunking import Chunk  # noqa: E402
from fastaiagent.kb.embedding import SimpleEmbedder  # noqa: E402

pytestmark = pytest.mark.chroma

DIM = 16


def _chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        content=text,
        metadata={"src": "chroma-live"},
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
def chroma_store() -> ChromaVectorStore:
    """Fresh ephemeral Chroma store per test. Unique collection name avoids bleed."""
    collection = f"test_{uuid.uuid4().hex[:8]}"
    return ChromaVectorStore(collection=collection, dimension=DIM)


def test_chroma_add_and_search_roundtrip(chroma_store: ChromaVectorStore) -> None:
    texts = ["alpha document", "bravo document", "charlie document"]
    chunks = [_chunk(t, i) for i, t in enumerate(texts)]
    chroma_store.add(chunks, _embed(texts))

    assert chroma_store.count() == 3

    q = _embed(["alpha document"])[0]
    hits = chroma_store.search(q, top_k=2)
    assert hits
    # Top hit should be the exact-match doc.
    assert hits[0][0].content == "alpha document"


def test_chroma_delete(chroma_store: ChromaVectorStore) -> None:
    c1 = _chunk("keep", 0)
    c2 = _chunk("delete", 1)
    chroma_store.add([c1, c2], _embed(["keep", "delete"]))
    chroma_store.delete([c2.id])
    assert chroma_store.count() == 1


def test_chroma_rebuild(chroma_store: ChromaVectorStore) -> None:
    chroma_store.add([_chunk("old", 0)], _embed(["old"]))
    new_chunks = [_chunk("n1", 0), _chunk("n2", 1)]
    chroma_store.rebuild(new_chunks, _embed(["n1", "n2"]))
    assert chroma_store.count() == 2


def test_chroma_reset(chroma_store: ChromaVectorStore) -> None:
    chroma_store.add([_chunk("x", 0)], _embed(["x"]))
    chroma_store.reset()
    assert chroma_store.count() == 0


def test_chroma_metadata_roundtrip(chroma_store: ChromaVectorStore) -> None:
    c = Chunk(
        id=str(uuid.uuid4()),
        content="flexible metadata",
        metadata={"tag": "a", "score": 0.42, "nested": {"deep": True}},
        index=0,
        start_char=0,
        end_char=18,
    )
    chroma_store.add([c], _embed(["flexible metadata"]))
    hits = chroma_store.search(_embed(["flexible metadata"])[0], top_k=1)
    assert hits
    chunk, _ = hits[0]
    assert chunk.metadata.get("tag") == "a"
    # Nested metadata comes back as JSON-parsed dict.
    assert chunk.metadata.get("nested") == {"deep": True}


def test_localkb_with_chroma_backend(tmp_path: object) -> None:
    """LocalKB pointed at a Chroma backend end-to-end."""
    tmp_path_str = str(tmp_path)
    collection = f"kb_{uuid.uuid4().hex[:8]}"
    kb = LocalKB(
        name="chroma-kb",
        path=tmp_path_str,
        embedder=SimpleEmbedder(dimensions=DIM),
        search_type="vector",
        chunk_size=128,
        chunk_overlap=16,
        vector_store=ChromaVectorStore(collection=collection, dimension=DIM),
        persist=False,  # skip sqlite for this focused test
    )
    kb.add("The mitochondria is the powerhouse of the cell.")
    kb.add("Octopuses have three hearts and blue blood.")
    results = kb.search("octopus", top_k=1)
    assert results
    # With SimpleEmbedder + small corpus, both texts are candidates; just
    # confirm we got a result back from Chroma without error.
    assert results[0].chunk.content
