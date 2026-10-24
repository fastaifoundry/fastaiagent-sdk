"""A KB touches only its own points in an explicit vector store (1.93.0).

``backends.md`` offers explicit construction "to share a store across KBs" and
to point at an existing Qdrant or Chroma collection. Before 1.93.0 every delete,
update and load rebuilt the store from the KB's in-memory chunks, so one
``kb.delete(...)`` dropped every other KB's points — or a whole existing
collection this KB had not loaded. Each case runs on FAISS, Qdrant (in-process)
and Chroma (ephemeral and on disk), with the real libraries.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.kb import LocalKB
from fastaiagent.kb.embedding import SimpleEmbedder

DIM = 32


def _faiss(_: Path) -> Any:
    pytest.importorskip("faiss")
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    return FaissVectorStore(dimension=DIM)


def _qdrant(_: Path) -> Any:
    pytest.importorskip("qdrant_client")
    from fastaiagent.kb.backends.qdrant import QdrantVectorStore

    return QdrantVectorStore(
        collection=f"shared_{uuid.uuid4().hex[:8]}", dimension=DIM, location=":memory:"
    )


def _chroma(_: Path) -> Any:
    pytest.importorskip("chromadb")
    from fastaiagent.kb.backends.chroma import ChromaVectorStore

    return ChromaVectorStore(collection=f"shared_{uuid.uuid4().hex[:8]}", dimension=DIM)


STORES: dict[str, Callable[[Path], Any]] = {"faiss": _faiss, "qdrant": _qdrant, "chroma": _chroma}


@pytest.fixture(params=sorted(STORES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    return STORES[request.param](tmp_path)


def _kb(name: str, store: Any, tmp_path: Path, persist: bool = False) -> LocalKB:
    return LocalKB(
        name=name,
        path=str(tmp_path / "kb"),
        search_type="vector",
        embedder=SimpleEmbedder(dimensions=DIM),
        vector_store=store,
        persist=persist,
    )


def _two_kbs(store: Any, tmp_path: Path) -> tuple[LocalKB, LocalKB]:
    a, b = _kb("a", store, tmp_path), _kb("b", store, tmp_path)
    a.add("Refunds are issued within 14 days.", metadata={"source": "refunds"})
    a.add("Shipping takes three to five days.", metadata={"source": "shipping"})
    b.add("Penguins live in the southern hemisphere.", metadata={"source": "penguins"})
    b.add("Owls hunt at night.", metadata={"source": "owls"})
    assert store.count() == 4
    return a, b


def test_delete_leaves_the_other_kbs_points(store: Any, tmp_path: Path) -> None:
    a, b = _two_kbs(store, tmp_path)
    assert b.delete(b._chunks[-1].id) is True
    assert store.count() == 3
    assert "Refunds" in a.search("refunds issued days", top_k=1)[0].chunk.content


def test_delete_by_source_leaves_the_other_kbs_points(store: Any, tmp_path: Path) -> None:
    a, b = _two_kbs(store, tmp_path)
    assert b.delete_by_source("penguins") == 1
    assert store.count() == 3


def test_deleting_the_last_chunk_does_not_empty_the_store(store: Any, tmp_path: Path) -> None:
    a, b = _two_kbs(store, tmp_path)
    for chunk in list(b._chunks):
        b.delete(chunk.id)
    assert store.count() == 2


def test_update_leaves_the_other_kbs_points(store: Any, tmp_path: Path) -> None:
    a, b = _two_kbs(store, tmp_path)
    assert b.update(b._chunks[0].id, "Emperor penguins breed in winter.") is True
    assert store.count() == 4
    hit = b.search("emperor penguins breed winter", top_k=1)[0].chunk
    assert hit.content == "Emperor penguins breed in winter."


def test_loading_a_persisted_kb_keeps_the_other_kbs_points(store: Any, tmp_path: Path) -> None:
    a = _kb("a", store, tmp_path, persist=True)
    a.add("Refunds are issued within 14 days.")
    other = _kb("other", store, tmp_path)
    other.add("Owls hunt at night.")
    assert store.count() == 2
    _kb("a", store, tmp_path, persist=True)  # reopen: replays a's chunks
    assert store.count() == 2


def test_a_chunk_only_in_an_existing_collection_can_be_deleted(tmp_path: Path) -> None:
    pytest.importorskip("chromadb")
    from fastaiagent.kb.backends.chroma import ChromaVectorStore

    def on_disk() -> ChromaVectorStore:
        return ChromaVectorStore(
            collection="existing", dimension=DIM, persist_path=str(tmp_path / "chroma")
        )

    writer = _kb("writer", on_disk(), tmp_path)
    writer.add("Refunds are issued within 14 days.")
    writer.add("Owls hunt at night.")

    store = on_disk()  # a later process: nothing in memory, 2 points on disk
    reader = _kb("reader", store, tmp_path)
    hit = reader.search("owls night", top_k=1)[0].chunk
    assert reader.delete(hit.id) is True
    assert store.count() == 1
    assert reader.delete(hit.id) is False
