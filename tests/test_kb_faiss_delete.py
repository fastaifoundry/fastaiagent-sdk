"""``FaissVectorStore.delete`` keeps the index and its chunks aligned (1.81.0).

``delete`` kept the surviving chunks but emptied the FAISS index, trusting the
caller to rebuild — and ``SemanticFactStore`` never did. Every later search
mapped index positions onto the wrong chunks: after one ``forget``, a query about
biking returned the peanut-allergy fact. It now rebuilds from the survivors'
own stored vectors. A failed ``add`` could also leave a chunk with no vector.

No mocking: real FAISS, for every index type the store offers.
"""

from __future__ import annotations

import pytest

pytest.importorskip("faiss")

from fastaiagent.kb.backends.faiss import FaissVectorStore  # noqa: E402
from fastaiagent.kb.chunking import Chunk  # noqa: E402

DIM = 16


def _unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, content=f"chunk {cid}")


@pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivf"])
def test_delete_keeps_every_survivor_findable(index_type):
    store = FaissVectorStore(dimension=DIM, index_type=index_type)
    store.add([_chunk(f"c{i}") for i in range(6)], [_unit(i) for i in range(6)])

    store.delete(["c2"])

    assert store.count() == 5
    for i in (0, 1, 3, 4, 5):
        assert store.search(_unit(i), 1)[0][0].id == f"c{i}"
    # A vector added after the delete maps to its own chunk, not a survivor's.
    store.add([_chunk("new")], [_unit(10)])
    assert store.search(_unit(10), 1)[0][0].id == "new"


def test_delete_everything_then_add():
    store = FaissVectorStore(dimension=DIM)
    store.add([_chunk("a"), _chunk("b")], [_unit(0), _unit(1)])
    store.delete(["a", "b"])
    assert store.count() == 0 and store.search(_unit(0), 1) == []
    store.add([_chunk("c")], [_unit(2)])
    assert store.search(_unit(2), 1)[0][0].id == "c"


def test_failed_add_leaves_the_store_unchanged():
    store = FaissVectorStore(dimension=DIM)
    store.add([_chunk("a")], [_unit(0)])
    with pytest.raises(Exception):
        store.add([_chunk("b")], [[0.5] * 8])  # wrong dimension
    assert store.count() == 1
    assert store.search(_unit(0), 5)[0][0].id == "a"
