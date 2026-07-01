"""Semantic retrieve(query) — deterministic (fake embedder + real FAISS).

No mocking of the store/index logic: a real ``FaissVectorStore`` and the real
``SemanticFactStore``/``Memory`` code run. A tiny keyword embedder makes the
nearest-neighbour outcome deterministic (no model download / network).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent.learn import Fact, MemoryStore, SemanticFactStore

faiss = pytest.importorskip("faiss")  # noqa: F841
from fastaiagent.kb.backends.faiss import FaissVectorStore  # noqa: E402

_VOCAB = ["peanut", "bik", "data"]


class _KeywordEmbedder:
    """One-hot over a tiny vocab by substring — deterministic nearest neighbour."""

    def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            v = [0.0] * len(_VOCAB)
            for i, w in enumerate(_VOCAB):
                if w in low:
                    v[i] = 1.0
            if not any(v):
                v[0] = 0.01  # avoid all-zero
            out.append(v)
        return out


@pytest.fixture
def db(tmp_path: Path, monkeypatch):
    p = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(p))
    reset_config()
    yield p
    reset_config()


def _semantic_store(db):
    index = FaissVectorStore(dimension=len(_VOCAB), index_type="flat")
    return SemanticFactStore(MemoryStore(db_path=str(db)), index, _KeywordEmbedder())


def test_semantic_retrieve_finds_by_meaning(db):
    from fastaiagent import Memory

    index = FaissVectorStore(dimension=len(_VOCAB), index_type="flat")
    mem = Memory(location=MemoryStore(db_path=str(db)), semantic=index, embedder=_KeywordEmbedder())
    mem.persist("The user is allergic to peanuts", tier="user", id="alice")
    mem.persist("The user enjoys mountain biking", tier="user", id="alice")
    mem.persist("The user works with data pipelines", tier="user", id="alice")

    top = mem.retrieve("any peanut concerns?", tier="user", id="alice", limit=1)
    assert [f.fact for f in top] == ["The user is allergic to peanuts"]
    top2 = mem.retrieve("weekend biking plans", tier="user", id="alice", limit=1)
    assert [f.fact for f in top2] == ["The user enjoys mountain biking"]


def test_semantic_retrieve_is_scope_isolated(db):
    from fastaiagent import Memory

    index = FaissVectorStore(dimension=len(_VOCAB), index_type="flat")
    mem = Memory(location=MemoryStore(db_path=str(db)), semantic=index, embedder=_KeywordEmbedder())
    mem.persist("Alice is allergic to peanuts", tier="user", id="alice")
    # Bob has no facts → semantic query returns nothing for bob.
    assert mem.retrieve("peanut", tier="user", id="bob") == []
    # And safe-by-default: no id → nothing.
    assert mem.retrieve("peanut", tier="user") == []


def test_semantic_search_skips_superseded(db):
    store = _semantic_store(db)
    old = store.add(Fact(scope="user", scope_id="alice", fact="peanut fact v1"))
    new = store.add(Fact(scope="user", scope_id="alice", fact="peanut fact v2"))
    store.supersede(old, new)
    hits = store.search("peanut", scope="user", scope_id="alice", top_k=5)
    facts = [f.fact for f, _ in hits]
    assert "peanut fact v2" in facts
    assert "peanut fact v1" not in facts
