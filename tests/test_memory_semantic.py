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


def test_forget_leaves_the_other_facts_findable(db):
    """After any ``forget`` the semantic index went empty while its chunk list
    stayed, so every later lookup read the wrong fact."""
    from fastaiagent import Memory

    index = FaissVectorStore(dimension=len(_VOCAB), index_type="flat")
    mem = Memory(location=MemoryStore(db_path=str(db)), semantic=index, embedder=_KeywordEmbedder())
    mem.persist("The user is allergic to peanuts", tier="user", id="alice")
    mem.persist("Bob works with data pipelines", tier="user", id="bob")
    mem.forget(tier="user", id="bob")

    top = mem.retrieve("any peanut concerns?", tier="user", id="alice", limit=1)
    assert [f.fact for f in top] == ["The user is allergic to peanuts"]
    mem.persist("The user enjoys mountain biking", tier="user", id="alice")
    top = mem.retrieve("weekend biking plans", tier="user", id="alice", limit=1)
    assert [f.fact for f in top] == ["The user enjoys mountain biking"]


# --- 1.81.0: the index follows the store -----------------------------------------


def test_a_restarted_process_finds_facts_by_meaning(db):
    """The FAISS index lived only in the process that wrote the facts: after a
    restart ``retrieve(query)`` returned ``[]`` while the facts were still
    stored."""
    from fastaiagent import Memory

    def fresh() -> Memory:  # a new process: same database, empty index
        index = FaissVectorStore(dimension=len(_VOCAB), index_type="flat")
        return Memory(
            location=MemoryStore(db_path=str(db)), semantic=index, embedder=_KeywordEmbedder()
        )

    fresh().persist("The user is allergic to peanuts", tier="user", id="alice")
    top = fresh().retrieve("any peanut concerns?", tier="user", id="alice", limit=1)
    assert [f.fact for f in top] == ["The user is allergic to peanuts"]


def test_facts_written_by_another_memory_are_found(db):
    from fastaiagent import Memory

    def mem() -> Memory:
        index = FaissVectorStore(dimension=len(_VOCAB), index_type="flat")
        return Memory(
            location=MemoryStore(db_path=str(db)), semantic=index, embedder=_KeywordEmbedder()
        )

    reader, writer = mem(), mem()
    reader.persist("The user enjoys mountain biking", tier="user", id="alice")
    writer.persist("The user works with data pipelines", tier="user", id="alice")
    top = reader.retrieve("data work", tier="user", id="alice", limit=1)
    assert [f.fact for f in top] == ["The user works with data pipelines"]


def test_other_subjects_cannot_crowd_out_a_subjects_facts(db):
    store = _semantic_store(db)
    for i in range(30):
        store.add(Fact(scope="user", scope_id=f"u{i}", fact=f"peanut note {i}"))
    store.add(Fact(scope="user", scope_id="alice", fact="alice peanut fact"))
    hits = store.search("peanut", scope="user", scope_id="alice", top_k=1)
    assert [f.fact for f, _ in hits] == ["alice peanut fact"]


def test_qdrant_index_round_trip(db):
    """Vector ids were ``str(fact_id)`` — ``"1"`` — which Qdrant rejects, and the
    error was swallowed: semantic memory on Qdrant never indexed anything."""
    pytest.importorskip("qdrant_client")
    from fastaiagent import Memory
    from fastaiagent.kb.backends.qdrant import QdrantVectorStore

    index = QdrantVectorStore(
        collection="semantic_facts", dimension=len(_VOCAB), location=":memory:"
    )
    mem = Memory(location=MemoryStore(db_path=str(db)), semantic=index, embedder=_KeywordEmbedder())
    mem.persist("The user is allergic to peanuts", tier="user", id="alice")
    mem.persist("The user enjoys mountain biking", tier="user", id="alice")
    top = mem.retrieve("weekend biking plans", tier="user", id="alice", limit=1)
    assert [f.fact for f in top] == ["The user enjoys mountain biking"]
    mem.forget(tier="user", id="alice", fact="The user enjoys mountain biking")
    assert [f.fact for f in mem.retrieve("biking", tier="user", id="alice", limit=1)] != [
        "The user enjoys mountain biking"
    ]
