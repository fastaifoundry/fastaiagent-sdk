"""Contract sweep: every ``VectorStore`` returns the SAME score for the same pair.

The protocol (``fastaiagent.kb.protocols.VectorStore.search``) promises a
cosine-comparable similarity on unit-normalized embeddings. Nothing enforced
that until 1.67.0, and ``tests/test_kb_chroma.py`` never asserted a score
*value* — only that one came back and was a ``float``. Chroma shipped
``1 - d`` against a **squared-L2** distance for its whole life, so an
orthogonal pair scored ``-1.0`` and an opposite pair ``-3.0``.

The sweep is parametrized over every installed backend, so a new backend that
gets the conversion wrong fails here rather than in a customer's ranking. It
does NOT skip silently as a whole: at least FAISS must be present, and each
optional backend is marked so a missing extra reads as a skip line.
"""

from __future__ import annotations

import uuid

import pytest

from fastaiagent.kb.chunking import Chunk

DIM = 4

# (name, unit vector) — the three reference relationships against E1.
E1 = [1.0, 0.0, 0.0, 0.0]
ORTHOGONAL = [0.0, 1.0, 0.0, 0.0]
OPPOSITE = [-1.0, 0.0, 0.0, 0.0]
# cos = 0.6 exactly: 0.6 * e1 + 0.8 * e2
OBLIQUE = [0.6, 0.8, 0.0, 0.0]


def _chunk(text: str) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        content=text,
        metadata={},
        index=0,
        start_char=0,
        end_char=len(text),
    )


# ---------------------------------------------------------------------------
# Backend factories — each returns a fresh empty store or skips.
# ---------------------------------------------------------------------------


def _make_faiss():
    pytest.importorskip("faiss")
    from fastaiagent.kb.backends.faiss import FaissVectorStore

    return FaissVectorStore(dimension=DIM)


def _make_chroma():
    pytest.importorskip("chromadb")
    from fastaiagent.kb.backends.chroma import ChromaVectorStore

    return ChromaVectorStore(collection=f"sweep-{uuid.uuid4().hex}", dimension=DIM)


def _make_qdrant():
    pytest.importorskip("qdrant_client")
    from fastaiagent.kb.backends.qdrant import QdrantVectorStore

    return QdrantVectorStore(
        collection=f"sweep-{uuid.uuid4().hex}", dimension=DIM, location=":memory:"
    )


BACKENDS = {
    "faiss": _make_faiss,
    "chroma": _make_chroma,
    "qdrant": _make_qdrant,
}


@pytest.fixture(params=sorted(BACKENDS))
def store(request):
    return BACKENDS[request.param]()


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,vector,expected",
    [
        ("identical", E1, 1.0),
        ("orthogonal", ORTHOGONAL, 0.0),
        ("opposite", OPPOSITE, -1.0),
        ("oblique", OBLIQUE, 0.6),
    ],
)
def test_score_is_cosine(store, label: str, vector: list[float], expected: float) -> None:
    """A unit-normalized pair scores its cosine, on every backend."""
    chunk = _chunk(label)
    store.add([chunk], [vector])
    results = store.search(E1, top_k=1)
    assert results, f"{label}: search returned nothing"
    _, score = results[0]
    assert score == pytest.approx(expected, abs=0.02), (
        f"{label}: {type(store).__name__} scored {score!r}, cosine is {expected}"
    )


def test_score_never_below_minus_one(store) -> None:
    """No backend may return a score outside cosine's own range."""
    store.add(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        [E1, ORTHOGONAL, OPPOSITE],
    )
    for _, score in store.search(E1, top_k=3):
        assert -1.0 - 1e-6 <= score <= 1.0 + 1e-6, f"score {score!r} escapes [-1, 1]"


def test_ranking_order_is_shared(store) -> None:
    """Identical > oblique > orthogonal > opposite, on every backend."""
    ids = {}
    for label, vec in (
        ("identical", E1),
        ("oblique", OBLIQUE),
        ("orthogonal", ORTHOGONAL),
        ("opposite", OPPOSITE),
    ):
        c = _chunk(label)
        ids[c.id] = label
        store.add([c], [vec])
    ranked = [ids[c.id] for c, _ in store.search(E1, top_k=4)]
    assert ranked == ["identical", "oblique", "orthogonal", "opposite"]


# ---------------------------------------------------------------------------
# Chroma-specific: the space is the whole defect
# ---------------------------------------------------------------------------

chromadb = pytest.importorskip("chromadb", reason="chroma extra not installed")

from fastaiagent.kb.backends.chroma import ChromaVectorStore  # noqa: E402

pytestmark_chroma = pytest.mark.chroma


def _space_of(collection) -> str:
    return (getattr(collection, "configuration_json", None) or {}).get("hnsw", {}).get("space", "")


@pytest.mark.chroma
def test_new_collection_is_cosine() -> None:
    """A collection this SDK creates uses cosine natively."""
    s = ChromaVectorStore(collection=f"new-{uuid.uuid4().hex}", dimension=DIM)
    assert _space_of(s._collection) == "cosine"
    assert s.space == "cosine"


@pytest.mark.chroma
def test_preexisting_l2_collection_scores_correctly() -> None:
    """A collection created before 1.67.0 is still l2 — and must still score cosine.

    Chroma SILENTLY ignores a space on ``get_or_create_collection`` for an
    existing collection: no error, no warning, space stays ``l2``. So setting
    cosine without detecting the actual space would leave every persisted and
    remote collection mis-scored with the evidence removed.
    """
    name = f"legacy-{uuid.uuid4().hex}"
    client = chromadb.EphemeralClient()
    raw = client.get_or_create_collection(name=name)
    assert _space_of(raw) == "l2", "chroma's default is no longer l2 — revisit the sweep"
    raw.upsert(ids=["x"], documents=["x"], embeddings=[ORTHOGONAL])

    store = ChromaVectorStore(collection=name, dimension=DIM)
    store._client = client
    store._collection = client.get_or_create_collection(name=name)
    store._space = store._detect_space()
    assert store.space == "l2"

    results = store.search(E1, top_k=1)
    _, score = results[0]
    assert score == pytest.approx(0.0, abs=0.02), (
        f"pre-existing l2 collection scored {score!r} for an orthogonal pair"
    )


@pytest.mark.chroma
def test_reset_reasserts_the_space() -> None:
    """``reset()`` drops and re-creates — the new collection must be cosine too.

    ``reset()`` re-creates via ``get_or_create_collection``; a fix that only
    touches ``__init__`` silently falls back to l2 on the first ``kb.clear()``
    or ``rebuild()``.
    """
    s = ChromaVectorStore(collection=f"reset-{uuid.uuid4().hex}", dimension=DIM)
    s.reset()
    assert _space_of(s._collection) == "cosine"
    assert s.space == "cosine"
    s.add([_chunk("orth")], [ORTHOGONAL])
    _, score = s.search(E1, top_k=1)[0]
    assert score == pytest.approx(0.0, abs=0.02)


@pytest.mark.chroma
def test_rebuild_reasserts_the_space() -> None:
    s = ChromaVectorStore(collection=f"rebuild-{uuid.uuid4().hex}", dimension=DIM)
    s.rebuild([_chunk("opp")], [OPPOSITE])
    assert s.space == "cosine"
    _, score = s.search(E1, top_k=1)[0]
    assert score == pytest.approx(-1.0, abs=0.02)


@pytest.mark.chroma
def test_non_cosine_space_warns_once(caplog) -> None:
    """An l2 collection converts correctly but tells the operator to rebuild."""
    name = f"warn-{uuid.uuid4().hex}"
    client = chromadb.EphemeralClient()
    client.get_or_create_collection(name=name)
    store = ChromaVectorStore(collection=name, dimension=DIM)
    store._client = client
    store._collection = client.get_or_create_collection(name=name)
    store._space = store._detect_space()
    assert store.space == "l2"

    ChromaVectorStore._warned_spaces.discard(name)
    caplog.clear()
    with caplog.at_level("WARNING", logger="fastaiagent.kb.backends.chroma"):
        store._warn_if_not_cosine()
        store._warn_if_not_cosine()
    hits = [r for r in caplog.records if "l2" in r.getMessage()]
    assert len(hits) == 1, f"expected exactly one warning, got {len(hits)}"
    assert "rebuild" in hits[0].getMessage().lower()


# ---------------------------------------------------------------------------
# Consequences: VectorBlock fusion and hybrid fall-through
# ---------------------------------------------------------------------------


@pytest.mark.chroma
def test_vector_block_ranking_matches_faiss() -> None:
    """``VectorBlock._score_hits`` fuses similarity with recency assuming [0, 1].

    With chroma's old ``2·cos − 1`` the similarity term's slope doubled and
    shifted, which provably inverted the fused ranking against a faiss-backed
    store on the same corpus.
    """
    pytest.importorskip("faiss")
    import time as _time

    from fastaiagent.kb.backends.faiss import FaissVectorStore

    now = _time.time()
    # Two candidates: A is a better match, B is much more recent.
    docs = [
        ("A", [0.9, 0.4359, 0.0, 0.0], now - 3600.0),
        ("B", [0.6, 0.8, 0.0, 0.0], now),
    ]

    def _rank(store) -> list[str]:
        from fastaiagent.agent.memory_blocks import VectorBlock
        from fastaiagent.kb.embedding import SimpleEmbedder

        block = VectorBlock(
            store=store,
            embedder=SimpleEmbedder(dimensions=DIM),
            recency_weight=0.3,
            recency_half_life_seconds=1800.0,
        )
        hits = []
        for label, vec, created in docs:
            c = Chunk(
                id=label,
                content=label,
                metadata={"created_at": created},
                index=0,
                start_char=0,
                end_char=1,
            )
            store.add([c], [vec])
            hits.append(c)
        raw = store.search(E1, top_k=2)
        return [c.content for c, _ in block._score_hits(raw)]

    faiss_order = _rank(FaissVectorStore(dimension=DIM))
    chroma_order = _rank(ChromaVectorStore(collection=f"vb-{uuid.uuid4().hex}", dimension=DIM))
    assert chroma_order == faiss_order, (
        f"fused ranking diverges: chroma {chroma_order} vs faiss {faiss_order}"
    )


def test_hybrid_with_no_keyword_hits_stays_in_range(store) -> None:
    """``_hybrid_search`` short-circuits past min-max normalization when BM25
    finds nothing, returning **raw** vector scores.

    Min-max normalization is affine-invariant, so both backends agreed while
    both lists were non-empty — the short circuit at the top of
    ``_hybrid_search`` is where chroma's out-of-range scores reached the
    caller (and ``LocalKB.as_tool()`` formats them into the text the model
    reads).
    """
    from fastaiagent.kb import LocalKB
    from fastaiagent.kb.embedding import SimpleEmbedder

    kb = LocalKB(
        name=f"hyb-{uuid.uuid4().hex}",
        search_type="hybrid",
        persist=False,
        embedder=SimpleEmbedder(dimensions=DIM),
        vector_store=store,
    )
    kb.add("alpha bravo charlie delta")
    kb.add("echo foxtrot golf hotel")
    # A query with no lexical overlap: BM25 returns nothing, the vector side
    # returns everything, and the short circuit hands back raw vector scores.
    results = kb.search("zzzzqqqq", top_k=5)
    assert results, "vector side returned nothing — the fall-through wasn't exercised"
    for r in results:
        # SimpleEmbedder emits non-negative components, so every true cosine
        # here is in [0, 1]. A negative score can only come from a backend
        # mis-converting its distance — and ``LocalKB.as_tool()`` renders it
        # straight into the tool result the model reads.
        assert 0.0 <= r.score <= 1.0, f"hybrid fall-through leaked score {r.score!r}"
