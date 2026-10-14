"""``LocalKB.add`` keeps the caller's source, and the UI searches a KB the way it can be searched.

Two defects (1.89.0):

* ``LocalKB.add(text, metadata={"source": "x"})`` overwrote ``source`` with the
  empty ``Document.source``, so ``delete_by_source("x")`` deleted nothing and
  the KB browser grouped every raw text under one blank document.
* The KB browser opened every KB as ``hybrid``. A keyword-only KB built the
  default embedder (a model load) for nothing, and the search response called a
  pure BM25 search "hybrid". There was no way to choose the search type, and a
  search typed into the UI was traced as a production retrieval.

Sweep: keyword-only KB vs a KB with embeddings × search_type omitted / keyword /
vector / hybrid × ``add`` with the source in metadata / on the Document / both,
with ``delete_by_source`` after each. Real ``LocalKB``, real SQLite, real
FastAPI app. No mocks.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.kb.document import Document
from fastaiagent.kb.embedding import SimpleEmbedder
from fastaiagent.kb.local import LocalKB

_HAS_FAISS = importlib.util.find_spec("faiss") is not None
_HAS_FASTEMBED = importlib.util.find_spec("fastembed") is not None
_needs_faiss = pytest.mark.skipif(
    not _HAS_FAISS, reason="vector/hybrid search needs faiss-cpu (the [kb] extra)"
)

#: (search_type, needs an embedder). Keyword-only builds no embedder at all.
KB_KINDS = [
    pytest.param("keyword", id="keyword-only"),
    pytest.param("vector", id="vector", marks=_needs_faiss),
    pytest.param("hybrid", id="hybrid", marks=_needs_faiss),
]


def _kb(tmp_path: Path, search_type: str, name: str = "t") -> LocalKB:
    embedder = None if search_type == "keyword" else SimpleEmbedder(dimensions=16)
    return LocalKB(
        name=name,
        path=str(tmp_path / "kb"),
        embedder=embedder,
        search_type=search_type,  # type: ignore[arg-type]
        chunk_size=200,
        chunk_overlap=0,
    )


def _sources(kb: LocalKB) -> list[str]:
    return sorted(c.metadata.get("source", "") for c in kb._chunks)  # noqa: SLF001


# --------------------------------------------------------------------------- #
# LocalKB.add: the source survives (D11)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("search_type", KB_KINDS)
def test_metadata_source_survives_add_and_delete_by_source(
    tmp_path: Path, search_type: str
) -> None:
    kb = _kb(tmp_path, search_type)
    if search_type == "keyword":
        assert kb._embedder is None  # noqa: SLF001 — no model is built
    kb.add("Refunds take seven days.", metadata={"source": "file-a.txt"})
    kb.add("Returns need the packing slip.", metadata={"source": "file-a.txt"})
    kb.add("Shipping is free over fifty.", metadata={"source": "file-b.txt"})
    assert _sources(kb) == ["file-a.txt", "file-a.txt", "file-b.txt"]

    assert kb.delete_by_source("file-a.txt") == 2
    assert _sources(kb) == ["file-b.txt"]
    remaining = [r.chunk.content for r in kb.search("refunds packing shipping", top_k=5)]
    assert remaining == ["Shipping is free over fifty."]
    kb.close()


@pytest.mark.parametrize("search_type", KB_KINDS)
def test_document_source_still_wins(tmp_path: Path, search_type: str) -> None:
    kb = _kb(tmp_path, search_type)
    kb.add_documents([Document(content="Doc only source.", source="doc.txt")])
    kb.add_documents(
        [Document(content="Both sources.", source="doc2.txt", metadata={"source": "meta.txt"})]
    )
    assert _sources(kb) == ["doc.txt", "doc2.txt"]
    # The metadata value lost to the non-empty Document.source.
    assert kb.delete_by_source("meta.txt") == 0
    assert kb.delete_by_source("doc2.txt") == 1
    assert kb.delete_by_source("doc.txt") == 1
    assert kb._chunks == []  # noqa: SLF001
    kb.close()


@pytest.mark.parametrize("search_type", KB_KINDS)
def test_source_persists_across_reopen(tmp_path: Path, search_type: str) -> None:
    kb = _kb(tmp_path, search_type)
    kb.add("Persisted with its source.", metadata={"source": "keep.txt"})
    kb.add("No source given.")
    kb.close()

    reopened = _kb(tmp_path, search_type)
    assert _sources(reopened) == ["", "keep.txt"]
    assert reopened.delete_by_source("keep.txt") == 1
    # An empty source no longer swallows texts added with a source.
    assert reopened.delete_by_source("") == 1
    assert reopened._chunks == []  # noqa: SLF001
    reopened.close()

    again = _kb(tmp_path, search_type)
    assert again._chunks == []  # noqa: SLF001 — the deletes reached SQLite
    again.close()


def test_same_metadata_source_texts_do_not_replace_each_other(tmp_path: Path) -> None:
    """``Document.source`` is not set from metadata: the metadata store keys
    document rows on it, so two texts sharing a source would overwrite."""
    kb = _kb(tmp_path, "keyword")
    kb.add("First text.", metadata={"source": "shared"})
    kb.add("Second text.", metadata={"source": "shared"})
    assert kb._metadata is not None  # noqa: SLF001
    docs = kb._metadata.list_documents()  # noqa: SLF001
    assert sorted(d.content for d in docs) == ["First text.", "Second text."]
    assert all(d.source == "" for d in docs)
    kb.close()


def test_directory_ingest_keeps_the_file_path(tmp_path: Path) -> None:
    src = tmp_path / "docs"
    src.mkdir()
    (src / "a.md").write_text("Alpha document.")
    kb = _kb(tmp_path, "keyword")
    kb.add(str(src), metadata={"source": "ignored", "team": "x"})
    assert _sources(kb) == [str(src / "a.md")]
    assert kb._chunks[0].metadata["team"] == "x"  # noqa: SLF001
    kb.close()


# --------------------------------------------------------------------------- #
# The KB browser: search type follows what the KB holds (D12)
# --------------------------------------------------------------------------- #

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("itsdangerous")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent._internal.config import reset_config  # noqa: E402
from fastaiagent.trace.otel import reset as reset_tracer  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.routes import kb as kb_routes  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402

_TEXTS = [
    ("Refunds are processed within seven business days.", "refunds.md"),
    ("Refund requests need the original packing slip.", "refunds.md"),
    ("Shipping is free on orders over fifty dollars.", "shipping.md"),
]


@pytest.fixture
def ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    kb_root = tmp_path / "kbroot"
    monkeypatch.setenv("FASTAIAGENT_KB_DIR", str(kb_root))
    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    reset_tracer()
    kb_routes._LOCALKB_CACHE.clear()  # noqa: SLF001

    kw = LocalKB(name="kwonly", path=str(kb_root), search_type="keyword")
    for text, source in _TEXTS:
        kw.add(text, metadata={"source": source})
    kw.close()

    if _HAS_FAISS and _HAS_FASTEMBED:
        # The browser builds the default embedder for a KB with embeddings, so
        # the KB must be written with it (dimensions must match).
        emb = LocalKB(name="embedded", path=str(kb_root))
        for text, source in _TEXTS:
            emb.add(text, metadata={"source": source})
        emb.close()

    init_local_db(db_path).close()
    client = TestClient(build_app(db_path=str(db_path), no_auth=True))
    try:
        yield client, db_path
    finally:
        kb_routes._LOCALKB_CACHE.clear()  # noqa: SLF001
        reset_tracer()
        reset_config()


_needs_embedded = pytest.mark.skipif(
    not (_HAS_FAISS and _HAS_FASTEMBED),
    reason="the KB-with-embeddings axis needs faiss-cpu and fastembed (the [kb] extra): "
    "the browser opens such a KB with the default embedder",
)

#: (kb, requested search_type, expected status, expected search_type reported)
SEARCH_CASES = [
    pytest.param("kwonly", None, 200, "keyword", id="kwonly-omitted"),
    pytest.param("kwonly", "keyword", 200, "keyword", id="kwonly-keyword"),
    pytest.param("kwonly", "vector", 400, None, id="kwonly-vector"),
    pytest.param("kwonly", "hybrid", 400, None, id="kwonly-hybrid"),
    pytest.param("embedded", None, 200, "hybrid", id="embedded-omitted", marks=_needs_embedded),
    pytest.param(
        "embedded", "keyword", 200, "keyword", id="embedded-keyword", marks=_needs_embedded
    ),
    pytest.param("embedded", "vector", 200, "vector", id="embedded-vector", marks=_needs_embedded),
    pytest.param("embedded", "hybrid", 200, "hybrid", id="embedded-hybrid", marks=_needs_embedded),
]


@pytest.mark.parametrize(("kb_name", "requested", "code", "reported"), SEARCH_CASES)
def test_search_type_is_truthful_and_selectable(
    ui: Any, kb_name: str, requested: str | None, code: int, reported: str | None
) -> None:
    client, _ = ui
    body: dict[str, Any] = {"query": "refund packing slip", "top_k": 3}
    if requested is not None:
        body["search_type"] = requested
    r = client.post(f"/api/kb/{kb_name}/search", json=body)
    assert r.status_code == code, r.text
    if code != 200:
        assert "keyword" in r.json()["detail"]
        return
    out = r.json()
    assert out["search_type"] == reported
    assert out["results"], out
    assert out["results"][0]["source"] == "refunds.md"
    assert {res["source"] for res in out["results"]} <= {"refunds.md", "shipping.md"}


def test_keyword_only_kb_builds_no_embedder(ui: Any) -> None:
    client, _ = ui
    r = client.post("/api/kb/kwonly/search", json={"query": "refund", "top_k": 2})
    assert r.status_code == 200, r.text
    opened = [kb for key, kb in kb_routes._LOCALKB_CACHE.items() if key[1] == "kwonly"]  # noqa: SLF001
    assert len(opened) == 1
    assert opened[0].search_type == "keyword"
    assert opened[0]._embedder is None  # noqa: SLF001 — no model load


def test_search_type_rejects_unknown_values(ui: Any) -> None:
    client, _ = ui
    r = client.post("/api/kb/kwonly/search", json={"query": "x", "search_type": "semantic"})
    assert r.status_code == 422


def test_detail_reports_embeddings_and_search_types(ui: Any) -> None:
    client, _ = ui
    d = client.get("/api/kb/kwonly").json()
    assert d["has_embeddings"] is False
    assert d["search_types"] == ["keyword"]
    assert d["search_type"] == "keyword"


@_needs_embedded
def test_detail_reports_embeddings_for_an_embedded_kb(ui: Any) -> None:
    client, _ = ui
    d = client.get("/api/kb/embedded").json()
    assert d["has_embeddings"] is True
    assert d["search_types"] == ["keyword", "vector", "hybrid"]
    assert d["search_type"] == "hybrid"


def test_documents_group_raw_texts_by_their_metadata_source(ui: Any) -> None:
    client, _ = ui
    body = client.get("/api/kb/kwonly/documents").json()
    assert body["total"] == 2
    by_source = {d["source"]: d["chunk_count"] for d in body["documents"]}
    assert by_source == {"refunds.md": 2, "shipping.md": 1}
    chunks = client.get("/api/kb/kwonly/chunks", params={"source": "refunds.md"}).json()
    assert len(chunks["chunks"]) == 2


def test_ui_search_is_traced_as_kb_search_not_production(ui: Any) -> None:
    client, db_path = ui
    r = client.post("/api/kb/kwonly/search", json={"query": "refund", "top_k": 2})
    assert r.status_code == 200, r.text

    db = init_local_db(db_path)
    try:
        rows = db.fetchall("SELECT trace_id, attributes FROM spans WHERE name = 'retrieval.kwonly'")
    finally:
        db.close()
    assert len(rows) == 1, "the UI search wrote no retrieval span to the UI's DB"
    attrs = json.loads(rows[0]["attributes"])
    assert attrs["fastaiagent.source"] == "kb_search"
    assert attrs["fastaiagent.export.local_only"] is True
    assert attrs["retrieval.search_type"] == "keyword"
    trace_id = rows[0]["trace_id"]

    listed = {t["trace_id"]: t for t in client.get("/api/traces?page_size=100").json()["rows"]}
    assert listed[trace_id]["runner_type"] == "retrieval"
    assert listed[trace_id]["source"] == "kb_search"
    production = client.get("/api/traces?production_only=true&page_size=100").json()
    assert trace_id not in {t["trace_id"] for t in production["rows"]}
    assert client.get(f"/api/traces/{trace_id}").json()["source"] == "kb_search"
