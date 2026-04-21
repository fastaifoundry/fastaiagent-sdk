"""KB browser route tests.

Use a real ``LocalKB`` (SimpleEmbedder so no network, ``faiss``-backed vectors)
seeded into a temporary directory. Each test hits the FastAPI app via
``TestClient`` to cover the real router wiring end to end — no mocks per the
``feedback_no_mocking`` rule.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")
pytest.importorskip("faiss")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.kb.local import LocalKB  # noqa: E402
from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _seed_kb(kb_root: Path, name: str, doc_dir: Path) -> LocalKB:
    """Seed a KB by ingesting two real files from ``doc_dir``.

    Going through the disk path means ``doc.source`` gets populated with the
    real file path, which is how a normal user ingests (``kb.add("docs/")``).
    """
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "refunds.md").write_text(
        "Refunds are processed within 7 business days after receiving the return. "
        "Customers must include the original packing slip.\n"
    )
    (doc_dir / "shipping.md").write_text(
        "Shipping is free on orders over $50. Expedited shipping is available at "
        "checkout for an extra charge.\n"
    )
    # Use the environment's default embedder so that the search route
    # (which doesn't pass an embedder) loads the same one at query time —
    # otherwise LocalKB raises on embedding-dimension mismatch.
    kb = LocalKB(
        name=name,
        path=str(kb_root),
        chunk_size=120,
        chunk_overlap=20,
    )
    kb.add(str(doc_dir / "refunds.md"))
    kb.add(str(doc_dir / "shipping.md"))
    return kb


def _seed_retrieval_span(db_path: str, kb_name: str, agent_name: str) -> str:
    """Insert one retrieval.<kb> span and its parent agent root into local.db."""
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    trace_id = uuid.uuid4().hex
    root_span = uuid.uuid4().hex
    child_span = uuid.uuid4().hex
    try:
        db.execute(
            """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                                   start_time, end_time, status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
            (
                root_span,
                trace_id,
                None,
                f"agent.{agent_name}",
                now,
                now,
                json.dumps({"agent.name": agent_name}),
            ),
        )
        db.execute(
            """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                                   start_time, end_time, status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
            (
                child_span,
                trace_id,
                root_span,
                f"retrieval.{kb_name}",
                now,
                now,
                json.dumps(
                    {
                        "agent.name": agent_name,
                        "retrieval.backend": "local",
                        "retrieval.top_k": 3,
                    }
                ),
            ),
        )
    finally:
        db.close()
    return trace_id


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    monkeypatch.setenv("FASTAIAGENT_KB_DIR", str(kb_root))
    _seed_kb(kb_root, "docs", tmp_path / "source-files")

    db_path = tmp_path / "local.db"
    _seed_retrieval_span(str(db_path), "docs", "support-bot")

    app = build_app(
        db_path=str(db_path),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
    )
    return app, kb_root, db_path


def test_list_collections_reports_counts(app_env):
    app, kb_root, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/kb")
    assert r.status_code == 200
    body = r.json()
    assert Path(body["root"]) == kb_root
    assert len(body["collections"]) == 1
    c = body["collections"][0]
    assert c["name"] == "docs"
    assert c["chunk_count"] >= 2
    assert c["doc_count"] == 2
    assert c["size_bytes"] > 0


def test_list_collections_is_empty_when_root_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_KB_DIR", str(tmp_path / "nope"))
    app = build_app(
        db_path=str(tmp_path / "local.db"),
        auth_path=tmp_path / "auth.json",
        no_auth=True,
    )
    with TestClient(app) as client:
        r = client.get("/api/kb")
    assert r.status_code == 200
    assert r.json()["collections"] == []


def test_collection_detail_surfaces_metadata_keys(app_env):
    app, _, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/kb/docs")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "docs"
    assert body["chunk_count"] >= 2
    assert "source" in body["metadata_keys"]


def test_collection_detail_404(app_env):
    app, _, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/kb/ghost")
    assert r.status_code == 404


def test_documents_groups_by_source(app_env):
    app, _, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/kb/docs/documents")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    names = {Path(d["source"]).name for d in body["documents"]}
    assert names == {"refunds.md", "shipping.md"}
    for doc in body["documents"]:
        assert doc["chunk_count"] >= 1
        assert doc["preview"]


def test_chunks_for_document(app_env):
    app, _, _ = app_env
    with TestClient(app) as client:
        list_r = client.get("/api/kb/docs/documents")
    src = next(
        d["source"] for d in list_r.json()["documents"]
        if Path(d["source"]).name == "refunds.md"
    )
    with TestClient(app) as client:
        r = client.get("/api/kb/docs/chunks", params={"source": src})
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == src
    assert len(body["chunks"]) >= 1
    assert all("content" in c for c in body["chunks"])


def test_search_returns_ranked_results(app_env):
    app, _, _ = app_env
    with TestClient(app) as client:
        r = client.post(
            "/api/kb/docs/search",
            json={"query": "refund policy", "top_k": 2},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "refund policy"
    assert body["top_k"] == 2
    assert len(body["results"]) >= 1
    top = body["results"][0]
    assert "refund" in top["content"].lower()
    assert Path(top["source"] or "").name == "refunds.md"
    assert isinstance(top["score"], float)


def test_search_rejects_blank_query(app_env):
    app, _, _ = app_env
    with TestClient(app) as client:
        r = client.post("/api/kb/docs/search", json={"query": "", "top_k": 3})
    assert r.status_code == 422


def test_lineage_aggregates_retrieval_spans(app_env):
    app, _, _ = app_env
    with TestClient(app) as client:
        r = client.get("/api/kb/docs/lineage")
    assert r.status_code == 200
    body = r.json()
    assert body["kb_name"] == "docs"
    assert body["retrieval_count"] == 1
    assert body["agents"] == [{"agent_name": "support-bot", "retrieval_count": 1}]
    assert len(body["recent_traces"]) == 1
    assert body["recent_traces"][0]["agent_name"] == "support-bot"
