"""UI: memory.read/memory.write spans render via the real trace-spans API.

Seeds memory spans into local.db and drives the FastAPI app through
``TestClient`` (no mocks) to confirm the span tree + attributes the frontend
renders are served correctly. Also asserts memory + KB retrieval spans coexist.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("bcrypt")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


def _seed_memory_trace(db_path: str) -> str:
    """Insert an agent root with memory.read/write parents + per-block children,
    plus a sibling retrieval span so we can assert coexistence."""
    db = init_local_db(db_path)
    now = datetime.now(tz=timezone.utc).isoformat()
    trace_id = uuid.uuid4().hex
    root = uuid.uuid4().hex

    def ins(span_id, parent, name, attrs):
        db.execute(
            """INSERT INTO spans (span_id, trace_id, parent_span_id, name,
                                   start_time, end_time, status, attributes, events)
               VALUES (?, ?, ?, ?, ?, ?, 'OK', ?, '[]')""",
            (span_id, trace_id, parent, name, now, now, json.dumps(attrs)),
        )

    read = uuid.uuid4().hex
    write = uuid.uuid4().hex
    vec = uuid.uuid4().hex
    retr = uuid.uuid4().hex
    try:
        ins(root, None, "agent.assistant", {"agent.name": "assistant"})
        ins(read, root, "memory.read", {
            "fastaiagent.runner.type": "memory",
            "memory.operation": "read",
            "memory.block_count": 1,
            "memory.message_count": 2,
            "memory.query": "what is my email",
        })
        ins(vec, read, "memory.read.vector", {
            "fastaiagent.runner.type": "memory",
            "memory.block_name": "vector",
            "memory.block_type": "VectorBlock",
            "memory.rendered_count": 2,
            "memory.scores": json.dumps([0.91, 0.55]),
            "memory.snippets": json.dumps(["[user] prior exchange", "[assistant] reply"]),
        })
        ins(write, root, "memory.write", {
            "fastaiagent.runner.type": "memory",
            "memory.operation": "write",
            "memory.messages_added": 1,
        })
        ins(retr, root, "retrieval.docs", {
            "fastaiagent.runner.type": "retrieval",
            "retrieval.backend": "local",
            "retrieval.top_k": 3,
        })
    finally:
        db.close()
    return trace_id


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from fastaiagent._internal.config import reset_config

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    trace_id = _seed_memory_trace(str(db_path))
    app = build_app(db_path=str(db_path), no_auth=True)
    return TestClient(app), trace_id


def _flatten(node, acc):
    acc.append(node["span"])
    for child in node.get("children", []):
        _flatten(child, acc)
    return acc


def test_memory_spans_render_in_span_tree(client):
    tc, trace_id = client
    r = tc.get(f"/api/traces/{trace_id}/spans")
    assert r.status_code == 200
    spans = _flatten(r.json()["tree"], [])
    names = {s["name"] for s in spans}
    assert "memory.read" in names
    assert "memory.write" in names
    assert "memory.read.vector" in names

    vec = next(s for s in spans if s["name"] == "memory.read.vector")
    scores = json.loads(vec["attributes"]["memory.scores"])
    assert scores == [0.91, 0.55]
    assert vec["attributes"]["memory.block_type"] == "VectorBlock"


def test_memory_and_retrieval_spans_coexist(client):
    tc, trace_id = client
    r = tc.get(f"/api/traces/{trace_id}/spans")
    spans = _flatten(r.json()["tree"], [])
    names = {s["name"] for s in spans}
    assert "retrieval.docs" in names
    assert "memory.read" in names
    # Distinguishable by runner.type.
    by_name = {s["name"]: s for s in spans}
    assert by_name["retrieval.docs"]["attributes"]["fastaiagent.runner.type"] == "retrieval"
    assert by_name["memory.read"]["attributes"]["fastaiagent.runner.type"] == "memory"
