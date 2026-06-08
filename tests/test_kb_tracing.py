"""Live tests — LocalKB.search emits a retrieval.<kb> span with correct attrs.

No mocking of OTel or the KB. We wire up a real LocalKB instance with an
in-memory vector store, run a search, then read the spans back out of
``local.db`` to verify the span shape the UI depends on.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fastaiagent._internal.config import reset_config
from fastaiagent._internal.storage import SQLiteHelper
from fastaiagent.trace.otel import reset as reset_tracer


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(tmp_path / "local.db"))
    reset_config()
    reset_tracer()
    yield tmp_path / "local.db"
    reset_tracer()
    reset_config()


def _read_spans(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    with SQLiteHelper(db_path) as db:
        existing = db.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='spans'"
        )
        if not existing:
            return []
        return db.fetchall("SELECT * FROM spans ORDER BY start_time")


def _attrs(row: dict) -> dict:
    return json.loads(row.get("attributes") or "{}")


class TestLocalKBRetrievalSpan:
    """LocalKB with a pure in-memory backend — no FAISS / Qdrant / Chroma needed."""

    def _build_kb(self, tmp_path: Path):
        """Build a minimal LocalKB with in-memory stores + a fake embedder."""
        from fastaiagent.kb.local import LocalKB
        from fastaiagent.kb.chunking import Chunk

        # Fake embedder: deterministic 4-dim vector per word length bucket.
        class FakeEmbedder:
            def embed(self, texts):
                return [[float(len(t)) / 10.0, 0.1, 0.2, 0.3] for t in texts]

        kb = LocalKB(
            name="probe-kb",
            path=str(tmp_path / "kb"),
            embedder=FakeEmbedder(),
            search_type="vector",
            persist=False,
        )
        kb._chunks = [
            Chunk(content="alpha bravo charlie", metadata={"doc_id": "doc-a"}),
            Chunk(content="delta echo foxtrot", metadata={"doc_id": "doc-b"}),
        ]

        # Minimal vector-store stub matching the protocol the code uses.
        class InMemVector:
            def __init__(self):
                self._items = []

            def add(self, chunks, embeddings):
                for c, e in zip(chunks, embeddings):
                    self._items.append((c, e))

            def count(self):
                return len(self._items)

            def search(self, query_emb, top_k):
                # Naive Euclidean scoring, capped at top_k.
                scored = []
                for c, e in self._items:
                    dist = sum((a - b) ** 2 for a, b in zip(query_emb, e)) ** 0.5
                    scored.append((c, 1.0 / (1.0 + dist)))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:top_k]

        kb._vector = InMemVector()
        kb._vector.add(kb._chunks, [[0.3, 0.1, 0.2, 0.3], [0.6, 0.1, 0.2, 0.3]])
        return kb

    def test_search_emits_retrieval_span(self, tmp_path, _isolated_db):
        kb = self._build_kb(tmp_path)
        results = kb.search("alpha", top_k=2)
        assert len(results) == 2

        rows = _read_spans(_isolated_db)
        retrieval = [r for r in rows if r["name"].startswith("retrieval.")]
        assert len(retrieval) == 1
        span = retrieval[0]
        assert span["name"] == "retrieval.probe-kb"
        attrs = _attrs(span)
        assert attrs["fastaiagent.runner.type"] == "retrieval"
        assert attrs["retrieval.kb_name"] == "probe-kb"
        assert attrs["retrieval.search_type"] == "vector"
        assert attrs["retrieval.top_k"] == 2
        assert attrs["retrieval.result_count"] == 2
        # Payload-gated but default-on: query + doc ids recorded.
        assert attrs["retrieval.query"] == "alpha"
        assert "doc-a" in json.loads(attrs["retrieval.doc_ids"])
        # Local KBs have no registered id → retrieval.kb_id is omitted, the
        # honest "central harness cannot re-execute this" signal.
        assert "retrieval.kb_id" not in attrs

    def test_empty_kb_does_not_emit_span(self, tmp_path, _isolated_db):
        """No hits yet → no retrieval span either."""
        from fastaiagent.kb.local import LocalKB

        class FakeEmbedder:
            def embed(self, texts):
                return [[0.0] * 4 for _ in texts]

        kb = LocalKB(
            name="empty-kb",
            path=str(tmp_path / "kb"),
            embedder=FakeEmbedder(),
            search_type="vector",
            persist=False,
        )
        # No chunks, no vector store — search short-circuits before tracing.
        results = kb.search("anything", top_k=3)
        assert results == []
        rows = _read_spans(_isolated_db)
        assert [r for r in rows if r["name"].startswith("retrieval.")] == []


class TestPayloadGating:
    def test_payload_disabled_strips_query_and_doc_ids(
        self, monkeypatch, tmp_path, _isolated_db
    ):
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")

        from fastaiagent.kb.local import LocalKB
        from fastaiagent.kb.chunking import Chunk

        class FakeEmbedder:
            def embed(self, texts):
                return [[float(len(t))] for t in texts]

        kb = LocalKB(
            name="no-payload-kb",
            path=str(tmp_path / "kb"),
            embedder=FakeEmbedder(),
            search_type="vector",
            persist=False,
        )
        kb._chunks = [Chunk(content="x", metadata={"doc_id": "d1"})]

        class V:
            def count(self):
                return 1

            def search(self, q, k):
                return [(kb._chunks[0], 0.9)]

        kb._vector = V()
        kb.search("secret query", top_k=1)

        rows = _read_spans(_isolated_db)
        span = [r for r in rows if r["name"].startswith("retrieval.")][0]
        attrs = _attrs(span)
        assert "retrieval.query" not in attrs
        assert "retrieval.doc_ids" not in attrs
        # Structural attrs still captured.
        assert attrs["retrieval.result_count"] == 1
        assert attrs["retrieval.top_k"] == 1


class TestRetrievalKbIdEmission:
    """retrieval.kb_id is emitted UNGATED when present, omitted when absent.

    Drives ``retrieval_span`` directly (real OTel pipeline, no network) — the
    helper is the single emission point both PlatformKB and LocalKB funnel
    through.
    """

    def _emit(self, kb_id: str | None) -> None:
        from fastaiagent.kb._tracing import retrieval_span

        with retrieval_span(
            kb_name="probe",
            kb_id=kb_id,
            backend="platform",
            search_type=None,
            query="some query",
            top_k=3,
        ) as handle:
            handle.record([])

    def _span_attrs(self, db_path: Path) -> dict:
        span = [r for r in _read_spans(db_path) if r["name"].startswith("retrieval.")][0]
        return _attrs(span)

    def test_kb_id_emitted_when_present(self, tmp_path, _isolated_db):
        self._emit("kb_abc123")
        assert self._span_attrs(_isolated_db)["retrieval.kb_id"] == "kb_abc123"

    def test_kb_id_omitted_when_none(self, tmp_path, _isolated_db):
        self._emit(None)
        assert "retrieval.kb_id" not in self._span_attrs(_isolated_db)

    def test_kb_id_is_ungated_by_payloads(self, monkeypatch, tmp_path, _isolated_db):
        # kb_id is a routing/index key, not payload: it survives payloads OFF,
        # while retrieval.query (payload) is stripped.
        monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
        self._emit("kb_xyz")
        attrs = self._span_attrs(_isolated_db)
        assert attrs["retrieval.kb_id"] == "kb_xyz"
        assert "retrieval.query" not in attrs


class TestPlatformKBRetrievalKbId:
    """Real PlatformKB.search against a local stand-in server emits retrieval.kb_id.

    No mocks: a real stdlib HTTP server stands in for the platform's KB-search
    endpoint, and the SDK connection is pointed at it.
    """

    def test_platform_kb_search_emits_kb_id(self, tmp_path, _isolated_db):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from fastaiagent.client import _connection

        kb_id = "kb_platform_123"
        captured: dict[str, str] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                captured["path"] = self.path
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = (
                    b'{"results": [{"chunk_id": "c1", "content": "hello",'
                    b' "score": 0.9, "metadata": {"doc_id": "d1"}}]}'
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a, **k) -> None:  # silence per-request logs
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[:2]

        prev_key, prev_target = _connection.api_key, _connection.target
        _connection.api_key = "test-key"
        _connection.target = f"http://{host}:{port}"
        try:
            from fastaiagent.kb.platform import PlatformKB

            results = PlatformKB(kb_id=kb_id).search("hello", top_k=1)
            assert len(results) == 1
        finally:
            _connection.api_key, _connection.target = prev_key, prev_target
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert captured["path"] == f"/public/v1/knowledge-bases/{kb_id}/search"
        span = [
            r for r in _read_spans(_isolated_db) if r["name"].startswith("retrieval.")
        ][0]
        attrs = _attrs(span)
        assert attrs["retrieval.kb_id"] == kb_id
        # PlatformKB.name == kb_id, so kb_name happens to match — but kb_id is
        # the explicit, separate routing key the central harness keys off.
        assert attrs["retrieval.kb_name"] == kb_id


def test_asynchronous_kb_search_nests_under_agent(tmp_path, _isolated_db):
    """Full integration: Agent with a KB tool → retrieval span under tool span."""
    from fastaiagent import Agent
    from fastaiagent.kb.local import LocalKB
    from fastaiagent.kb.chunking import Chunk
    from fastaiagent.llm.client import LLMResponse
    from fastaiagent.llm.message import ToolCall
    from tests.conftest import MockLLMClient

    class FakeEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    kb = LocalKB(
        name="hotspot-kb",
        path=str(tmp_path / "kb"),
        embedder=FakeEmbedder(),
        search_type="vector",
        persist=False,
    )
    kb._chunks = [
        Chunk(
            content="fastaiagent 0.8 ships the Local UI",
            metadata={"doc_id": "d1"},
        )
    ]

    class V:
        def count(self):
            return 1

        def search(self, q, k):
            return [(kb._chunks[0], 0.95)]

    kb._vector = V()

    llm = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="t1", name=f"search_{kb.name}", arguments={"query": "0.8"})
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="Here's what I found.", finish_reason="stop"),
        ]
    )
    agent = Agent(
        name="retrieval-agent",
        system_prompt="Use the KB.",
        llm=llm,
        tools=[kb.as_tool()],
    )
    asyncio.run(agent.arun("what's new?"))

    rows = _read_spans(_isolated_db)
    retrieval = [r for r in rows if r["name"].startswith("retrieval.")]
    tool = [r for r in rows if r["name"].startswith("tool.")]
    agent_spans = [r for r in rows if r["name"].startswith("agent.")]
    assert len(retrieval) == 1
    assert len(tool) == 1
    assert len(agent_spans) == 1

    # retrieval span nests inside the tool span, which nests inside the agent span.
    retrieval_row = retrieval[0]
    tool_row = tool[0]
    agent_row = agent_spans[0]
    assert retrieval_row["parent_span_id"] == tool_row["span_id"]
    assert tool_row["parent_span_id"] == agent_row["span_id"]
    assert agent_row["parent_span_id"] in (None, "")
