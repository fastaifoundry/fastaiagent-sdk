"""``PlatformKB.as_tool()`` hands the model the whole chunk (1.99.0).

The tool cut every hit to its first 200 characters. Live, against hosted KBs whose
facts sat past character 200 of a ~450-character chunk, a swarm answered with 0 of
5 facts through the stock tool and 5 of 5 through a full-chunk one; agents
re-searched three times per hop hunting for the missing text. ``LocalKB.as_tool()``
never cut.

No mocks: a real stdlib HTTP server stands in for the plane's KB-search endpoint.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fastaiagent.kb.platform import PlatformKB

FILLER = "This policy applies to every consumer plan sold after January 2026. " * 4
CHUNK = FILLER + "Annual plans cancelled within 30 days carry a processing fee of $17.40."


@pytest.fixture
def plane_kb() -> Iterator[None]:
    from fastaiagent.client import _connection

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps(
                {"results": [{"chunk_id": "c1", "content": CHUNK, "score": 0.81}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    prev = _connection.api_key, _connection.target
    _connection.api_key, _connection.target = "test-key", f"http://{host}:{port}"
    try:
        yield
    finally:
        _connection.api_key, _connection.target = prev
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_the_tool_returns_the_whole_chunk(plane_kb: None) -> None:
    assert CHUNK.index("$17.40") > 200  # the case the cut lost

    out = PlatformKB(kb_id="kb-1").as_tool().execute({"query": "refund fee"}).output

    assert out == f"[Score: 0.810] {CHUNK}"


# ── what the agent sees: name and description (1.99.0) ─────────────────────


def test_a_named_kb_tells_the_agent_what_it_holds(plane_kb: None) -> None:
    """``search_<uuid>`` described by the id alone: an agent holding its own KB
    and a shared one could not tell them apart."""
    kb = PlatformKB(kb_id="7f0e3ac9-uuid", name="shared FAQ", description="escalations, loyalty")
    tool = kb.as_tool()
    assert tool.name == "search_shared_FAQ"
    assert tool.description == (
        "Search the 'shared FAQ' platform knowledge base: escalations, loyalty"
    )


def test_an_unnamed_kb_is_described_as_before() -> None:
    tool = PlatformKB(kb_id="7f0e3ac9-uuid").as_tool()
    assert (tool.name, tool.description) == (
        "search_7f0e3ac9-uuid",
        "Search the '7f0e3ac9-uuid' platform knowledge base",
    )


def test_the_retrieval_span_shows_the_name_and_keeps_the_id(plane_kb: None) -> None:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    from fastaiagent.trace.otel import get_tracer_provider

    class _Spans(SpanExporter):
        spans: list = []

        def export(self, spans):  # type: ignore[override]
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

    col = _Spans()
    get_tracer_provider().add_span_processor(SimpleSpanProcessor(col))
    PlatformKB(kb_id="kb-9", name="policy").search("fee", top_k=1)
    span = next(s for s in col.spans if s.name.startswith("retrieval."))
    assert span.attributes["retrieval.kb_name"] == "policy"
    assert span.attributes["retrieval.kb_id"] == "kb-9"


def test_local_kb_takes_a_description_and_a_safe_tool_name() -> None:
    from fastaiagent.kb import LocalKB

    tool = LocalKB(name="Product docs", persist=False, description="setup guides").as_tool()
    assert tool.name == "search_Product_docs"
    assert tool.description == "Search the 'Product docs' knowledge base: setup guides"
    assert LocalKB(name="research-kb", persist=False).as_tool().name == "search_research-kb"
