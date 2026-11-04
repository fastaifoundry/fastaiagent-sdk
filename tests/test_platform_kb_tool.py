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
