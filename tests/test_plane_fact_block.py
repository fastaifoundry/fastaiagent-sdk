"""``PlaneFactBlock`` reads the plane the way the operator configured (1.81.0).

* The user's raw question went to the plane as a URL parameter even with
  ``FASTAIAGENT_TRACE_PAYLOADS=0``; it is now left out (the plane then returns
  the flat, importance-ordered list — ``query`` is optional there).
* With ``refresh_every > 1`` the facts fetched for one question were reused for
  different questions; a new question now refetches.
* A 403 / 404 (feature not enabled, unknown agent) failed silently; it is now
  logged once.

No mocking: a real local HTTP server stands in for the plane and records what
the SDK sends.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from fastaiagent.agent.memory_blocks import PlaneFactBlock
from fastaiagent.client import _connection


class _Plane(BaseHTTPRequestHandler):
    status = 200
    requests: list[dict[str, list[str]]] = []

    def do_GET(self):  # noqa: N802
        type(self).requests.append(parse_qs(urlparse(self.path).query))
        body = json.dumps({"facts": [{"content": "Refunds need a ticket id"}]}).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def plane():
    _Plane.status, _Plane.requests = 200, []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Plane)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    saved = (_connection.api_key, _connection.target)
    _connection.api_key = "fa_k_test"
    _connection.target = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield _Plane
    finally:
        _connection.api_key, _connection.target = saved
        server.shutdown()
        server.server_close()


def test_the_question_is_sent_when_payloads_may_leave(plane, monkeypatch):
    monkeypatch.delenv("FASTAIAGENT_TRACE_PAYLOADS", raising=False)
    out = PlaneFactBlock("support").render("my SSN is 123-45-6789, refund me")
    assert out and "Refunds need a ticket id" in out[0].content
    assert plane.requests[0]["query"] == ["my SSN is 123-45-6789, refund me"]


def test_the_question_stays_home_when_payloads_are_off(plane, monkeypatch):
    monkeypatch.setenv("FASTAIAGENT_TRACE_PAYLOADS", "0")
    out = PlaneFactBlock("support").render("my SSN is 123-45-6789, refund me")
    assert out and "Refunds need a ticket id" in out[0].content  # still recalled
    assert "query" not in plane.requests[0]
    assert plane.requests[0]["agent_id"] == ["support"]


def test_a_new_question_refetches_even_between_refreshes(plane):
    block = PlaneFactBlock("support", refresh_every=5)
    block.render("where is my order?")
    block.render("how do I get a refund?")
    block.render("how do I get a refund?")  # same question: cached
    assert [r["query"] for r in plane.requests] == [
        ["where is my order?"],
        ["how do I get a refund?"],
    ]


def test_a_forbidden_read_is_logged_once(plane, caplog):
    plane.status = 403
    block = PlaneFactBlock("support")
    with caplog.at_level(logging.WARNING, logger="fastaiagent.agent.memory_blocks"):
        for _ in range(3):
            assert block.render("hello") == []
    warnings = [r for r in caplog.records if "HTTP 403" in r.getMessage()]
    assert len(warnings) == 1
