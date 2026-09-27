"""``PlaneFactBlock`` reads the plane the way the operator configured (1.81.0).

* The user's raw question went to the plane as a URL parameter even with
  ``FASTAIAGENT_TRACE_PAYLOADS=0``; it is now left out (the plane then returns
  the flat, importance-ordered list — ``query`` is optional there).
* With ``refresh_every > 1`` the facts fetched for one question were reused for
  different questions; a new question now refetches.
* A 403 / 404 (feature not enabled, unknown agent) failed silently; it is now
  logged once.
* (1.82.0) A fact the plane served twice — two identical rows, or one row per
  matching vector — was injected twice; it is now injected once.

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

_DEFAULT_FACTS = [{"content": "Refunds need a ticket id"}]


class _Plane(BaseHTTPRequestHandler):
    status = 200
    target = ""
    requests: list[dict[str, list[str]]] = []
    facts: list[dict[str, str]] = _DEFAULT_FACTS

    drop = False  # close the connection without a response: a transport failure

    def do_GET(self):  # noqa: N802
        if type(self).drop:
            self.close_connection = True
            return
        type(self).requests.append(parse_qs(urlparse(self.path).query))
        body = json.dumps({"facts": type(self).facts}).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def plane():
    _Plane.status, _Plane.requests, _Plane.facts, _Plane.drop = 200, [], _DEFAULT_FACTS, False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Plane)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    saved = (_connection.api_key, _connection.target)
    _connection.api_key = "fa_k_test"
    _connection.target = f"http://127.0.0.1:{server.server_address[1]}"
    _Plane.target = _connection.target
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


def test_a_fact_served_twice_is_injected_once(plane):
    plane.facts = [
        {"content": "Refunds need a ticket id"},
        {"content": "Orders ship within 2 days"},
        {"content": "refunds  need a ticket ID"},  # same fact, other case/spacing
        {"content": "Refunds need a ticket id"},
    ]
    block = PlaneFactBlock("support")
    out = block.render("refund?")
    assert out[0].content.splitlines()[1:] == [
        "- Refunds need a ticket id",  # the first (most important) copy is kept
        "- Orders ship within 2 days",
    ]
    report = block.last_render_report()
    assert report.rendered_count == 2 and report.deduped_count == 2


# --- 1.82.0: an unreachable plane costs one wait and one warning, not one per turn


@pytest.fixture
def silent_plane():
    """A plane that accepts the connection and never answers — a hung server."""
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    saved = (_connection.api_key, _connection.target)
    _connection.api_key = "fa_k_test"
    _connection.target = f"http://127.0.0.1:{sock.getsockname()[1]}"
    try:
        yield
    finally:
        _connection.api_key, _connection.target = saved
        sock.close()


def test_an_unreachable_plane_is_not_retried_on_every_turn(plane, silent_plane, caplog):
    import time

    block = PlaneFactBlock("support", timeout=0.3)
    hung = _connection.target
    _connection.target = plane.target  # up: fetch the last-known facts
    assert block.render("first question")
    _connection.target = hung  # now the plane hangs

    with caplog.at_level(logging.WARNING, logger="fastaiagent.agent.memory_blocks"):
        started = time.monotonic()
        out = block.render("second question")  # waits for the timeout once
        first_wait = time.monotonic() - started
        started = time.monotonic()
        for q in ("third question", "fourth question"):
            again = block.render(q)
        later_waits = time.monotonic() - started

    assert out and "Refunds need a ticket id" in out[0].content  # last-known facts
    assert again and "Refunds need a ticket id" in again[0].content
    assert first_wait >= 0.25 and later_waits < 0.2  # no wait while backing off
    warnings = [r for r in caplog.records if "PlaneFactBlock" in r.getMessage()]
    assert len(warnings) == 1  # once per outage, not once per turn


def test_the_plane_is_tried_again_after_the_back_off(plane, caplog):
    block = PlaneFactBlock("recovery-agent", timeout=0.3)
    block.retry_after_seconds = 0.0  # no back-off, so the next turn retries
    plane.drop = True  # the plane is down
    with caplog.at_level(logging.INFO, logger="fastaiagent.agent.memory_blocks"):
        assert block.render("while down") == []
        plane.drop = False  # and back
        out = block.render("plane is back")
    assert out and "Refunds need a ticket id" in out[0].content
    assert any("reachable again" in r.getMessage() for r in caplog.records)


def test_one_outage_pauses_every_block_for_that_agent(plane, silent_plane, caplog):
    """Each user of ``Memory(plane_agent_id=)`` has their own block. Each kept
    its own back-off, so every user waited out the timeout and warned."""
    import time

    alice_block = PlaneFactBlock("shared-agent", timeout=0.3)
    bob_block = PlaneFactBlock("shared-agent", timeout=0.3)
    other_agent = PlaneFactBlock("other-agent", timeout=0.3)
    with caplog.at_level(logging.WARNING, logger="fastaiagent.agent.memory_blocks"):
        started = time.monotonic()
        alice_block.render("alice asks")  # finds the plane down: one wait
        alice_wait = time.monotonic() - started
        started = time.monotonic()
        bob_block.render("bob asks")  # same plane and agent: already paused
        bob_wait = time.monotonic() - started
        started = time.monotonic()
        other_agent.render("another agent")  # its own agent: its own probe
        other_wait = time.monotonic() - started
    assert alice_wait >= 0.25 and bob_wait < 0.2 and other_wait >= 0.25
    warnings = [r for r in caplog.records if "PlaneFactBlock" in r.getMessage()]
    assert len(warnings) == 2  # one per outage per agent, not one per user


# --- 1.82.0: a per-user Memory can read plane facts ----------------------------


def test_a_per_user_memory_includes_plane_facts(plane):
    """Plane facts needed a hand-built ComposableMemory, which has one window for
    every user. ``Memory(plane_agent_id=)`` adds them and keeps windows per user."""
    from dataclasses import dataclass

    from fastaiagent import Agent, Memory, RunContext
    from fastaiagent.learn import MemoryStore
    from fastaiagent.testing import TestModel

    @dataclass
    class St:
        user_id: str

    import tempfile

    store = MemoryStore(db_path=tempfile.mkdtemp() + "/facts.db")
    mem = Memory(location=store, user_id=lambda ctx: ctx.state.user_id, plane_agent_id="support")
    model = TestModel(response="noted")
    agent = Agent(name="support", llm=model, memory=mem)

    def prompt() -> str:
        return " ".join(m.content or "" for m in model.calls[-1]["messages"])

    agent.run("alice-private-message", context=RunContext(state=St("alice")))
    assert "Refunds need a ticket id" in prompt()
    agent.run("bob here", context=RunContext(state=St("bob")))
    assert "Refunds need a ticket id" in prompt() and "alice-private-message" not in prompt()
    agent.run("no user")  # an unresolved caller still gets the agent's plane facts
    assert "Refunds need a ticket id" in prompt()
    assert plane.requests[0]["agent_id"] == ["support"]
