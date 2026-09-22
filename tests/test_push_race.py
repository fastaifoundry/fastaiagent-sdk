"""Concurrent ``push_agent`` calls for one name must produce exactly one POST.

No mocking: a real ``http.server`` on a real socket, the real ``PlatformAPI``,
the real ``push_agent``. The only thing standing in for the plane is a server
that counts requests and answers like the plane does.

Why this exists. ``_pushed`` was written only *after* the POST returned, with
``_lock`` released across the network call, so two concurrent ``run()`` calls
both passed the check and both POSTed. The plane found the duplicate rows that
produced — inserted 0.5 ms, 2.7 ms and 14.6 ms apart by
``examples/96_connected_eval_export.py`` at ``concurrency=2`` — and reported
that once two rows existed, *every* later push of that name failed permanently
on their side. They have since made the insert idempotent, so this is no longer
a correctness bug, but the SDK was still spending two requests where one does
and its dedupe cache was not doing its job.

The server sleeps before replying. Without that the window is real but narrow,
and a test that only sometimes reproduces a race is worse than no test.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import fastaiagent as fa
from fastaiagent._platform import push as push_mod
from fastaiagent.testing import TestModel


class _Recorder(ThreadingHTTPServer):
    """An HTTP server that counts the agent pushes it received.

    Threading, not the plain ``HTTPServer``: a single-threaded server handles
    one request at a time, which serializes concurrent pushes in the *harness*
    and would make the parallelism assertion below measure the test rather than
    the code. It failed exactly that way first.
    """

    posts: list[dict]
    lock: threading.Lock
    delay: float
    status: int


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep pytest output clean
        pass

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        server: _Recorder = self.server  # type: ignore[assignment]

        # Only agent pushes count. ``connect()`` also POSTs an SDK-instance
        # handshake to this same server, and counting it made every assertion
        # here off by one — and off by a *timing-dependent* one, since the
        # handshake is best-effort.
        if not self.path.endswith("/sdk/agents"):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return

        # Hold the connection open so the race window is wide and deterministic.
        time.sleep(server.delay)

        with server.lock:
            server.posts.append(body)
            # The plane assigns one id per name; a duplicate push adopts it.
            seq = len({p.get("name") for p in server.posts})

        payload = json.dumps(
            {"id": f"00000000-0000-0000-0000-{seq:012d}", "name": body.get("name"), "version": 1}
        ).encode()
        self.send_response(server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def plane() -> Iterator[_Recorder]:
    """A real HTTP server standing in for the plane's agent-push endpoint."""
    server: _Recorder = _Recorder(("127.0.0.1", 0), _Handler)  # type: ignore[assignment]
    server.posts = []
    server.lock = threading.Lock()
    server.delay = 0.25
    server.status = 200
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def connected(plane: _Recorder) -> Iterator[_Recorder]:
    host, port = plane.server_address[0], plane.server_address[1]
    push_mod.reset_registration_state()
    fa.connect(api_key="fa_k_test", target=f"http://{host}:{port}", auto_register=False)
    try:
        yield plane
    finally:
        fa.disconnect()
        push_mod.reset_registration_state()


def _agent(name: str) -> fa.Agent:
    return fa.Agent(name=name, system_prompt="s", llm=TestModel())


def _push_concurrently(agent: fa.Agent, n: int) -> list[object]:
    results: list[object] = [None] * n
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        barrier.wait()  # release all threads into push_agent together
        results[i] = push_mod.push_agent(agent, best_effort=True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


def test_four_concurrent_pushes_of_one_agent_send_one_request(connected: _Recorder) -> None:
    """The defect, stated as a test: it was 4 POSTs, it must be 1."""
    agent = _agent("race-agent")

    _push_concurrently(agent, 4)

    assert len(connected.posts) == 1, (
        f"expected one POST for one agent, got {len(connected.posts)} — "
        "the claim is being taken after the response again"
    )
    assert connected.posts[0]["name"] == "race-agent"


def test_every_caller_still_gets_a_result(connected: _Recorder) -> None:
    """The loser must not be handed ``None`` where it used to get a result.

    An explicit ``agent.push()`` colliding with a background auto-register is
    the real case: skipping the request is right, returning nothing is not.
    """
    agent = _agent("race-result")

    results = _push_concurrently(agent, 4)

    assert all(r is not None for r in results), "a concurrent caller got None"
    ids = {r.agent_id for r in results}  # type: ignore[union-attr]
    assert len(ids) == 1, f"callers disagreed about the agent id: {ids}"
    assert sum(1 for r in results if not r.skipped) == 1  # type: ignore[union-attr]


def test_different_agents_are_not_serialized(connected: _Recorder) -> None:
    """Two *different* names must overlap, not queue behind each other.

    Holding ``_lock`` across the POST would also have fixed the duplicate, at
    the cost of serializing every unrelated push behind the slowest request.
    """
    agents = [_agent(f"parallel-{i}") for i in range(4)]
    barrier = threading.Barrier(len(agents))

    def worker(a: fa.Agent) -> None:
        barrier.wait()
        push_mod.push_agent(a, best_effort=True)

    threads = [threading.Thread(target=worker, args=(a,)) for a in agents]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.monotonic() - started

    assert len(connected.posts) == 4, "each distinct agent needs its own POST"
    serial = connected.delay * len(agents)
    assert elapsed < serial * 0.8, (
        f"{len(agents)} distinct pushes took {elapsed:.2f}s; serial would be "
        f"~{serial:.2f}s — unrelated agents are queueing behind one another"
    )


def test_a_failed_push_does_not_wedge_the_name(connected: _Recorder) -> None:
    """A claim must be released on failure, or the name is dead for the process."""
    connected.status = 500
    agent = _agent("wedge-check")

    assert push_mod.push_agent(agent, best_effort=True) is None
    assert "wedge-check" not in push_mod._inflight, "the claim outlived a failed push"

    connected.status = 200
    result = push_mod.push_agent(agent, best_effort=True)

    assert result is not None, "the name stayed claimed after a failure"
    assert result.agent_id


def test_a_second_push_after_success_costs_no_request(connected: _Recorder) -> None:
    """The cache still does its original job: the 2nd..Nth push is free."""
    agent = _agent("cached-agent")

    first = push_mod.push_agent(agent, best_effort=True)
    second = push_mod.push_agent(agent, best_effort=True)

    assert len(connected.posts) == 1
    assert first is not None and second is not None
    assert second.skipped is True
    assert second.agent_id == first.agent_id


def test_force_re_pushes_even_after_a_cached_success(connected: _Recorder) -> None:
    """``force=True`` is the escape hatch for a same-process definition change."""
    agent = _agent("forced-agent")

    push_mod.push_agent(agent, best_effort=True)
    forced = push_mod.push_agent(agent, force=True, best_effort=True)

    assert len(connected.posts) == 2, "force must reach the plane"
    assert forced is not None and forced.skipped is False


def test_reset_wakes_a_waiting_caller(connected: _Recorder) -> None:
    """Clearing state mid-flight must not strand a thread on its timeout."""
    agent = _agent("reset-agent")
    connected.delay = 1.0

    winner = threading.Thread(
        target=push_mod.push_agent, args=(agent,), kwargs={"best_effort": True}
    )
    winner.start()
    time.sleep(0.2)  # let the winner claim the name and start its POST

    push_mod.reset_registration_state()
    assert push_mod._inflight == {}, "reset left a claim behind"

    winner.join(timeout=30)
    assert not winner.is_alive()
