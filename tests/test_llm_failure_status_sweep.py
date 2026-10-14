"""A failed LLM call leaves an ERROR span, streamed or not (P5, 1.89.0).

A streamed call holds its ``llm.*`` span by hand (an async generator cannot keep
a span *current* across ``yield``), and on failure it only called
``record_exception``, and only for ``LLMProviderError``. So a failed stream
ended with status UNSET, while the same failure through ``complete()`` was
ERROR (OTel's ``start_as_current_span`` sets it). The Playground's ``/stream``
caught the failure inside its root span and returned normally, so its root
stayed UNSET too. A trace whose only failure was a streamed call never showed
as failed, locally or on the plane (which derives ``trace.status`` from the
root).

The sweep: every call path × every failure kind, against a real localhost
server playing the provider (``CaptureServer`` with scripted responses), with
real tracing and storage. Plus the negative axes: success, a consumer that
stops early and a cancelled stream are **not** errors, and with tracing off
the original exception type still propagates.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from fastaiagent.llm import LLMClient, UserMessage
from fastaiagent.llm.stream import TextDelta

# ---------------------------------------------------------------------------
# Fixtures: a traced private DB, and a provider that fails on demand
# ---------------------------------------------------------------------------


@pytest.fixture
def traced_db(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config
    from fastaiagent.trace import otel

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    for name in ("FASTAIAGENT_API_KEY", "FASTAIAGENT_TARGET", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    reset_config()
    _project.set_project_id("test-proj")
    otel.reset()
    try:
        yield db_path
    finally:
        otel.reset()
        _project.reset_for_testing()
        reset_config()


def _dead_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


def _sse(*payloads: Any) -> str:
    return "".join(f"data: {p if isinstance(p, str) else json.dumps(p)}\n\n" for p in payloads)


_OK_SSE = _sse(
    {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
    {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
    {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    },
    "[DONE]",
)

_OK_JSON = {
    "id": "c1",
    "object": "chat.completion",
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


def _fail(kind: str, server: Any, *, stream: bool) -> tuple[str, int]:
    """Script ``server`` to fail as ``kind``; return ``(base_url, max_retries)``."""
    err = {"error": {"message": "scripted failure", "type": "server_error"}}
    if kind == "http_400":
        server.set_response_sequence([(400, err)] * 4)
        return server.url, 0
    if kind == "http_500_after_retry":
        server.set_response_sequence([(500, err)] * 4)
        return server.url, 1
    if kind == "connection_refused":
        return _dead_url(), 0
    if kind == "malformed_body":
        body = (
            (200, "data: {this is not json\n\n", "text/event-stream")
            if stream
            else (200, "this is not json", "application/json")
        )
        server.set_response_sequence([body] * 4)
        return server.url, 0
    raise AssertionError(kind)


KINDS = ["http_400", "http_500_after_retry", "connection_refused", "malformed_body"]


def _spans(db_path: Any) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT name, parent_span_id, status, events FROM spans ORDER BY start_time"
        ).fetchall()
    return [{**dict(r), "events": json.loads(r["events"] or "[]")} for r in rows]


def _assert_failed_trace(db_path: Any, where: str) -> None:
    spans = _spans(db_path)
    assert spans, f"{where}: nothing was traced, so nothing was checked"
    llm = [s for s in spans if s["name"].startswith("llm.")]
    for s in llm:
        assert s["status"] == "ERROR", f"{where}: {s['name']} ended {s['status']}"
        types = [
            e["attributes"].get("exception.type")
            for e in s["events"]
            if e.get("name") == "exception"
        ]
        assert types and types[-1], f"{where}: {s['name']} has no exception event"
    roots = [s for s in spans if s["parent_span_id"] is None]
    assert roots, f"{where}: no root span"
    for root in roots:
        assert root["status"] == "ERROR", f"{where}: root {root['name']} ended {root['status']}"

    from fastaiagent.trace.storage import TraceStore

    summaries = TraceStore.default().list_traces(last_hours=None)
    assert summaries and summaries[0].status == "ERROR", f"{where}: the trace list shows it OK"


# ---------------------------------------------------------------------------
# The call paths
# ---------------------------------------------------------------------------


def _client(url: str, retries: int) -> LLMClient:
    return LLMClient(
        provider="openai", model="m", base_url=url, api_key="stub", max_retries=retries
    )


async def _drain(gen: Any) -> None:
    async for _ in gen:
        pass


def _agent(url: str, retries: int) -> Any:
    import fastaiagent as fa

    return fa.Agent(name="p5", system_prompt="s", llm=_client(url, retries))


_HI = [UserMessage("hi")]

PATHS: dict[str, tuple[bool, Callable[[str, int], Any]]] = {
    # name: (streams?, call)
    "client.acomplete": (False, lambda u, r: asyncio.run(_client(u, r).acomplete(_HI))),
    "client.complete": (False, lambda u, r: _client(u, r).complete([UserMessage("hi")])),
    "client.astream": (True, lambda u, r: asyncio.run(_drain(_client(u, r).astream(_HI)))),
    "client.stream": (True, lambda u, r: list(_client(u, r).stream([UserMessage("hi")]))),
    "agent.run": (False, lambda u, r: _agent(u, r).run("hi")),
    "agent.arun": (False, lambda u, r: asyncio.run(_agent(u, r).arun("hi"))),
    "agent.stream": (True, lambda u, r: _agent(u, r).stream("hi")),
    "agent.astream": (True, lambda u, r: asyncio.run(_drain(_agent(u, r).astream("hi")))),
    "agent.stream[trace=False]": (True, lambda u, r: _agent(u, r).stream("hi", trace=False)),
    "agent.astream[trace=False]": (
        True,
        lambda u, r: asyncio.run(_drain(_agent(u, r).astream("hi", trace=False))),
    ),
    "swarm.stream": (True, lambda u, r: _swarm(u, r).stream("hi")),
    "supervisor.stream": (True, lambda u, r: _supervisor(u, r).stream("hi")),
    "supervisor.astream": (True, lambda u, r: asyncio.run(_drain(_supervisor(u, r).astream("hi")))),
}


def _swarm(url: str, retries: int) -> Any:
    from fastaiagent.agent.swarm import Swarm

    a = _agent(url, retries)
    return Swarm(name="sw", agents=[a], entrypoint=a.name)


def _supervisor(url: str, retries: int) -> Any:
    import fastaiagent as fa
    from fastaiagent.agent.team import Supervisor, Worker
    from fastaiagent.testing import TestModel

    worker = fa.Agent(name="w", system_prompt="s", llm=TestModel(response="ok"))
    return Supervisor(name="sup", llm=_client(url, retries), workers=[Worker(agent=worker)])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("path", list(PATHS))
def test_a_failed_call_leaves_an_error_trace(
    traced_db: Any, capture_server: Any, path: str, kind: str
) -> None:
    streams, call = PATHS[path]
    url, retries = _fail(kind, capture_server, stream=streams)

    with pytest.raises(Exception):  # noqa: B017 — each path raises its own type
        call(url, retries)

    _assert_failed_trace(traced_db, f"{path}/{kind}")


@pytest.mark.parametrize("kind", KINDS)
def test_streamed_and_non_streamed_failures_record_the_same_thing(
    traced_db: Any, capture_server: Any, kind: str
) -> None:
    def outcome(stream: bool) -> tuple[str, str]:
        url, retries = _fail(kind, capture_server, stream=stream)
        with pytest.raises(Exception):  # noqa: B017
            if stream:
                asyncio.run(_drain(_client(url, retries).astream([UserMessage("hi")])))
            else:
                asyncio.run(_client(url, retries).acomplete([UserMessage("hi")]))
        span = [s for s in _spans(traced_db) if s["name"].startswith("llm.")][-1]
        exc = [e for e in span["events"] if e.get("name") == "exception"]
        return span["status"], exc[-1]["attributes"].get("exception.type") if exc else ""

    assert outcome(stream=False) == outcome(stream=True)


# ---------------------------------------------------------------------------
# The Playground routes
# ---------------------------------------------------------------------------


@pytest.fixture
def playground(traced_db: Any) -> Any:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from fastaiagent.ui.db import init_local_db
    from fastaiagent.ui.server import build_app

    init_local_db(traced_db).close()
    return TestClient(build_app(db_path=str(traced_db), no_auth=True))


def _pg_body(url: str) -> dict[str, Any]:
    return {
        "provider": "custom",
        "model": "m",
        "prompt_template": "hi",
        "variables": {},
        "base_url": url,
        "api_key": "stub",
    }


@pytest.mark.parametrize("kind", KINDS)
def test_playground_run_failure_is_an_error_trace(
    traced_db: Any, playground: Any, capture_server: Any, kind: str
) -> None:
    url, _ = _fail(kind, capture_server, stream=False)
    r = playground.post("/api/playground/run", json=_pg_body(url))
    assert r.status_code == 502, r.text
    _assert_failed_trace(traced_db, f"playground/run/{kind}")


@pytest.mark.parametrize("kind", KINDS)
def test_playground_stream_failure_is_an_error_trace(
    traced_db: Any, playground: Any, capture_server: Any, kind: str
) -> None:
    url, _ = _fail(kind, capture_server, stream=True)
    r = playground.post("/api/playground/stream", json=_pg_body(url))
    assert "event: error" in r.text, r.text
    _assert_failed_trace(traced_db, f"playground/stream/{kind}")
    root = [s for s in _spans(traced_db) if s["parent_span_id"] is None][0]
    messages = [e["attributes"].get("exception.message", "") for e in root["events"]]
    assert not any("scripted failure" in m for m in messages), (
        "the provider's own text reached the root span"
    )


# ---------------------------------------------------------------------------
# What is NOT an error
# ---------------------------------------------------------------------------


def _llm_statuses(db_path: Any) -> list[str]:
    return [s["status"] for s in _spans(db_path) if s["name"].startswith("llm.")]


def test_a_successful_stream_is_not_an_error(traced_db: Any, capture_server: Any) -> None:
    capture_server.set_response_sequence([(200, _OK_SSE, "text/event-stream")])
    asyncio.run(_drain(_client(capture_server.url, 0).astream([UserMessage("hi")])))
    assert _llm_statuses(traced_db) and "ERROR" not in _llm_statuses(traced_db)


def test_a_consumer_that_stops_early_is_not_an_error(traced_db: Any, capture_server: Any) -> None:
    capture_server.set_response_sequence([(200, _OK_SSE, "text/event-stream")])

    async def first_delta_then_stop() -> None:
        gen = _client(capture_server.url, 0).astream([UserMessage("hi")])
        async with contextlib.aclosing(gen):
            async for ev in gen:
                if isinstance(ev, TextDelta):
                    break

    asyncio.run(first_delta_then_stop())
    statuses = _llm_statuses(traced_db)
    assert statuses, "the abandoned stream's span never ended"
    assert "ERROR" not in statuses


def test_a_cancelled_stream_is_not_an_error(traced_db: Any, capture_server: Any) -> None:
    capture_server.set_response_sequence([(200, _OK_SSE, "text/event-stream")])

    async def cancel_after_first_delta() -> None:
        gen = _client(capture_server.url, 0).astream([UserMessage("hi")])
        async for ev in gen:
            if isinstance(ev, TextDelta):
                with contextlib.suppress(asyncio.CancelledError):
                    await gen.athrow(asyncio.CancelledError())
                break
        await gen.aclose()

    asyncio.run(cancel_after_first_delta())
    statuses = _llm_statuses(traced_db)
    assert statuses and "ERROR" not in statuses


def test_with_tracing_off_the_original_exception_propagates(
    tmp_path: Any, capture_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent._internal.config import reset_config
    from fastaiagent._internal.errors import LLMProviderError
    from fastaiagent.trace import otel

    db = tmp_path / "off.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db))
    monkeypatch.setenv("FASTAIAGENT_TRACE_ENABLED", "0")
    reset_config()
    otel.reset()
    try:
        url, _ = _fail("http_400", capture_server, stream=True)
        with pytest.raises(LLMProviderError):
            asyncio.run(_drain(_client(url, 0).astream([UserMessage("hi")])))
        assert _spans(db) == []
    finally:
        otel.reset()
        reset_config()
