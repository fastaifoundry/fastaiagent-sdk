"""A transport failure is a transport failure, however ``httpx`` spells it (1.91.0).

Found by CI on 2026-10-21. ``httpx`` maps ``httpcore``'s network errors to its own
(``httpx.ConnectError``) through a map of ``httpcore`` classes it builds once.
When ``httpcore`` is imported again in the same process, the classes it raises
are new objects that map does not hold, and a refused connection escapes as raw
``httpcore.ConnectError``. The SDK recognised only ``httpx.TransportError``, so
``max_retries`` and ``fallbacks=`` silently never applied, and the live fallback
test failed in the e2e gate.

Re-importing ``httpcore`` breaks ``httpx`` for the rest of the process, so the
end-to-end cases run in a subprocess. The fallback model there is a real
``tests.conftest.CaptureServer`` in this process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import httpx

from fastaiagent._internal.http_errors import is_transport_error

REPLY = {
    "id": "c",
    "object": "chat.completion",
    "model": "gpt-5-mini",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}

#: Re-import httpcore the way CI's run ended up with it, then call a dead port.
_SCRIPT = textwrap.dedent(
    """
    import asyncio, importlib, socket, sys
    import httpx

    def dead_url():
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        return f"http://127.0.0.1:{port}/v1"

    async def raw_kind():
        async with httpx.AsyncClient() as c:
            try:
                await c.post(dead_url() + "/x")
            except BaseException as e:
                return f"{type(e).__module__}.{type(e).__name__}"

    # httpx builds its httpcore -> httpx map on its first error; then the re-import.
    print("BEFORE", asyncio.run(raw_kind()))
    for name in [m for m in sys.modules if m == "httpcore" or m.startswith("httpcore.")]:
        del sys.modules[name]
    importlib.import_module("httpcore")
    print("RAW", asyncio.run(raw_kind()))

    from fastaiagent import LLMClient
    from fastaiagent.llm.message import UserMessage
    from fastaiagent.llm.stream import TextDelta

    capture = sys.argv[1]
    client = LLMClient(
        provider="openai", model="gpt-5-mini", api_key="k", base_url=dead_url(),
        fallbacks=[LLMClient(provider="openai", model="gpt-5-mini", api_key="k", base_url=capture)],
    )
    print("FALLBACK", client.complete([UserMessage("ping")]).content)

    async def streamed():
        text = ""
        async for event in client.astream([UserMessage("ping")]):
            if isinstance(event, TextDelta):
                text += event.text
        return text

    print("STREAM_FALLBACK", asyncio.run(streamed()))
    """
)


def test_is_transport_error_by_name() -> None:
    assert is_transport_error(httpx.ConnectError("x"))
    assert is_transport_error(TimeoutError())
    assert not is_transport_error(ValueError("x"))
    assert not is_transport_error(httpx.HTTPStatusError("x", request=None, response=None))  # type: ignore[arg-type]

    # A class from a second import of httpcore: same names, a different object.
    network = type("NetworkError", (Exception,), {"__module__": "httpcore._exceptions"})
    stale = type("ConnectError", (network,), {"__module__": "httpcore._exceptions"})
    assert is_transport_error(stale("All connection attempts failed"))
    other = type("ConnectError", (Exception,), {"__module__": "somelib"})
    assert not is_transport_error(other("x"))


def test_retries_and_fallbacks_survive_a_reimported_httpcore(capture_server) -> None:
    sse = (
        "".join(
            f"data: {json.dumps(chunk)}\n\n"
            for chunk in (
                {"choices": [{"index": 0, "delta": {"content": "PONG"}}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            )
        )
        + "data: [DONE]\n\n"
    )
    capture_server.set_response_sequence([(200, REPLY), (200, sse, "text/event-stream")])
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, capture_server.url + "/v1"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout
    assert proc.returncode == 0, proc.stderr[-2000:]
    # The condition is real: after the re-import httpx hands back raw httpcore.
    assert "RAW httpcore.ConnectError" in out, out
    assert "FALLBACK PONG" in out, out
    assert "STREAM_FALLBACK PONG" in out, out


_EXPORT_SCRIPT = textwrap.dedent(
    """
    import importlib, json, socket, sys
    from datetime import datetime, timezone
    import httpx

    def raw_kind():
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        try:
            httpx.post(f"http://127.0.0.1:{port}/x")
        except BaseException as e:
            return f"{type(e).__module__}.{type(e).__name__}"

    raw_kind()  # httpx builds its map on its first error; then the re-import
    for name in [m for m in sys.modules if m == "httpcore" or m.startswith("httpcore.")]:
        del sys.modules[name]
    importlib.import_module("httpcore")
    print("RAW", raw_kind())

    from fastaiagent._internal import project
    from fastaiagent.client import _connection
    from fastaiagent.trace.platform_export import PlatformSpanExporter
    from fastaiagent.trace.storage import TraceStore

    project.set_project_id("test-proj")
    store = TraceStore()
    now = datetime.now(timezone.utc).isoformat()
    store._db.execute(
        "INSERT INTO spans (span_id, trace_id, parent_span_id, name, start_time, end_time, "
        "status, attributes, events, project_id, synced) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("s1", "t1", None, "span-s1", now, now, "OK", json.dumps({"k": "v"}), "[]", "test-proj", 0),
    )
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    _connection.api_key = "fa_k_test"
    _connection.target = f"http://127.0.0.1:{port}"
    _connection.project = "test-project"
    _connection.project_id = "platform-uuid-1"
    PlatformSpanExporter._BACKOFF_BASE = 0.0
    result = PlatformSpanExporter().export([])
    print("EXPORT", result.name, "unsynced", store.count_unsynced("test-proj"))
    """
)


def test_the_span_exporter_buffers_through_a_reimported_httpcore(tmp_path) -> None:
    """The exporters catch the same failure the same way: a refused plane after the
    re-import leaves the span buffered and the run unaffected, instead of a raw
    httpcore error escaping the export loop."""
    import os

    env = {**os.environ, "FASTAIAGENT_LOCAL_DB": str(tmp_path / "local.db")}
    env.pop("FASTAIAGENT_API_KEY", None)
    env.pop("FASTAIAGENT_TARGET", None)
    proc = subprocess.run(
        [sys.executable, "-c", _EXPORT_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "RAW httpcore.ConnectError" in proc.stdout, proc.stdout  # the condition is real
    assert "EXPORT SUCCESS unsynced 1" in proc.stdout, proc.stdout + proc.stderr[-1500:]
