"""Shared test fixtures for the FastAIAgent SDK test suite."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from fastaiagent.llm.client import LLMClient, LLMResponse
from fastaiagent.llm.message import ToolCall


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data."""
    return tmp_path


class CaptureServer:
    """A real localhost HTTP server that records requests and returns
    programmable status codes.

    Used to exercise :class:`fastaiagent.trace.platform_export.PlatformSpanExporter`
    against a real socket with real ``httpx`` — **no mocking**. Status codes are
    served from a queue (one per request); when the queue drains it returns 200,
    so a default server is a happy-path platform. ``set_status_sequence([...])``
    scripts transient failures (e.g. ``[500, 500, 200]`` to drive a retry).
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self._status_codes: list[int] = []
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def set_status_sequence(self, codes: list[int]) -> None:
        with self._lock:
            self._status_codes = list(codes)

    def _next_status(self) -> int:
        with self._lock:
            return self._status_codes.pop(0) if self._status_codes else 200

    @property
    def ingest_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"].endswith("/traces/ingest")]

    @property
    def url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> CaptureServer:
        capture = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence stderr noise
                pass

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except Exception:
                    body = None
                with capture._lock:
                    capture.requests.append(
                        {"path": self.path, "headers": dict(self.headers), "body": body}
                    )
                code = capture._next_status()
                payload = (
                    b'{"ingested": 0}' if 200 <= code < 300 else b'{"error": "scripted failure"}'
                )
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            # BaseHTTPRequestHandler dispatches by exact method name — these
            # must stay do_POST/do_GET (N815 mixedCase is unavoidable here).
            do_POST = _handle  # noqa: N815
            do_GET = _handle  # noqa: N815 — also answers /auth/check if a test connects

        self._server = HTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


@pytest.fixture
def capture_server() -> Any:
    """A started :class:`CaptureServer`, stopped on teardown."""
    server = CaptureServer().start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def isolated_local_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point all local storage at a fresh per-test SQLite file + a fixed project.

    Sets ``FASTAIAGENT_LOCAL_DB`` to a temp file, clears the cached config, and
    pins ``project_id`` so seeded rows and the exporter's drain filter agree.
    """
    from fastaiagent._internal import instance as _instance
    from fastaiagent._internal import project as _project
    from fastaiagent._internal.config import reset_config

    db_path = tmp_path / "local.db"
    monkeypatch.setenv("FASTAIAGENT_LOCAL_DB", str(db_path))
    reset_config()
    _project.set_project_id("test-proj")
    _instance.reset_for_testing()  # drop any cached instance_id from a prior temp DB
    try:
        yield db_path
    finally:
        _instance.reset_for_testing()
        _project.reset_for_testing()
        reset_config()


class MockLLMClient(LLMClient):
    """A mock LLM client that returns predefined responses."""

    def __init__(self, responses: list[LLMResponse] | None = None):
        super().__init__(provider="mock", model="mock-model")
        self._responses = responses or [
            LLMResponse(content="Hello! How can I help?", finish_reason="stop")
        ]
        self._call_count = 0
        self._calls: list[dict] = []

    async def acomplete(self, messages, tools=None, **kwargs):
        self._calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = self._responses[-1]
        self._call_count += 1
        return response

    async def astream(self, messages, tools=None, **kwargs):
        """Yield stream events from the next canned response.

        Not a real token stream — we emit the full text as a single
        ``TextDelta`` plus any tool calls as paired ``ToolCallStart``/
        ``ToolCallEnd``. Good enough to exercise
        :class:`fastaiagent.agent.Swarm.astream`, middleware on the stream
        path, and anything else that consumes stream events.
        """
        from fastaiagent.llm.stream import (
            StreamDone,
            TextDelta,
            ToolCallEnd,
            ToolCallStart,
            Usage,
        )

        self._calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        if self._call_count < len(self._responses):
            response = self._responses[self._call_count]
        else:
            response = self._responses[-1]
        self._call_count += 1

        if response.content:
            yield TextDelta(text=response.content)
        for tc in response.tool_calls:
            yield ToolCallStart(call_id=tc.id, tool_name=tc.name)
            yield ToolCallEnd(call_id=tc.id, tool_name=tc.name, arguments=dict(tc.arguments))
        usage = response.usage or {}
        yield Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )
        yield StreamDone()


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """A mock LLM that returns a simple text response."""
    return MockLLMClient()


@pytest.fixture
def mock_llm_with_tools() -> MockLLMClient:
    """A mock LLM that makes one tool call then returns a final answer."""
    return MockLLMClient(
        responses=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "test"})],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            LLMResponse(
                content="Based on the search results, here is the answer.",
                finish_reason="stop",
                usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            ),
        ]
    )


@pytest.fixture
def recording_middleware():
    """Factory for a middleware that records every hook invocation.

    Returns a ``(middleware, records)`` tuple. ``records`` is a dict with
    keys ``before_model``, ``after_model``, ``wrap_tool`` each mapping to
    a list of capture dicts. Tests assert on these to verify ordering and
    hook semantics.
    """
    from fastaiagent.agent.middleware import AgentMiddleware

    def _factory(name: str = "rec"):
        records: dict = {"before_model": [], "after_model": [], "wrap_tool": []}

        class _Recording(AgentMiddleware):
            def __init__(self) -> None:
                self.name = name

            async def before_model(self, ctx, messages):
                records["before_model"].append(
                    {
                        "name": self.name,
                        "turn": ctx.turn,
                        "agent_name": ctx.agent_name,
                        "message_count": len(messages),
                    }
                )
                return messages

            async def after_model(self, ctx, response):
                records["after_model"].append(
                    {
                        "name": self.name,
                        "turn": ctx.turn,
                        "content": response.content,
                    }
                )
                return response

            async def wrap_tool(self, ctx, tool, args, call_next):
                records["wrap_tool"].append(
                    {
                        "name": self.name,
                        "phase": "enter",
                        "tool": tool.name,
                        "tool_call_index": ctx.tool_call_index,
                    }
                )
                result = await call_next(tool, args)
                records["wrap_tool"].append(
                    {
                        "name": self.name,
                        "phase": "exit",
                        "tool": tool.name,
                    }
                )
                return result

        return _Recording(), records

    return _factory


@pytest.fixture
def noop_middleware():
    """A canonical no-op middleware — byte-for-byte identity on every hook."""
    from fastaiagent.agent.middleware import AgentMiddleware

    class _NoOp(AgentMiddleware):
        name = "noop"

    return _NoOp()
