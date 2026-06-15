"""E2E (Task A) — the registered runner pushes its job traces to the platform.

No mocks. A **real** ``gpt-4o-mini`` agent executes the dispatched commands via the
real runner daemon + ``execute_command``; the **real** ``PlatformSpanExporter`` —
wired by the real ``connect()`` + ``BatchSpanProcessor`` exactly as the shipped
``fastaiagent runner`` CLI now does — drains to a **real** localhost server that
implements the frozen ``/public/v1`` runner-channel + ``/traces/ingest`` endpoints.

We assert the executed job's trace is ingested and linked by the ``trace_id`` the
runner reports back, for both ``live_playground`` and ``eval_run``. This is the
SDK-side proof of the trace-push fix (RC1: the exporter is wired; RC2: the
background-thread drain finds the job's spans). The wire shape is frozen and the
server side is covered by the Enterprise contract tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e

_API_KEY = "fa_k_runner_e2e"


class _PlaneState:
    """Records what the runner sends to the stand-in plane."""

    def __init__(self, commands: list[dict]) -> None:
        self.lock = threading.Lock()
        self.runner_id = "runner-1"
        self.runner_token = "tok-secret-xyz"
        self.queue: list[dict] = list(commands)
        self.results: dict[str, dict] = {}
        self.ingested_spans: list[dict] = []  # every span POSTed to /traces/ingest
        self.ingest_api_keys: list[str | None] = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any, **k: Any) -> None:  # silence
        pass

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self) -> None:  # noqa: N802
        st: _PlaneState = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/runners":
            self._json(201, {"runner_id": st.runner_id, "runner_token": st.runner_token})
        elif self.path.endswith("/heartbeat"):
            self._json(200, {"ok": True, "ttl_seconds": 6})
        elif self.path.endswith("/results"):
            body = self._body()
            with st.lock:
                st.results[body["command_id"]] = body
            self._json(202, {"accepted": True})
        elif self.path == "/public/v1/traces/ingest":
            body = self._body()
            spans = body.get("spans", [])
            with st.lock:
                st.ingest_api_keys.append(self.headers.get("X-API-Key"))
                st.ingested_spans.extend(spans)
            self._json(201, {"ingested": len(spans)})
        else:
            self._json(404, {"detail": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        st: _PlaneState = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/auth/check":
            self._json(
                200,
                {
                    "ok": True, "domain_id": "dom-1",
                    "project_id": "proj-1", "scopes": ["runner:register"],
                },
            )
        elif self.path.endswith("/commands"):
            with st.lock:
                cmds = st.queue[:]
                st.queue.clear()
            if not cmds:
                time.sleep(0.4)  # long-poll hold so the daemon doesn't busy-loop
            self._json(200, {"commands": cmds})
        else:
            self._json(404, {"detail": "not found"})


@pytest.fixture
def _clean_platform():
    """Reset the platform connection + tracer provider around each test so the
    real BatchSpanProcessor/exporter never leak across tests."""
    import fastaiagent
    from fastaiagent.client import _connection
    from fastaiagent.trace import otel

    yield
    try:
        fastaiagent.disconnect()
    except Exception:
        pass
    otel.reset()
    _connection.api_key = None
    _connection.target = "https://app.fastaiagent.net"
    _connection.project = None
    _connection.project_id = None
    _connection.domain_id = None
    _connection._platform_processor = None


def _run_one_command(cmd: dict, command_id: str) -> _PlaneState:
    """Drive the real daemon against a stand-in plane until ``command_id`` lands.

    Wires the real exporter via ``connect()`` (the CLI's RC1 fix), then runs one
    real-LLM command through the daemon and returns the recorded plane state.
    """
    state = _PlaneState([cmd])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"

    import fastaiagent
    from fastaiagent.client import _connection
    from fastaiagent.runner.channel import RunnerChannel
    from fastaiagent.runner.daemon import RunnerDaemon
    from fastaiagent.trace import otel

    # Fresh tracer provider, then connect() so the real PlatformSpanExporter is
    # registered against it — the exact wiring the runner CLI now performs.
    otel.reset()
    fastaiagent.connect(api_key=_API_KEY, target=base_url)
    assert _connection.is_connected

    channel = RunnerChannel(base_url=base_url, api_key=_API_KEY)
    daemon = RunnerDaemon(channel, max_concurrency=1)

    async def _drive() -> None:
        task = asyncio.create_task(daemon.run())
        for _ in range(600):  # up to ~60s for real LLM call(s) + push
            with state.lock:
                done = command_id in state.results
            if done:
                break
            await asyncio.sleep(0.1)
        daemon.request_stop()
        await asyncio.wait_for(task, timeout=25)

    try:
        asyncio.run(_drive())
    finally:
        server.shutdown()
        server.server_close()

    return state


def _make_agent_config() -> dict:
    from fastaiagent import Agent, LLMClient

    agent = Agent(
        name="runner-echo",
        system_prompt="Reply with exactly the word OK and nothing else.",
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
    )
    return agent.to_dict()


def test_runner_pushes_live_playground_trace(isolated_local_db, _clean_platform) -> None:
    os.environ.setdefault("E2E_SKIP_PLATFORM", "1")  # local stand-in plane, not a remote one
    require_env()  # OPENAI_API_KEY — runs a real gpt-4o-mini agent

    cmd = {
        "command_id": "cmd-lp-1",
        "type": "live_playground",
        "tenant": "dom-1",
        "deadline": None,
        "payload": {"agent": _make_agent_config(), "input": "say ok"},
    }
    state = _run_one_command(cmd, "cmd-lp-1")

    assert "cmd-lp-1" in state.results, state.results
    res = state.results["cmd-lp-1"]
    assert res["status"] == "completed", res
    trace_id = res.get("trace_id")
    assert trace_id, f"runner must report a trace_id: {res}"

    # RC1: the trace reached /traces/ingest; linked by the reported trace_id.
    pushed = {s["trace_id"] for s in state.ingested_spans}
    assert trace_id in pushed, f"reported trace_id {trace_id} not among ingested {pushed}"
    # Routed by the runner's key (the plane resolves the project from it).
    assert _API_KEY in state.ingest_api_keys, state.ingest_api_keys

    # RC2: the bg-thread drain matched the job's spans — none left buffered.
    from fastaiagent.trace.storage import TraceStore

    store = TraceStore()
    assert store.count_unsynced("test-proj") == 0
    store.close()


def test_runner_pushes_eval_run_traces(isolated_local_db, _clean_platform) -> None:
    os.environ.setdefault("E2E_SKIP_PLATFORM", "1")
    require_env()

    cmd = {
        "command_id": "cmd-eval-1",
        "type": "eval_run",
        "tenant": "dom-1",
        "deadline": None,
        "payload": {
            "agent": _make_agent_config(),
            "suite_id": "suite-1",
            "cases": [
                {
                    "case_id": "c1", "input": "say ok",
                    "expected_output": "OK", "criteria": {"type": "contains", "value": "OK"},
                },
                {
                    "case_id": "c2", "input": "say ok again",
                    "expected_output": "OK", "criteria": {"type": "contains", "value": "OK"},
                },
            ],
        },
    }
    state = _run_one_command(cmd, "cmd-eval-1")

    assert "cmd-eval-1" in state.results, state.results
    res = state.results["cmd-eval-1"]
    assert res["status"] == "completed", res

    # eval_run result shape (frozen): {"outputs": [{case_id, output, trace_id}, ...]} in order.
    result = res.get("result")
    assert isinstance(result, dict) and "outputs" in result, result
    outputs = result["outputs"]
    assert [o["case_id"] for o in outputs] == ["c1", "c2"], outputs
    assert all(o["output"] for o in outputs), outputs  # each case produced real output

    # Each case emitted its own trace, and all were pushed to the plane.
    case_trace_ids = {o["trace_id"] for o in outputs}
    assert len(case_trace_ids) == 2, outputs
    pushed = {s["trace_id"] for s in state.ingested_spans}
    assert case_trace_ids <= pushed, f"case traces {case_trace_ids} not all ingested {pushed}"

    from fastaiagent.trace.storage import TraceStore

    store = TraceStore()
    assert store.count_unsynced("test-proj") == 0
    store.close()
