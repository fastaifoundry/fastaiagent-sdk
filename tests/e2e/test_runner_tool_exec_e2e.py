"""E2E (Task B) — the runner executes a dispatched ``tool_exec`` against a LOCAL
connector that uses a real credential (``OPENAI_API_KEY`` from ~/.zshrc).

No mocks. The real daemon registers + long-polls a real localhost plane, claims a
``tool_exec`` command, resolves the operator's locally-registered tool by its
``exposed_name``, and runs it — the tool makes a real ``gpt-4o-mini`` call with the
operator's own key. The runner reports ``{"success","result"}`` and pushes the
``tool.<name>`` trace. We assert the result is recorded and the trace is ingested +
linked. Run via ``zsh -lc`` so ~/.zshrc keys reach the process.
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

_API_KEY = "fa_k_toolexec_e2e"


class _PlaneState:
    def __init__(self, commands: list[dict]) -> None:
        self.lock = threading.Lock()
        self.runner_id = "runner-1"
        self.runner_token = "tok-secret-xyz"
        self.queue: list[dict] = list(commands)
        self.results: dict[str, dict] = {}
        self.ingested_spans: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a: Any, **k: Any) -> None:
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
            with st.lock:
                st.ingested_spans.extend(body.get("spans", []))
            self._json(201, {"ingested": len(body.get("spans", []))})
        else:
            self._json(404, {"detail": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        st: _PlaneState = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/auth/check":
            self._json(
                200, {"ok": True, "domain_id": "dom-1", "project_id": "proj-1", "scopes": []}
            )
        elif self.path.endswith("/commands"):
            with st.lock:
                cmds = st.queue[:]
                st.queue.clear()
            if not cmds:
                time.sleep(0.4)
            self._json(200, {"commands": cmds})
        else:
            self._json(404, {"detail": "not found"})


@pytest.fixture
def _clean_platform():
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


def test_runner_executes_tool_exec_against_local_connector(
    isolated_local_db, _clean_platform
) -> None:
    os.environ.setdefault("E2E_SKIP_PLATFORM", "1")  # local stand-in plane
    require_env()  # OPENAI_API_KEY — the local connector makes a real LLM call

    import fastaiagent
    from fastaiagent.runner.channel import RunnerChannel
    from fastaiagent.runner.daemon import RunnerDaemon
    from fastaiagent.tool.function import FunctionTool
    from fastaiagent.trace import otel

    # The operator's LOCAL connector tool — runs in the runner's boundary with the
    # operator's OWN key (never shipped by the plane). Async so it nests cleanly.
    async def ask_llm(question: str) -> str:
        from fastaiagent import Agent, LLMClient

        agent = Agent(
            name="connector",
            system_prompt="Reply with exactly the word OK and nothing else.",
            llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        )
        return (await agent.arun(question)).output

    FunctionTool(name="ask_llm", fn=ask_llm)  # auto-registers in ToolRegistry

    cmd = {
        "command_id": "te-e2e-1",
        "type": "tool_exec",
        "tenant": "dom-1",
        "deadline": None,
        "payload": {
            "tool_exec": {
                "tool_type": "connector",
                "connector": {"instance_id": "inst-1", "action": "run", "fixed_params": {}},
                "exposed_name": "ask_llm",
                "arguments": {"question": "Say the word OK."},
            },
            "hosted_server_id": "srv-1",
        },
    }

    state = _PlaneState([cmd])
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://{server.server_address[0]}:{server.server_address[1]}"

    otel.reset()
    fastaiagent.connect(api_key=_API_KEY, target=base_url)
    channel = RunnerChannel(base_url=base_url, api_key=_API_KEY)
    daemon = RunnerDaemon(
        channel, max_concurrency=1, capabilities=("live_playground", "eval_run", "tool_exec")
    )

    async def _drive() -> None:
        task = asyncio.create_task(daemon.run())
        for _ in range(600):  # up to ~60s for the real LLM call + push
            with state.lock:
                done = "te-e2e-1" in state.results
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

    assert "te-e2e-1" in state.results, state.results
    res = state.results["te-e2e-1"]
    assert res["status"] == "completed", res
    assert res["result"]["success"] is True, res
    assert res["result"]["result"], res  # the connector's (LLM's) answer
    trace_id = res.get("trace_id")
    assert trace_id, res

    # The tool call was traced and pushed, linked by the reported trace_id.
    pushed = {s["trace_id"] for s in state.ingested_spans}
    assert trace_id in pushed, (trace_id, pushed)
    names = {s["name"] for s in state.ingested_spans}
    assert "tool.ask_llm" in names, names
