"""End-to-end gate — the registered-runner daemon (task 2.6).

No Python mocks. A **real** local HTTP server implements the four §7.5 runner-
channel endpoints, and the daemon runs **real `gpt-4.1` agents** for the
`live_playground` jobs. We drive the full loop and assert:

* register uses ``X-API-Key``; every later call uses ``Authorization: Bearer``;
* queued commands execute and report ``completed`` with real model output;
* concurrency is bounded by ``--max-concurrency`` (3 jobs, limit 2 -> peak 2);
* graceful shutdown sends a final ``status="stopping"`` heartbeat.

The real Enterprise channel server is task 2.7 (not built yet), so this local
stand-in is the maximal live verification available now.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e


class _ChannelState:
    def __init__(self, commands: list[dict]) -> None:
        self.lock = threading.Lock()
        self.runner_id = "runner-1"
        self.runner_token = "tok-secret-xyz"
        self.register_api_key: str | None = None
        self.auth_headers: list[str | None] = []  # Authorization on every post-register call
        self.heartbeats: list[dict] = []
        self.results: dict[str, dict] = {}
        self.queue: list[dict] = list(commands)


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
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self) -> None:  # noqa: N802
        st: _ChannelState = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/runners":
            with st.lock:
                st.register_api_key = self.headers.get("X-API-Key")
            self._json(201, {"runner_id": st.runner_id, "runner_token": st.runner_token})
        elif self.path.endswith("/heartbeat"):
            body = self._body()
            with st.lock:
                st.auth_headers.append(self.headers.get("Authorization"))
                st.heartbeats.append(
                    {"status": body.get("status"), "active_jobs": body.get("active_jobs")}
                )
            self._json(200, {"ok": True, "ttl_seconds": 6})
        elif self.path.endswith("/results"):
            body = self._body()
            with st.lock:
                st.auth_headers.append(self.headers.get("Authorization"))
                st.results[body["command_id"]] = body
            self._json(202, {"accepted": True})
        else:
            self._json(404, {"detail": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        st: _ChannelState = self.server.state  # type: ignore[attr-defined]
        if self.path.endswith("/commands"):
            with st.lock:
                st.auth_headers.append(self.headers.get("Authorization"))
                cmds = st.queue[:]
                st.queue.clear()
            if not cmds:
                time.sleep(0.4)  # long-poll hold so the daemon doesn't busy-loop
            self._json(200, {"commands": cmds})
        else:
            self._json(404, {"detail": "not found"})


def test_runner_daemon_full_loop_live(tmp_path: Any) -> None:
    import os

    os.environ.setdefault("E2E_SKIP_PLATFORM", "1")
    require_env()  # OPENAI_API_KEY — the runner executes a real gpt-4.1 agent

    from fastaiagent import Agent, LLMClient
    from fastaiagent.runner.channel import RunnerChannel
    from fastaiagent.runner.daemon import RunnerDaemon
    from fastaiagent.runner.execute import execute_command

    # A real agent config carried in the command payload (config only — no keys).
    agent = Agent(
        name="runner-echo",
        system_prompt="Reply with exactly the word OK and nothing else.",
        llm=LLMClient(provider="openai", model="gpt-4.1"),
    )
    agent_cfg = agent.to_dict()
    commands = [
        {
            "command_id": f"cmd-{i}",
            "type": "live_playground",
            "tenant": "tenant-A",
            "deadline": None,
            "payload": {"agent": agent_cfg, "input": "say ok"},
        }
        for i in range(3)
    ]

    state = _ChannelState(commands)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"

    # Instrument the REAL executor to observe peak concurrency (not a stub — it
    # runs the real agent inside).
    peak = {"cur": 0, "max": 0}

    async def instrumented(cmd: dict):
        peak["cur"] += 1
        peak["max"] = max(peak["max"], peak["cur"])
        try:
            return await execute_command(cmd)
        finally:
            peak["cur"] -= 1

    channel = RunnerChannel(base_url=base_url, api_key="test-key")
    daemon = RunnerDaemon(channel, max_concurrency=2, executor=instrumented)

    async def _drive() -> None:
        task = asyncio.create_task(daemon.run())
        for _ in range(400):  # up to ~40s for 3 real LLM calls
            with state.lock:
                done = len(state.results)
            if done >= 3:
                break
            await asyncio.sleep(0.1)
        daemon.request_stop()
        await asyncio.wait_for(task, timeout=25)

    try:
        asyncio.run(_drive())
    finally:
        server.shutdown()
        server.server_close()

    # --- assertions ---
    assert state.register_api_key == "test-key", "register must use the X-API-Key"
    post_register = [h for h in state.auth_headers if h is not None]
    assert post_register, "expected authenticated post-register calls"
    assert all(h == "Bearer tok-secret-xyz" for h in post_register), (
        f"all post-register calls must use the runner_token Bearer: {set(post_register)}"
    )

    assert len(state.results) == 3, state.results
    for cid in ("cmd-0", "cmd-1", "cmd-2"):
        r = state.results[cid]
        assert r["status"] == "completed", r
        assert "ok" in str(r.get("result", "")).lower(), r

    assert 1 <= peak["max"] <= 2, f"concurrency must be bounded by 2, peaked at {peak['max']}"

    statuses = [h["status"] for h in state.heartbeats]
    assert "stopping" in statuses, f"graceful shutdown must send a stopping heartbeat: {statuses}"
