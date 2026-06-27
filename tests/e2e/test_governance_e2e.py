"""E2E (Task C) — a connected agent honors a managed approval policy.

No mocks. A **real** ``gpt-4o-mini`` agent runs against a **real** localhost plane
that implements the frozen governance endpoints (``/policy``, ``/policy/decide``,
``/runs/{id}/pending``). A high-stakes tool (``transfer_funds``) matches the cached
approval policy → ``/policy/decide`` returns ``require_approval`` → the SDK posts a
pending run and **pauses** (a real checkpoint). We then "approve" (flip the plane's
pending status, the way the console does via the Public API) and the agent
**resumes** — tested both non-blocking (``wait_for_approval=False`` + ``aresume``)
and blocking (``arun`` auto-waits + resumes).

Mirrors the live :20001 verification; the wire shapes are frozen and the plane
side is covered by the Enterprise contract tests.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tests.e2e.conftest import require_env

pytestmark = pytest.mark.e2e

_API_KEY = "fa_k_gov_e2e"
_AGENT_ID = "agent-banker-1"


class _GovPlane:
    """Records governance calls and serves a programmable pending-run status."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.decide_calls: list[dict] = []
        self.pending_posts: list[dict] = []
        self.pending_polls: int = 0
        # Per-run status the SDK polls; flip to "approved" to simulate the console.
        self.status_by_run: dict[str, str] = {}
        self.auto_approve = False  # if True, a posted pending is immediately approved


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

    def _run_id(self) -> str:
        # /public/v1/runs/{run_id}/pending
        return self.path.split("/runs/", 1)[1].rsplit("/pending", 1)[0]

    def do_GET(self) -> None:  # noqa: N802
        st: _GovPlane = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/auth/check":
            self._json(200, {
                "ok": True, "domain_id": "dom-1", "project_id": "proj-1",
                "scopes": ["policy:read", "policy:decide", "run:write", "run:read"],
            })
        elif self.path == "/public/v1/policy":
            self._json(200, {
                "version": "gov-e2e-v1",
                "guardrail_rules": [],
                "approval_policies": [{
                    "id": "ap-1", "name": "transfer-approval", "agent_id": None,
                    "tool_pattern": "transfer_funds", "condition_type": "always",
                    "condition_config": None, "timeout_minutes": 60,
                }],
            })
        elif self.path.endswith("/pending"):
            run_id = self._run_id()
            with st.lock:
                st.pending_polls += 1
                status = st.status_by_run.get(run_id, "pending")
            self._json(200, {"pending_id": f"pr-{run_id}", "status": status})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        st: _GovPlane = self.server.state  # type: ignore[attr-defined]
        if self.path == "/public/v1/policy/decide":
            body = self._body()
            with st.lock:
                st.decide_calls.append(body)
            if body.get("tool_name") == "transfer_funds":
                self._json(200, {"decision": "require_approval", "approval_request_id": "apr-1",
                                 "reason": "Matched approval policy 'transfer-approval'"})
            else:
                self._json(200, {"decision": "allow", "approval_request_id": None, "reason": None})
        elif self.path.endswith("/pending"):
            run_id = self._run_id()
            with st.lock:
                st.pending_posts.append({"run_id": run_id, "body": self._body()})
                st.status_by_run[run_id] = "approved" if st.auto_approve else "pending"
            self._json(201, {"pending_id": f"pr-{run_id}", "status": "pending"})
        elif self.path == "/public/v1/traces/ingest":
            self._json(201, {"ingested": len(self._body().get("spans", []))})
        else:
            self._json(404, {"detail": "not found"})


@pytest.fixture
def gov_plane():
    state = _GovPlane()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.state = state  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    try:
        yield state, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


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
    for attr, val in (("api_key", None), ("target", "https://app.fastaiagent.net"),
                      ("project", None), ("project_id", None), ("domain_id", None),
                      ("policy_cache", None), ("governance_fail_mode", "open"),
                      ("_platform_processor", None)):
        setattr(_connection, attr, val)


def _make_agent(tmp_path):
    from fastaiagent import Agent, FunctionTool, LLMClient
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer

    def transfer_funds(amount: int, to: str) -> str:
        return f"Transferred ${amount} to {to}."

    return Agent(
        name="banker",
        agent_id=_AGENT_ID,
        system_prompt=(
            "You are a banking assistant. To move money, call transfer_funds(amount, to). "
            "After the tool returns, confirm to the user in one sentence."
        ),
        llm=LLMClient(provider="openai", model="gpt-4o-mini"),
        tools=[FunctionTool(name="transfer_funds", fn=transfer_funds)],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )


def test_governance_pause_approve_resume_nonblocking(gov_plane, _clean_platform, tmp_path) -> None:
    require_env()  # OPENAI_API_KEY — real gpt-4o-mini
    state, url = gov_plane

    import fastaiagent
    from fastaiagent.chain.interrupt import Resume
    from fastaiagent.client import _connection

    fastaiagent.connect(api_key=_API_KEY, target=url)
    assert (_connection.policy_cache or {}).get("approval_policies"), "policy should be cached"

    agent = _make_agent(tmp_path)
    res = asyncio.run(
        agent.arun("Transfer $500 to Bob.", wait_for_approval=False, execution_id="run-nb")
    )

    # The high-stakes tool paused for approval (decide → require_approval → pending).
    assert res.status == "paused", res
    assert res.pending_interrupt.get("reason") == "policy_approval_required"
    assert state.decide_calls and state.decide_calls[0]["tool_name"] == "transfer_funds"
    assert state.decide_calls[0]["agent_id"] == _AGENT_ID  # the platform agent id, not the name
    assert state.pending_posts and state.pending_posts[0]["body"]["kind"] == "approval"

    # The console approves (flip the pending status), then the agent resumes.
    state.status_by_run["run-nb"] = "approved"
    final = asyncio.run(agent.aresume("run-nb", resume_value=Resume(approved=True)))
    assert final.status == "completed", final
    assert "bob" in final.output.lower()


def test_governance_blocking_autoresume(gov_plane, _clean_platform, tmp_path) -> None:
    require_env()
    state, url = gov_plane
    state.auto_approve = True  # console approves immediately

    import fastaiagent

    fastaiagent.connect(api_key=_API_KEY, target=url)
    agent = _make_agent(tmp_path)

    # arun() blocks on approval and auto-resumes (wait_for_approval defaults True).
    final = asyncio.run(agent.arun("Transfer $250 to Alice.", execution_id="run-blk"))

    assert final.status == "completed", final
    assert "alice" in final.output.lower()
    assert state.pending_posts, "a pending run should have been registered"
    assert state.pending_polls >= 1, "the SDK should have polled for the decision"
