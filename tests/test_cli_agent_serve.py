"""Security-surface tests for ``fastaiagent agent serve`` (security_audit_2 N1).

These exercise only the paths that do not require a live LLM — the auth
dependency, the body-size cap, and the open ``/health`` probe — so no network
or model calls are mocked. The happy-path ``/run`` execution is covered
elsewhere by the deployment examples.
"""

from __future__ import annotations

import contextlib
import io

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.agent.agent import Agent  # noqa: E402
from fastaiagent.cli import agent as agent_cli  # noqa: E402


def _agent() -> Agent:
    return Agent(name="t", system_prompt="x")


def test_health_is_open_without_auth() -> None:
    client = TestClient(agent_cli._build_app(_agent()))
    assert client.get("/health").status_code == 200


def test_health_stays_open_when_auth_enabled() -> None:
    client = TestClient(agent_cli._build_app(_agent(), auth_token="s3cret"))
    # Liveness probes must not need the token.
    assert client.get("/health").status_code == 200


def test_run_requires_token_when_configured() -> None:
    # raise_server_exceptions=False so the correct-token path can proceed to
    # execution without an LLM key (it 500s on a missing key rather than raising)
    # — we only care that auth was cleared, not that the agent ran. Keeps the
    # test key-free so it passes in CI (no live provider credentials).
    client = TestClient(
        agent_cli._build_app(_agent(), auth_token="s3cret"),
        raise_server_exceptions=False,
    )
    assert client.post("/run", json={"input": "x"}).status_code == 401
    assert (
        client.post(
            "/run", json={"input": "x"}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    # Correct token clears auth — the request is no longer a 401 (it proceeds to
    # execution, which may 500 without a provider key; that's fine here).
    assert (
        client.post(
            "/run", json={"input": "x"}, headers={"Authorization": "Bearer s3cret"}
        ).status_code
        != 401
    )


def test_stream_requires_token_when_configured() -> None:
    client = TestClient(agent_cli._build_app(_agent(), auth_token="s3cret"))
    assert client.post("/run/stream", json={"input": "x"}).status_code == 401


def test_body_size_cap_rejects_oversized_request() -> None:
    client = TestClient(agent_cli._build_app(_agent(), max_body_bytes=50))
    resp = client.post("/run", json={"input": "z" * 500})
    assert resp.status_code == 413


def test_warn_on_exposure_loud_without_auth() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        agent_cli._warn_on_exposure("0.0.0.0", None)
    assert "NO authentication" in buf.getvalue()


def test_warn_on_exposure_silent_on_loopback() -> None:
    for host in ("127.0.0.1", "localhost", "::1"):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            agent_cli._warn_on_exposure(host, None)
        assert buf.getvalue() == ""


def test_run_reports_a_paused_run_as_paused(isolated_local_db, tmp_path) -> None:
    """A policy-gated call pauses the run (the calling application approves,
    1.74.0). ``/run`` used to answer 200 with an empty ``output`` and nothing
    else — indistinguishable from an agent that simply said nothing."""
    import fastaiagent
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
    from tests._governance_plane import AGENT_ID, reset_connection, serve
    from tests.test_governance_approvals import TRANSFER, _bank, _Script

    ran, tool = _bank()
    agent = Agent(
        name="banker",
        agent_id=AGENT_ID,
        llm=_Script(TRANSFER),
        tools=[tool],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    with serve() as (_, url):
        fastaiagent.connect(api_key="fa_k_serve_test", target=url)
        try:
            body = (
                TestClient(agent_cli._build_app(agent))
                .post("/run", json={"input": "Transfer $500 to Bob."})
                .json()
            )
        finally:
            reset_connection()

    assert body["status"] == "paused", body
    assert body["output"] == ""
    assert body["execution_id"]
    assert body["pending_interrupt"]["reason"] == "policy_approval_required"
    assert body["pending_interrupt"]["context"]["tool_input"] == {"amount": 500, "to": "Bob"}
    assert ran == []


def test_run_stream_sends_a_paused_event(isolated_local_db, tmp_path) -> None:
    """A paused run ends ``/run/stream`` with a ``paused`` event (1.77.0, audit M4) —
    before, the private exception escaped ``astream`` and broke the response."""
    import json

    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
    from tests.test_stream_pause import _llm, _sign_off_tool

    agent = Agent(
        name="clerk",
        llm=_llm("ask_human", {"question": "ok?"}),
        tools=[_sign_off_tool()],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    body = (
        TestClient(agent_cli._build_app(agent))
        .post("/run/stream", json={"input": "Get sign-off."})
        .text
    )

    events = [
        json.loads(line[len("data: ") :]) for line in body.splitlines() if line.startswith("data: ")
    ]
    types = [e["type"] for e in events]
    assert types[-2:] == ["paused", "done"], types
    paused = events[-2]
    assert paused["reason"] == "need_sign_off"
    assert paused["execution_id"]
    assert paused["context"] == {"question": "ok?"}


# --- POST /resume (1.92.0): a run paused through the service is resumed through it


def _clerk(tmp_path) -> Agent:
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
    from tests.test_stream_pause import _llm, _sign_off_tool

    return Agent(
        name="clerk",
        llm=_llm("ask_human", {"question": "ok?"}),
        tools=[_sign_off_tool()],
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )


def test_resume_completes_a_paused_agent_run(isolated_local_db, tmp_path) -> None:
    client = TestClient(agent_cli._build_app(_clerk(tmp_path)))
    paused = client.post("/run", json={"input": "Get sign-off."}).json()
    assert paused["status"] == "paused"

    r = client.post("/resume", json={"execution_id": paused["execution_id"], "approved": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["output"] == "done"
    assert body["execution_id"] == paused["execution_id"]

    again = client.post("/resume", json={"execution_id": paused["execution_id"], "approved": True})
    assert again.status_code == 409


def test_resume_completes_a_paused_chain_run(isolated_local_db, tmp_path) -> None:
    from fastaiagent import Chain
    from fastaiagent.checkpointers.sqlite import SQLiteCheckpointer
    from tests.test_stream_pause import _sign_off_tool

    chain = Chain(
        "sign-off",
        checkpoint_enabled=True,
        checkpointer=SQLiteCheckpointer(str(tmp_path / "ckpt.db")),
    )
    chain.add_node(
        "ask",
        type="tool",
        tool=_sign_off_tool(),
        input_mapping={"question": "{{input.input}}"},
    )
    client = TestClient(agent_cli._build_app(chain))
    paused = client.post("/run", json={"input": "ok?"}).json()
    assert paused["status"] == "paused"

    body = client.post(
        "/resume", json={"execution_id": paused["execution_id"], "approved": False}
    ).json()
    assert body["status"] == "completed"
    assert "declined" in body["output"]

    unknown = client.post("/resume", json={"execution_id": "nope", "approved": True})
    assert unknown.status_code == 404


def test_resume_requires_token_when_configured(tmp_path) -> None:
    client = TestClient(agent_cli._build_app(_clerk(tmp_path), auth_token="s3cret"))
    r = client.post("/resume", json={"execution_id": "x", "approved": True})
    assert r.status_code == 401


def test_openapi_json_is_served() -> None:
    """1.93.0: ``/openapi.json`` answered 500 — FastAPI could not resolve the
    ``StreamingResponse`` return annotation — so ``/docs`` showed nothing."""
    client = TestClient(agent_cli._build_app(_agent()))
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert {"/run", "/run/stream", "/resume", "/cancel"} <= set(resp.json()["paths"])


def test_cancel_of_a_run_not_in_progress_is_404() -> None:
    client = TestClient(agent_cli._build_app(_agent(), auth_token="s3cret"))
    assert client.post("/cancel", json={"execution_id": "nope"}).status_code == 401
    resp = client.post(
        "/cancel", json={"execution_id": "nope"}, headers={"Authorization": "Bearer s3cret"}
    )
    assert resp.status_code == 404


class _Server:
    """The serve app on a real uvicorn, so a request can be in flight or dropped."""

    def __init__(self, app: object) -> None:
        import socket
        import threading

        import uvicorn

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> str:
        import time

        self._thread.start()
        while not self._server.started:
            time.sleep(0.05)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def _slow_agent(finished: list[str], started: object = None) -> Agent:
    import asyncio

    from fastaiagent import FunctionTool
    from fastaiagent.testing import TestModel

    async def aslow() -> str:
        """Take a while."""
        if started is not None:
            started.set()  # type: ignore[attr-defined]
        await asyncio.sleep(3)
        finished.append("yes")
        return "late"

    return Agent(
        name="slow",
        llm=TestModel(response=["", "done"], tool_calls=[{"name": "aslow", "arguments": {}}]),
        tools=[FunctionTool(name="aslow", fn=aslow)],
    )


def test_cancel_stops_a_run_in_progress(isolated_local_db) -> None:
    import threading
    import time

    import httpx

    finished: list[str] = []
    in_tool = threading.Event()
    with _Server(agent_cli._build_app(_slow_agent(finished, in_tool))) as url:
        answer: dict[str, object] = {}

        def call_run() -> None:
            body = {"input": "go", "execution_id": "srv-1"}
            answer.update(httpx.post(f"{url}/run", json=body, timeout=10).json())

        caller = threading.Thread(target=call_run)
        caller.start()
        assert in_tool.wait(10), "the run never reached its tool"
        stopped = time.monotonic()
        resp = httpx.post(f"{url}/cancel", json={"execution_id": "srv-1"}, timeout=5)
        caller.join(timeout=5)
    assert resp.status_code == 200 and resp.json()["cancelled"] is True
    assert answer["status"] == "cancelled" and answer["execution_id"] == "srv-1"
    assert time.monotonic() - stopped < 2 and finished == []


def test_a_client_that_hangs_up_stops_the_run(isolated_local_db) -> None:
    """1.93.0: an abandoned ``/run`` kept calling the model and tools to the end."""
    import time

    import httpx

    from fastaiagent.agent import run_control

    finished: list[str] = []
    with _Server(agent_cli._build_app(_slow_agent(finished))) as url:
        with pytest.raises(httpx.ReadTimeout):
            httpx.post(f"{url}/run", json={"input": "go", "execution_id": "srv-2"}, timeout=0.4)
        time.sleep(1.5)  # the server notices within its 0.5 s poll
        assert run_control.cancel("srv-2") is False, "the run is still going"
        time.sleep(2)
    assert finished == []


def test_serve_has_no_reload_flag() -> None:
    """``--reload`` never worked: uvicorn reloads only from an import string,
    and serve passes it an app object. Removed in 1.92.0."""
    from typer.testing import CliRunner

    result = CliRunner().invoke(agent_cli.agent_app, ["serve", "x.py:a", "--reload"])
    assert result.exit_code == 2
    assert "No such option" in result.output
