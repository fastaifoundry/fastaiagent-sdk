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
