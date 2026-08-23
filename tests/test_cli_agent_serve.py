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
    client = TestClient(agent_cli._build_app(_agent(), auth_token="s3cret"))
    assert client.post("/run", json={"input": "x"}).status_code == 401
    assert (
        client.post(
            "/run", json={"input": "x"}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    # Correct token clears auth — it then proceeds to execution (not a 401).
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
