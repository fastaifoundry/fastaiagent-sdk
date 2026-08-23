"""security_audit_2 N16 — Host allowlist (anti DNS-rebinding) + cross-origin
Origin rejection for state-changing requests.

Real FastAPI + SQLite, no mocks. The autouse conftest fixture adds
``testserver`` (TestClient's default Host) to the allowlist; these tests send
explicit Host/Origin headers to exercise the reject paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from fastaiagent.ui.db import init_local_db  # noqa: E402
from fastaiagent.ui.server import build_app  # noqa: E402


@pytest.fixture
def client(temp_dir: Path) -> TestClient:
    db = temp_dir / "local.db"
    init_local_db(db).close()
    return TestClient(build_app(db_path=str(db), no_auth=True))


class TestHostValidation:
    def test_allowed_host_passes(self, client: TestClient) -> None:
        assert client.get("/api/auth/status").status_code == 200  # Host: testserver

    @pytest.mark.parametrize("bad_host", ["evil.attacker.com", "example.com", "rebind.test"])
    def test_rebound_host_rejected(self, client: TestClient, bad_host: str) -> None:
        r = client.get("/api/auth/status", headers={"host": bad_host})
        assert r.status_code == 400
        assert "not allowed" in r.json()["detail"].lower()

    def test_loopback_host_allowed(self, client: TestClient) -> None:
        for host in ("localhost", "127.0.0.1", "127.0.0.1:7842", "[::1]:7842"):
            assert client.get("/api/auth/status", headers={"host": host}).status_code == 200

    def test_env_allowlist_extends(self, temp_dir: Path, monkeypatch) -> None:
        monkeypatch.setenv("FASTAIAGENT_UI_ALLOWED_HOSTS", "testserver,ui.internal.corp")
        db = temp_dir / "local.db"
        init_local_db(db).close()
        c = TestClient(build_app(db_path=str(db), no_auth=True))
        assert c.get("/api/auth/status", headers={"host": "ui.internal.corp"}).status_code == 200


class TestCrossOriginRejection:
    def test_cross_origin_post_rejected(self, client: TestClient) -> None:
        r = client.post(
            "/api/traces/bulk-delete",
            json={"trace_ids": ["x"]},
            headers={"origin": "http://evil.attacker.com"},
        )
        assert r.status_code == 403
        assert "cross-origin" in r.json()["detail"].lower()

    def test_same_origin_post_allowed(self, client: TestClient) -> None:
        # Same-origin Origin (an allowed host) passes the Origin gate.
        r = client.post(
            "/api/traces/bulk-delete",
            json={"trace_ids": ["x"]},
            headers={"origin": "http://localhost:7842"},
        )
        assert r.status_code == 200  # no such trace, but the request is accepted

    def test_no_origin_header_allowed(self, client: TestClient) -> None:
        # curl / scripts / TestClient send no Origin → unaffected.
        r = client.post("/api/traces/bulk-delete", json={"trace_ids": ["x"]})
        assert r.status_code == 200

    def test_safe_method_never_blocked_by_origin(self, client: TestClient) -> None:
        r = client.get("/api/auth/status", headers={"origin": "http://evil.attacker.com"})
        assert r.status_code == 200
