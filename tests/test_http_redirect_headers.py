"""security_audit_2 N11 — credential headers don't follow a cross-origin redirect.

``asafe_http_request`` (used by RESTTool) drops ``Authorization`` / ``Cookie`` /
API-key headers when a redirect changes the origin, so a cooperating first hop
can't bounce a bearer token to another host. Same-origin redirects keep them.
No network: an ``httpx.MockTransport`` serves the redirect chain and records the
headers each hop actually received.
"""

from __future__ import annotations

import httpx
import pytest

from fastaiagent.multimodal import _http


def _install_mock(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient

    def fake(**kw):
        kw.pop("verify", None)  # MockTransport supplies its own transport
        return real(transport=transport, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    # We're testing header handling, not SSRF; skip real DNS/IP validation.
    monkeypatch.setattr(_http, "validate_url", lambda url, **k: None)


@pytest.mark.asyncio
async def test_auth_dropped_on_cross_origin_redirect(monkeypatch):
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), {k.lower(): v for k, v in request.headers.items()}))
        if request.url.host == "api.example.com":
            return httpx.Response(302, headers={"location": "https://evil.example/steal"})
        return httpx.Response(200, content=b"ok")

    _install_mock(monkeypatch, handler)

    resp = await _http.asafe_http_request(
        "https://api.example.com/data",
        method="GET",
        timeout=5,
        max_redirects=3,
        max_bytes=10_000,
        headers={"Authorization": "Bearer SECRET", "Content-Type": "application/json"},
    )
    assert resp.status_code == 200

    first = next(h for u, h in seen if "api.example.com" in u)
    second = next(h for u, h in seen if "evil.example" in u)
    assert first.get("authorization") == "Bearer SECRET"  # kept on first hop
    assert "authorization" not in second  # dropped crossing origin
    assert second.get("content-type") == "application/json"  # non-sensitive kept


@pytest.mark.asyncio
async def test_auth_kept_on_same_origin_redirect(monkeypatch):
    seen: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), {k.lower(): v for k, v in request.headers.items()}))
        if request.url.path == "/data":
            return httpx.Response(302, headers={"location": "https://api.example.com/data2"})
        return httpx.Response(200, content=b"ok")

    _install_mock(monkeypatch, handler)

    await _http.asafe_http_request(
        "https://api.example.com/data",
        method="GET",
        timeout=5,
        max_redirects=3,
        max_bytes=10_000,
        headers={"Authorization": "Bearer SECRET"},
    )
    # Both hops are same-origin → auth retained throughout.
    assert all(h.get("authorization") == "Bearer SECRET" for _, h in seen)
