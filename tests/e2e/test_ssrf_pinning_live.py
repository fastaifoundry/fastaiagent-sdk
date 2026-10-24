"""Live: the SSRF guard's pinned connections (1.92.0) against real public
sites, on every path that makes guarded requests.

Each request now connects to the address the guard checked, not to whatever the
name resolves to next. That must not weaken TLS: the certificate is still
verified for the hostname (a wrong-host or self-signed certificate is refused),
redirects still work, and a real MCP server over HTTPS still answers. The
rebinding attack itself is proven offline in ``tests/test_ssrf_pinning.py``.
"""

from __future__ import annotations

import httpx
import pytest

from fastaiagent.multimodal._http import asafe_http_request, safe_http_fetch

pytestmark = pytest.mark.e2e


async def test_https_through_the_async_helper() -> None:
    resp = await asafe_http_request(
        "https://example.com/", timeout=20, max_redirects=0, max_bytes=1_000_000
    )
    assert resp.status_code == 200
    assert "Example Domain" in resp.text
    assert str(resp.url) == "https://example.com/"


def test_https_redirect_through_the_sync_helper() -> None:
    resp = safe_http_fetch("http://github.com/", timeout=20, max_redirects=3, max_bytes=10_000_000)
    assert resp.status_code == 200
    assert str(resp.url).startswith("https://github.com")


@pytest.mark.parametrize(
    "url", ["https://wrong.host.badssl.com/", "https://self-signed.badssl.com/"]
)
async def test_a_bad_certificate_is_still_refused(url: str) -> None:
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED|certificate"):
        await asafe_http_request(url, timeout=20, max_redirects=0, max_bytes=1_000_000)


def test_image_from_url() -> None:
    from fastaiagent.multimodal.image import Image

    image = Image.from_url("https://www.python.org/static/img/python-logo.png")
    assert image.media_type == "image/png"
    assert image.size_bytes() > 1000


async def test_rest_tool_over_https() -> None:
    from fastaiagent.tool.rest import RESTTool

    result = await RESTTool(name="page", url="https://example.com/", method="GET").ainvoke({})
    assert result.success, result.error
    assert "Example Domain" in str(result.output)


async def test_mcp_tool_over_https_to_a_public_server() -> None:
    pytest.importorskip("mcp")
    from fastaiagent.tool.mcp import MCPTool

    tools = await MCPTool.afrom_server("https://mcp.deepwiki.com/mcp")
    assert tools, "the server listed no tools"
    assert all(t.server_url == "https://mcp.deepwiki.com/mcp" for t in tools)
