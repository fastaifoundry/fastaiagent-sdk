"""The SSRF guard connects to the address it checked, and caps a body while
reading it (1.92.0). Run against real local servers, on every path that makes
guarded requests: ``asafe_http_request`` (RESTTool), ``safe_http_fetch``
(``Image``/``PDF.from_url``) and ``MCPTool``.

DNS rebinding: before 1.92.0 the guard resolved a hostname, checked the answer,
and httpx then resolved it again to connect. A DNS server that answers "public"
first and "internal" second got the request into the internal service. The one
stand-in here is the resolver (agreed with the maintainer): it plays that DNS
server, answering a public address first and then the address of a real local
server that plays the internal service. That server must see nothing.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest

from fastaiagent._internal.errors import MultimodalError, ToolExecutionError
from fastaiagent.multimodal._http import asafe_http_request, safe_http_fetch
from fastaiagent.tool.rest import RESTTool

REBIND_HOST = "rebind.fastaiagent.test"
#: Treated as public by the guard and not expected to answer on a test port.
PUBLIC_FIRST_ANSWER = "1.2.3.4"


@pytest.fixture
def rebinding_dns(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``REBIND_HOST`` resolves to a public address the first time and to
    127.0.0.1 afterwards. Returns the answers given, in order."""
    real = socket.getaddrinfo
    answers: list[str] = []

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        # anyio (httpx's async connect) passes the name IDNA-encoded, as bytes.
        name = host.decode() if isinstance(host, bytes) else host
        if name != REBIND_HOST:
            return real(host, port, *args, **kwargs)
        addr = PUBLIC_FIRST_ANSWER if not answers else "127.0.0.1"
        answers.append(addr)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, int(port or 0)))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return answers


def test_the_fake_dns_answers_every_lookup_of_the_name(rebinding_dns: list[str]) -> None:
    """Guards the test itself: both the str and bytes forms reach the stand-in,
    so a lookup cannot slip past it to real DNS (this happened while writing it)."""
    socket.getaddrinfo(REBIND_HOST, 80)
    socket.getaddrinfo(REBIND_HOST.encode("idna"), 80)
    assert rebinding_dns == [PUBLIC_FIRST_ANSWER, "127.0.0.1"]


# --- the rebinding attack, on every path ----------------------------------------


async def test_rest_requests_do_not_follow_a_rebound_name(
    capture_server: Any, rebinding_dns: list[str]
) -> None:
    port = capture_server.url.rsplit(":", 1)[1]
    # Connecting to the checked public address fails here (nothing listens);
    # what matters is that the request did not go to the second answer.
    with pytest.raises((httpx.HTTPError, MultimodalError)):
        await asafe_http_request(
            f"http://{REBIND_HOST}:{port}/internal", timeout=1, max_redirects=0, max_bytes=1000
        )
    assert capture_server.requests == []


async def test_rest_tool_does_not_follow_a_rebound_name(
    capture_server: Any, rebinding_dns: list[str]
) -> None:
    port = capture_server.url.rsplit(":", 1)[1]
    tool = RESTTool(name="t", url=f"http://{REBIND_HOST}:{port}/internal", method="GET", timeout=2)
    with pytest.raises(ToolExecutionError):
        await tool.ainvoke({})
    assert capture_server.requests == []


def test_url_fetches_do_not_follow_a_rebound_name(
    capture_server: Any, rebinding_dns: list[str]
) -> None:
    port = capture_server.url.rsplit(":", 1)[1]
    with pytest.raises((httpx.HTTPError, MultimodalError)):
        safe_http_fetch(
            f"http://{REBIND_HOST}:{port}/internal", timeout=1, max_redirects=0, max_bytes=1000
        )
    assert capture_server.requests == []


async def test_mcp_tool_does_not_follow_a_rebound_name(
    capture_server: Any, rebinding_dns: list[str]
) -> None:
    pytest.importorskip("mcp")
    from fastaiagent.tool.mcp import MCPTool

    port = capture_server.url.rsplit(":", 1)[1]
    tool = MCPTool(name="t", server_url=f"http://{REBIND_HOST}:{port}/mcp", timeout=2)
    with pytest.raises(ToolExecutionError):
        await tool.ainvoke({})
    assert capture_server.requests == []


# --- addresses that are not globally routable -----------------------------------

#: Carrier-grade NAT (100.64.0.0/10) — Tailscale gives tailnet devices addresses
#: here, 100.100.100.100 is its own service — and benchmarking space. Before
#: 1.92.0 the guard checked "private" only, and these are not.
NOT_GLOBAL = ["100.64.0.1", "100.100.100.100", "100.127.255.254", "198.18.0.1"]


@pytest.mark.parametrize("host", NOT_GLOBAL)
async def test_addresses_that_are_not_global_are_refused(host: str) -> None:
    with pytest.raises(MultimodalError, match="not a public address"):
        await asafe_http_request(f"http://{host}/", timeout=1, max_redirects=0, max_bytes=1000)
    with pytest.raises(MultimodalError, match="not a public address"):
        safe_http_fetch(f"http://{host}/", timeout=1, max_redirects=0, max_bytes=1000)
    with pytest.raises(ToolExecutionError, match="not a public address"):
        await RESTTool(name="t", url=f"http://{host}/", method="GET").ainvoke({})


@pytest.mark.parametrize("host", NOT_GLOBAL)
async def test_mcp_tool_refuses_addresses_that_are_not_global(host: str) -> None:
    pytest.importorskip("mcp")
    from fastaiagent.tool.mcp import MCPTool

    with pytest.raises(ToolExecutionError, match="not a public address"):
        await MCPTool(name="t", server_url=f"http://{host}/mcp").aexecute({})


def test_global_addresses_still_pass_the_check() -> None:
    from fastaiagent.multimodal._http import validate_url

    for url in ("http://1.1.1.1/", "http://8.8.8.8/", "http://[2606:4700:4700::1111]/"):
        assert validate_url(url) is None  # a literal, allowed IP: nothing to pin


async def test_the_opt_out_still_opens_them(
    capture_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastaiagent.multimodal._http import validate_url

    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    assert validate_url("http://100.100.100.100/") is None


# --- a name still reaches its server, with its own Host header ------------------


async def test_a_hostname_keeps_its_host_header(capture_server: Any) -> None:
    port = capture_server.url.rsplit(":", 1)[1]
    resp = await asafe_http_request(
        f"http://localhost:{port}/x",
        timeout=5,
        max_redirects=0,
        max_bytes=1000,
        allow_loopback=True,
    )
    assert resp.status_code == 200
    assert str(resp.url) == f"http://localhost:{port}/x"
    assert capture_server.requests[0]["headers"]["Host"] == f"localhost:{port}"


class _IPv6Only:
    """A real HTTP server on ::1 only. ``localhost`` resolves to 127.0.0.1 and
    ::1, and IPv4 is tried first, so reaching this server by name proves the
    pinned send moved on to the next checked address. No resolver stand-in."""

    def __init__(self) -> None:
        self.hosts: list[str] = []
        seen = self.hosts

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                seen.append(self.headers.get("Host", ""))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        class Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        self._server = Server(("::1", 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture
def ipv6_only() -> Any:
    addrs = {a[4][0] for a in socket.getaddrinfo("localhost", 80, proto=socket.IPPROTO_TCP)}
    if not {"127.0.0.1", "::1"} <= addrs or not socket.has_ipv6:
        pytest.skip(f"localhost resolves to {sorted(addrs)}, not both 127.0.0.1 and ::1")
    server = _IPv6Only()
    yield server
    server.stop()


async def test_the_next_checked_address_is_tried_when_one_refuses(ipv6_only: Any) -> None:
    resp = await asafe_http_request(
        f"http://localhost:{ipv6_only.port}/x",
        timeout=5,
        max_redirects=0,
        max_bytes=1000,
        allow_loopback=True,
    )
    assert resp.status_code == 200 and resp.text == "ok"
    assert ipv6_only.hosts == [f"localhost:{ipv6_only.port}"]


def test_the_fallback_survives_a_reimported_httpcore(ipv6_only: Any) -> None:
    """After a re-import of httpcore, httpx lets the raw ``httpcore.ConnectError``
    through (X25, 1.91.0). The fallback must still recognise "cannot connect"."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import asyncio, importlib, socket, sys
        import httpx

        # httpx builds its httpcore -> httpx error map on its first error, so
        # fail once (a dead port), then re-import httpcore — as CI's run did.
        s = socket.socket(); s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]; s.close()
        try:
            httpx.get(f"http://127.0.0.1:{{dead}}/")
        except httpx.ConnectError:
            pass
        for name in [m for m in sys.modules if m == "httpcore" or m.startswith("httpcore.")]:
            del sys.modules[name]
        importlib.import_module("httpcore")
        from fastaiagent.multimodal._http import asafe_http_request
        resp = asyncio.run(asafe_http_request(
            "http://localhost:{ipv6_only.port}/x", timeout=5, max_redirects=0,
            max_bytes=1000, allow_loopback=True))
        print("STATUS", resp.status_code)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "STATUS 200" in proc.stdout, proc.stderr[-2000:]


def test_is_connect_error_by_name() -> None:
    from fastaiagent._internal.http_errors import is_connect_error

    assert is_connect_error(httpx.ConnectError("refused"))
    assert is_connect_error(httpx.ConnectTimeout("slow"))
    # A class from a second import of httpcore: same name, a different object.
    stale = type("ConnectError", (Exception,), {"__module__": "httpcore._exceptions"})
    assert is_connect_error(stale("All connection attempts failed"))
    assert not is_connect_error(httpx.ReadError("dropped"))
    assert not is_connect_error(type("ConnectError", (Exception,), {"__module__": "x"})("x"))


async def test_mcp_tool_reaches_an_official_server_by_name(official_mcp_server: Any) -> None:
    """FastMCP checks the Host header itself, so this also proves the pinned
    connection still says who it is talking to."""
    from fastaiagent.tool.mcp import MCPTool

    url = official_mcp_server.url.replace("127.0.0.1", "localhost")
    assert (await MCPTool(name="add", server_url=url).aexecute({"a": 2, "b": 2})).output == "4"


# --- the size cap applies while reading ------------------------------------------


class _Endless:
    """A real HTTP server streaming a chunked body of ``total`` bytes, counting
    what it managed to send before the client hung up."""

    def __init__(self, total: int = 64 * 1024 * 1024) -> None:
        self.sent = 0
        endless = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                chunk = b"x" * 65536
                try:
                    while endless.sent < total:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        endless.sent += len(chunk)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/big"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()


@pytest.fixture
def endless(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    server = _Endless()
    yield server
    server.stop()


async def test_the_async_helper_stops_reading_at_the_cap(endless: Any) -> None:
    with pytest.raises(MultimodalError, match="cap"):
        await asyncio.wait_for(
            asafe_http_request(endless.url, timeout=10, max_redirects=0, max_bytes=1_000_000),
            timeout=60,
        )
    # Before 1.92.0 the whole 64 MiB was read, then measured.
    assert endless.sent < 16 * 1024 * 1024


def test_the_sync_helper_stops_reading_at_the_cap(endless: Any) -> None:
    with pytest.raises(MultimodalError, match="cap"):
        safe_http_fetch(endless.url, timeout=10, max_redirects=0, max_bytes=1_000_000)
    assert endless.sent < 16 * 1024 * 1024


async def test_a_declared_length_over_the_cap_is_refused(
    capture_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    capture_server.set_response_sequence([(200, "y" * 5000, "text/plain")])
    with pytest.raises(MultimodalError, match="cap"):
        await asafe_http_request(
            f"{capture_server.url}/big", timeout=5, max_redirects=0, max_bytes=1000
        )


async def test_a_body_under_the_cap_comes_back_whole(
    capture_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTAIAGENT_ALLOW_PRIVATE_NETWORKS", "1")
    capture_server.set_response_sequence([(200, {"ok": True})])
    resp = await asafe_http_request(
        f"{capture_server.url}/small", timeout=5, max_redirects=0, max_bytes=1000
    )
    assert resp.json() == {"ok": True}
    assert resp.headers["content-type"].startswith("application/json")
