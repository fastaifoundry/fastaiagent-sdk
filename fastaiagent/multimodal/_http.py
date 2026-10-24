"""Shared SSRF-hardened HTTP fetch helper for multimodal URL ingestion.

The previous implementation in :mod:`fastaiagent.multimodal.image` and
:mod:`fastaiagent.multimodal.pdf` checked only the URL scheme, then handed
the URL to ``httpx.Client(follow_redirects=True)``. That allowed two SSRF
shapes:

1. The initial host could be a private/loopback/link-local address (e.g.
   ``http://127.0.0.1:7842/api/...`` or ``http://169.254.169.254/...``).
2. An attacker-controlled public host could 302-redirect into one.

This helper rejects both. Hardening:

* Scheme must be ``http`` or ``https``.
* Host must resolve to a public address (not private / loopback /
  link-local / reserved / multicast / unspecified). Set
  ``FASTAIAGENT_ALLOW_PRIVATE_NETWORKS=1`` to opt out (intranet use).
* Redirects are followed manually with a per-hop scheme + host check.
* Response body is capped at ``max_bytes``, while it is read (1.92.0).
* The connection goes to the address that was checked (1.92.0). Before, httpx
  resolved the name again to connect, so a DNS server answering "public" to
  the check and "internal" to the connect (DNS rebinding) got through. The
  request keeps its ``Host`` header, and TLS still verifies the certificate
  against the hostname (httpx's ``sni_hostname`` extension).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

import httpx

from fastaiagent._internal.errors import MultimodalError, UnsupportedFormatError

logger = logging.getLogger(__name__)

ALLOW_PRIVATE_NETWORKS_ENV = "FASTAIAGENT_ALLOW_PRIVATE_NETWORKS"


def _allow_private_networks() -> bool:
    """Resolver for ``FASTAIAGENT_ALLOW_PRIVATE_NETWORKS`` (registered in ``ENV_FLAGS``).

    A capability grant that widens the SSRF guard, so an unparseable value
    resolves to *not granted* — private/intranet hosts stay blocked.
    """
    from fastaiagent._internal.env import env_flag

    return env_flag(ALLOW_PRIVATE_NETWORKS_ENV, default=False, on_unparsed=False)


def _is_blocked_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_loopback: bool = False,
) -> bool:
    # ``allow_loopback`` permits 127.0.0.0/8 and ::1 while STILL blocking the
    # SSRF targets that actually matter — cloud metadata (link-local
    # 169.254.169.254), RFC1918 private ranges, reserved/multicast. Used by MCP
    # tools, which are very commonly run on localhost (security_audit_2 N15).
    if allow_loopback and ip.is_loopback:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        # Anything else not routable on the public internet (1.92.0): carrier-
        # grade NAT 100.64.0.0/10 — where Tailscale puts tailnet devices — and
        # benchmarking space were not "private", so they passed.
        or not ip.is_global
    )


def validate_url(url: str, *, allow_loopback: bool = False) -> list[str] | None:
    """Raise if ``url`` is not a public http(s) URL we are willing to fetch.

    ``allow_loopback=True`` additionally permits loopback addresses (for local
    MCP servers etc.) while keeping every other private/link-local/metadata
    block in force.

    Returns the checked addresses to connect to (IPv4 first) when the host is a
    name that was resolved here; ``None`` for a literal IP or when private
    networks are allowed (nothing to pin). See :func:`send_pinned`.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsupportedFormatError(
            f"unsupported URL scheme {parsed.scheme!r}; only http(s) allowed"
        )
    host = parsed.hostname
    if not host:
        raise UnsupportedFormatError(f"URL missing host: {url!r}")
    if _allow_private_networks():
        return None
    # Literal IP in the URL — check directly.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if _is_blocked_ip(ip, allow_loopback=allow_loopback):
            raise MultimodalError(
                f"Refusing to fetch {url!r}: host {host!r} is not a public "
                f"address. Set {ALLOW_PRIVATE_NETWORKS_ENV}=1 to override."
            )
        return None
    # Hostname — resolve and check every record. ``getaddrinfo`` may return
    # both IPv4 and IPv6 entries; reject if any is non-public.
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise MultimodalError(f"Could not resolve host {host!r}: {exc}") from exc
    checked: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _fam, _type, _proto, _canon, sockaddr in infos:
        addr = sockaddr[0]
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if _is_blocked_ip(resolved, allow_loopback=allow_loopback):
            raise MultimodalError(
                f"Refusing to fetch {url!r}: host {host!r} resolves to "
                f"non-public address {addr!r}. Set "
                f"{ALLOW_PRIVATE_NETWORKS_ENV}=1 to override."
            )
        if resolved not in checked:
            checked.append(resolved)
    if not checked:
        raise MultimodalError(f"Could not resolve host {host!r} to an IP address")
    return [str(a) for a in sorted(checked, key=lambda a: a.version)]


def pinned(request: httpx.Request, address: str) -> httpx.Request:
    """``request`` sent to the checked ``address`` instead of being resolved
    again: same method, headers (``Host`` included), body and extensions, plus
    ``sni_hostname`` so TLS verifies the certificate for the hostname."""
    return httpx.Request(
        request.method,
        request.url.copy_with(host=address),
        headers=request.headers,
        stream=request.stream,
        extensions={**request.extensions, "sni_hostname": request.url.host},
    )


def send_pinned(
    client: httpx.Client, request: httpx.Request, addresses: list[str] | None
) -> httpx.Response:
    """Send ``request`` (streamed) to the checked addresses in turn — the next
    one only when a connection cannot be made, as httpx does for a name.

    "Cannot connect" is recognised by name (``is_connect_error``): after a
    re-import of ``httpcore``, httpx lets the raw ``httpcore.ConnectError``
    through unmapped (see ``_internal/http_errors.py``)."""
    from fastaiagent._internal.http_errors import is_connect_error

    if not addresses:
        return client.send(request, stream=True)
    for address in addresses[:-1]:
        try:
            return client.send(pinned(request, address), stream=True)
        except Exception as e:
            if not is_connect_error(e):
                raise
    return client.send(pinned(request, addresses[-1]), stream=True)


async def asend_pinned(
    send: Callable[[httpx.Request], Awaitable[httpx.Response]],
    request: httpx.Request,
    addresses: list[str] | None,
) -> httpx.Response:
    """Async :func:`send_pinned`; ``send`` is a client's ``send`` or a
    transport's ``handle_async_request`` (then ``stream`` does not apply)."""
    from fastaiagent._internal.http_errors import is_connect_error

    if not addresses:
        return await send(request)
    for address in addresses[:-1]:
        try:
            return await send(pinned(request, address))
        except Exception as e:
            if not is_connect_error(e):
                raise
    return await send(pinned(request, addresses[-1]))


def _too_big(url: str, size: int, max_bytes: int) -> MultimodalError:
    return MultimodalError(f"Response body from {url!r} is {size} bytes; cap is {max_bytes}.")


def _declared_size(resp: httpx.Response) -> int | None:
    try:
        return int(resp.headers["content-length"])
    except (KeyError, ValueError):
        return None


def _capped(resp: httpx.Response, request: httpx.Request, body: bytes) -> httpx.Response:
    """The response as callers expect it: the body read under the cap (already
    decoded, so the encoding headers go), and the request they made — the
    hostname URL, not the pinned address."""
    headers = httpx.Headers(resp.headers)
    for name in ("content-encoding", "content-length", "transfer-encoding"):
        headers.pop(name, None)
    return httpx.Response(resp.status_code, headers=headers, content=body, request=request)


# Headers that must not follow a redirect to a different origin — credentials
# and session material. Compared case-insensitively (security_audit_2 N11).
_SENSITIVE_REDIRECT_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-api-key",
        "api-key",
        "x-auth-token",
        "x-amz-security-token",
    }
)


def _same_origin(a: str, b: str) -> bool:
    """True if two URLs share scheme + host + port (case-insensitive host)."""
    pa, pb = urlparse(a), urlparse(b)
    return (
        pa.scheme == pb.scheme
        and (pa.hostname or "").lower() == (pb.hostname or "").lower()
        and pa.port == pb.port
    )


def _strip_sensitive_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Return ``headers`` without any credential/session headers (case-insensitive)."""
    if not headers:
        return headers
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in _SENSITIVE_REDIRECT_HEADERS
    }


def safe_http_fetch(
    url: str,
    *,
    timeout: float,
    max_redirects: int,
    max_bytes: int,
) -> httpx.Response:
    """Fetch ``url`` over HTTP(S) with SSRF + size hardening.

    Redirects are walked manually so each hop is re-validated against the
    private-IP block. Returns the final non-redirect ``httpx.Response``.
    The caller may inspect ``.headers`` / ``.content``.
    """
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout, verify=True) as client:
        for _hop in range(max_redirects + 1):
            # Check this hop, then connect to the address that was checked.
            addresses = validate_url(current)
            request = client.build_request("GET", current)
            resp = send_pinned(client, request, addresses)
            try:
                if 300 <= resp.status_code < 400 and "location" in resp.headers:
                    # Resolve a relative redirect against the hop URL; the next
                    # iteration validates the target before requesting it.
                    current = str(httpx.URL(current).join(resp.headers["location"]))
                    continue
                declared = _declared_size(resp)
                if declared is not None and declared > max_bytes:
                    raise _too_big(url, declared, max_bytes)
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > max_bytes:
                        raise _too_big(url, len(body), max_bytes)
            finally:
                resp.close()
            out = _capped(resp, request, bytes(body))
            out.raise_for_status()
            return out
    raise MultimodalError(f"Too many redirects (>{max_redirects}) fetching {url!r}")


async def asafe_http_request(
    url: str,
    *,
    method: str = "GET",
    timeout: float,
    max_redirects: int,
    max_bytes: int,
    headers: dict[str, str] | None = None,
    json: Any | None = None,
    params: dict[str, Any] | None = None,
    allow_loopback: bool = False,
) -> httpx.Response:
    """Async variant of :func:`safe_http_fetch` for arbitrary verbs.

    Used by ``RESTTool`` and any other async caller that needs the same
    SSRF hardening (private-IP block + per-hop redirect validation +
    response-size cap) but with a non-GET method, headers, or a body.

    ``allow_loopback`` permits loopback targets (for local MCP servers) while
    keeping the private/link-local/metadata block in force. The flag is applied
    on every hop, so a redirect can't escape it either way.

    On a 303 redirect (the only redirect that mandates a method change),
    we drop the body and switch to GET — matching what ``httpx`` does
    when ``follow_redirects=True``.

    Credential/session headers (``Authorization``, ``Cookie``, API-key headers)
    are dropped on a cross-origin redirect so a cooperating first hop can't
    bounce a bearer token to another host (security_audit_2 N11).
    """
    current = url
    current_method = method.upper()
    current_json = json
    current_headers = dict(headers) if headers else headers
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout, verify=True) as client:
        for _hop in range(max_redirects + 1):
            # Check this hop, then connect to the address that was checked.
            addresses = validate_url(current, allow_loopback=allow_loopback)
            request = client.build_request(
                current_method, current, headers=current_headers, json=current_json, params=params
            )
            resp = await asend_pinned(lambda r: client.send(r, stream=True), request, addresses)
            try:
                if 300 <= resp.status_code < 400 and "location" in resp.headers:
                    next_target = str(httpx.URL(current).join(resp.headers["location"]))
                    # Strip credentials before following a redirect to a new origin.
                    if not _same_origin(current, next_target):
                        current_headers = _strip_sensitive_headers(current_headers)
                    current = next_target
                    if resp.status_code == 303:
                        current_method = "GET"
                        current_json = None
                    continue
                declared = _declared_size(resp)
                if declared is not None and declared > max_bytes:
                    raise _too_big(url, declared, max_bytes)
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > max_bytes:
                        raise _too_big(url, len(body), max_bytes)
            finally:
                await resp.aclose()
            out = _capped(resp, request, bytes(body))
            out.raise_for_status()
            return out
    raise MultimodalError(f"Too many redirects (>{max_redirects}) fetching {url!r}")


# Backwards-compat alias — older callers import the underscored name.
_validate_url = validate_url


__all__ = [
    "ALLOW_PRIVATE_NETWORKS_ENV",
    "asafe_http_request",
    "safe_http_fetch",
    "validate_url",
]
