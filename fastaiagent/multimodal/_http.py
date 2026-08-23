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
* Response body is capped at ``max_bytes``.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

from fastaiagent._internal.errors import MultimodalError, UnsupportedFormatError

logger = logging.getLogger(__name__)

ALLOW_PRIVATE_NETWORKS_ENV = "FASTAIAGENT_ALLOW_PRIVATE_NETWORKS"


def _allow_private_networks() -> bool:
    return os.environ.get(ALLOW_PRIVATE_NETWORKS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


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
    )


def validate_url(url: str, *, allow_loopback: bool = False) -> None:
    """Raise if ``url`` is not a public http(s) URL we are willing to fetch.

    ``allow_loopback=True`` additionally permits loopback addresses (for local
    MCP servers etc.) while keeping every other private/link-local/metadata
    block in force.
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
        return
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
        return
    # Hostname — resolve and check every record. ``getaddrinfo`` may return
    # both IPv4 and IPv6 entries; reject if any is non-public.
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise MultimodalError(f"Could not resolve host {host!r}: {exc}") from exc
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
    validate_url(url)
    current = url
    with httpx.Client(follow_redirects=False, timeout=timeout, verify=True) as client:
        for _hop in range(max_redirects + 1):
            resp = client.get(current)
            if 300 <= resp.status_code < 400 and "location" in resp.headers:
                # Resolve relative redirect against the hop URL, then
                # re-validate the resolved target before issuing the next GET.
                next_target = str(httpx.URL(current).join(resp.headers["location"]))
                validate_url(next_target)
                current = next_target
                continue
            resp.raise_for_status()
            if len(resp.content) > max_bytes:
                raise MultimodalError(
                    f"Response body from {url!r} is "
                    f"{len(resp.content)} bytes; cap is {max_bytes}."
                )
            return resp
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
    validate_url(url, allow_loopback=allow_loopback)
    current = url
    current_method = method.upper()
    current_json = json
    current_headers = dict(headers) if headers else headers
    async with httpx.AsyncClient(follow_redirects=False, timeout=timeout, verify=True) as client:
        for _hop in range(max_redirects + 1):
            resp = await client.request(
                current_method,
                current,
                headers=current_headers,
                json=current_json,
                params=params,
            )
            if 300 <= resp.status_code < 400 and "location" in resp.headers:
                next_target = str(httpx.URL(current).join(resp.headers["location"]))
                validate_url(next_target, allow_loopback=allow_loopback)
                # Strip credentials before following a redirect to a new origin.
                if not _same_origin(current, next_target):
                    current_headers = _strip_sensitive_headers(current_headers)
                current = next_target
                if resp.status_code == 303:
                    current_method = "GET"
                    current_json = None
                continue
            resp.raise_for_status()
            if len(resp.content) > max_bytes:
                raise MultimodalError(
                    f"Response body from {url!r} is "
                    f"{len(resp.content)} bytes; cap is {max_bytes}."
                )
            return resp
    raise MultimodalError(f"Too many redirects (>{max_redirects}) fetching {url!r}")


# Backwards-compat alias — older callers import the underscored name.
_validate_url = validate_url


__all__ = [
    "ALLOW_PRIVATE_NETWORKS_ENV",
    "asafe_http_request",
    "safe_http_fetch",
    "validate_url",
]
