"""HTTP client to FastAIAgent platform public API."""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from fastaiagent._internal.errors import (
    PlatformAuthError,
    PlatformConnectionError,
    PlatformNotConnectedError,
    PlatformNotFoundError,
    PlatformRateLimitError,
    PlatformTierLimitError,
)
from fastaiagent._version import __version__

logger = logging.getLogger(__name__)


#: Hosted program addresses live under this domain (``<label>-dev.api.fastaiagent.net``).
#: Used to decide whether a 5xx hint about "that deployment's own logs" applies.
_HOSTED_DOMAIN = ".fastaiagent.net"


class PlatformAPI:
    """HTTP client for the FastAIAgent platform public API.

    Authenticates via X-API-Key header.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: int = 30,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self, *, json_body: bool = True) -> dict[str, str]:
        from fastaiagent._internal.wire import PLANE_WIRE_PROTOCOL, PROTOCOL_HEADER

        headers = {
            "X-API-Key": self._api_key,
            "User-Agent": f"fastaiagent-sdk/{__version__}",
            PROTOCOL_HEADER: PLANE_WIRE_PROTOCOL,
        }
        if json_body:
            # A multipart upload sets its own Content-Type (with the boundary).
            headers["Content-Type"] = "application/json"
        return headers

    def __repr__(self) -> str:
        # security_review_1.md M8 — never spill the API key into logs,
        # tracebacks, REPLs, or pytest assertion diffs. Show just enough
        # to confirm the object is wired up (last 4 chars + length).
        key = self._api_key or ""
        suffix = key[-4:] if len(key) >= 4 else ""
        redacted = f"***{suffix} (len={len(key)})" if key else "<unset>"
        return (
            f"PlatformAPI(base_url={self._base_url!r}, api_key={redacted}, timeout={self._timeout})"
        )

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle HTTP response, raising appropriate SDK errors.

        Returns the decoded JSON body — an object for most routes, a list for the
        Studio list routes — or ``{}`` when the plane answered with no body (a 204).
        """
        if response.status_code == 401:
            raise PlatformAuthError(
                "Invalid API key. Check your key in the console "
                "(https://login.fastaiagent.net/ for the hosted plane)."
            )
        elif response.status_code == 403:
            detail = ""
            try:
                body = response.json()
                detail = body.get("detail", {})
                if isinstance(detail, dict):
                    detail = detail.get("detail", str(detail))
            except Exception:
                logger.debug("Failed to parse platform 403 response body", exc_info=True)
            if "tier" in str(detail).lower():
                raise PlatformTierLimitError(
                    f"Tier limit reached: {detail}. "
                    "Upgrade in the console (https://login.fastaiagent.net/)."
                )
            if "scope" in str(detail).lower():
                raise PlatformAuthError(
                    f"Insufficient permissions: {detail}. "
                    "Ensure your API key has the required scopes."
                )
            raise PlatformAuthError(f"Forbidden: {detail}")
        elif response.status_code == 404:
            raise PlatformNotFoundError(f"Resource not found: {response.url}")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            try:
                seconds: float | None = float(retry_after)
            except ValueError:  # an HTTP-date, or garbage: say so, but no number to wait on
                seconds = None
            raise PlatformRateLimitError(
                f"Rate limit exceeded. Retry after {retry_after} seconds.", retry_after=seconds
            )
        elif response.status_code >= 500:
            # Name the plane the caller actually configured. A self-hosted
            # operator gets pointed at their own deployment; the hosted plane
            # has no public status page (status.fastaiagent.net does not
            # resolve), so it gets no hint.
            host = (httpx.URL(self._base_url).host or "").lower()
            raise PlatformConnectionError(
                f"Platform server error ({response.status_code}) from {self._base_url}."
                + (
                    ""
                    if host.endswith(_HOSTED_DOMAIN)
                    else " Check that deployment's own logs and health endpoint."
                )
            )

        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Synchronous GET request."""
        return self._send("GET", path, params=params)

    async def aget(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Async GET request."""
        return await self._asend("GET", path, params=params)

    def post(
        self,
        path: str,
        data: dict[str, Any] | None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Synchronous POST request (``data=None`` sends no body)."""
        return self._send("POST", path, params=params, json=data)

    async def apost(
        self,
        path: str,
        data: dict[str, Any] | None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Async POST request (``data=None`` sends no body)."""
        return await self._asend("POST", path, params=params, json=data)

    # -- the other verbs (1.102.0, for the Studio client) ------------------------------
    #
    # They share ``_handle_response`` with ``get``/``post``, so a subclass that
    # overrides it (the Studio client maps the plane's ``detail.error`` codes) changes
    # every verb at once.

    def patch(self, path: str, data: dict[str, Any], params: dict[str, Any] | None = None) -> Any:
        """Synchronous PATCH request."""
        return self._send("PATCH", path, params=params, json=data)

    async def apatch(
        self, path: str, data: dict[str, Any], params: dict[str, Any] | None = None
    ) -> Any:
        """Async PATCH request."""
        return await self._asend("PATCH", path, params=params, json=data)

    def delete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Synchronous DELETE request. A 204 answer comes back as ``{}``."""
        return self._send("DELETE", path, params=params)

    async def adelete(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Async DELETE request. A 204 answer comes back as ``{}``."""
        return await self._asend("DELETE", path, params=params)

    def post_multipart(
        self,
        path: str,
        files: Sequence[tuple[str, bytes]],
        *,
        field: str = "files",
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Upload ``(filename, content)`` pairs as one multipart form, under ``field``."""
        return self._send("POST", path, params=params, files=_multipart(field, files))

    async def apost_multipart(
        self,
        path: str,
        files: Sequence[tuple[str, bytes]],
        *,
        field: str = "files",
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Async :meth:`post_multipart`."""
        return await self._asend("POST", path, params=params, files=_multipart(field, files))

    def download(self, path: str, dest: str | Path, params: dict[str, Any] | None = None) -> Path:
        """Stream a GET answer to ``dest`` and return it.

        The body is written in chunks, so a large export is never held in memory. The
        file is created only once the plane has answered 2xx; an error status is mapped
        exactly as for any other call and nothing is written.
        """
        target = Path(dest)
        try:
            with httpx.Client(timeout=self._timeout, verify=True) as client:
                with client.stream(
                    "GET",
                    f"{self._base_url}{path}",
                    params=params,
                    headers=self._headers(json_body=False),
                ) as response:
                    if not response.is_success:
                        response.read()
                        self._handle_response(response)
                        response.raise_for_status()
                    with target.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
        except httpx.TransportError as err:
            raise self._transport_error(err) from err
        return target

    async def adownload(
        self, path: str, dest: str | Path, params: dict[str, Any] | None = None
    ) -> Path:
        """Async :meth:`download`."""
        target = Path(dest)
        try:
            async with httpx.AsyncClient(timeout=self._timeout, verify=True) as client:
                async with client.stream(
                    "GET",
                    f"{self._base_url}{path}",
                    params=params,
                    headers=self._headers(json_body=False),
                ) as response:
                    if not response.is_success:
                        await response.aread()
                        self._handle_response(response)
                        response.raise_for_status()
                    with target.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            handle.write(chunk)
        except httpx.TransportError as err:
            raise self._transport_error(err) from err
        return target

    # -- plumbing -------------------------------------------------------------------

    def _transport_error(self, err: httpx.TransportError) -> PlatformConnectionError:
        """One message for a connection that never opened, another for one that broke."""
        if isinstance(err, httpx.ConnectError):
            return PlatformConnectionError(
                "Cannot connect to platform. Check your internet connection "
                "and verify the target URL is correct."
            )
        return PlatformConnectionError(
            f"The connection to {self._base_url} failed ({type(err).__name__}: {err})."
        )

    def _send(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            with httpx.Client(timeout=self._timeout, verify=True) as client:
                response = client.request(
                    method,
                    f"{self._base_url}{path}",
                    headers=self._headers(json_body="files" not in kwargs),
                    **kwargs,
                )
                return self._handle_response(response)
        except httpx.TransportError as err:
            raise self._transport_error(err) from err

    async def _asend(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, verify=True) as client:
                response = await client.request(
                    method,
                    f"{self._base_url}{path}",
                    headers=self._headers(json_body="files" not in kwargs),
                    **kwargs,
                )
                return self._handle_response(response)
        except httpx.TransportError as err:
            raise self._transport_error(err) from err


def _multipart(field: str, files: Sequence[tuple[str, bytes]]) -> list[tuple[str, Any]]:
    """``(filename, content)`` pairs as httpx's multipart list, typed from the extension."""
    return [
        (field, (name, content, mimetypes.guess_type(name)[0] or "application/octet-stream"))
        for name, content in files
    ]


def get_platform_api() -> PlatformAPI:
    """Get a PlatformAPI instance using the current connection."""
    from fastaiagent.client import _connection

    if not _connection.is_connected:
        raise PlatformNotConnectedError("Not connected to platform. Call fa.connect() first.")
    return PlatformAPI(
        api_key=_connection.api_key or "",
        base_url=_connection.target,
    )
