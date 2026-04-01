"""HTTP client to FastAIAgent platform public API."""

from __future__ import annotations

from typing import Any

import httpx

from fastaiagent._internal.errors import (
    PlatformAuthError,
    PlatformConnectionError,
    PlatformNotFoundError,
    PlatformRateLimitError,
    PlatformTierLimitError,
)
from fastaiagent._version import __version__


class PlatformAPI:
    """HTTP client for the FastAIAgent platform public API.

    Authenticates via X-API-Key header. All requests go to /public/v1/sdk/*.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://app.fastaiagent.net",
        timeout: int = 30,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
            "User-Agent": f"fastaiagent-sdk/{__version__}",
        }

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle HTTP response, raising appropriate SDK errors."""
        if response.status_code == 401:
            raise PlatformAuthError(
                "Invalid API key. Check your key at https://app.fastaiagent.net/settings/api-keys"
            )
        elif response.status_code == 403:
            detail = ""
            try:
                body = response.json()
                detail = body.get("detail", {})
                if isinstance(detail, dict):
                    detail = detail.get("detail", str(detail))
            except Exception:
                pass
            if "tier" in str(detail).lower():
                raise PlatformTierLimitError(
                    f"Tier limit reached: {detail}. Upgrade at https://app.fastaiagent.net/billing"
                )
            if "scope" in str(detail).lower():
                raise PlatformAuthError(
                    f"Insufficient permissions: {detail}. "
                    "Ensure your API key has the required write scopes."
                )
            raise PlatformAuthError(f"Forbidden: {detail}")
        elif response.status_code == 404:
            raise PlatformNotFoundError(f"Resource not found: {response.url}")
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            raise PlatformRateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds.")
        elif response.status_code >= 500:
            raise PlatformConnectionError(
                f"Platform server error ({response.status_code}). "
                f"Check status at https://status.fastaiagent.net"
            )

        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Synchronous POST request."""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}{path}",
                    json=data,
                    headers=self._headers(),
                )
                return self._handle_response(response)
        except httpx.ConnectError:
            raise PlatformConnectionError(
                "Cannot connect to platform. Check your internet connection "
                "and verify the target URL is correct."
            )

    async def apost(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """Async POST request."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}{path}",
                    json=data,
                    headers=self._headers(),
                )
                return self._handle_response(response)
        except httpx.ConnectError:
            raise PlatformConnectionError(
                "Cannot connect to platform. Check your internet connection "
                "and verify the target URL is correct."
            )
