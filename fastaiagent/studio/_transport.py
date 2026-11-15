"""The HTTP layer under ``StudioClient``: ``PlatformAPI`` plus the plane's Studio errors.

Every Studio refusal has one shape::

    {"detail": {"error": "name_taken", "message": "...", "...": "extras per code"}}

``PlatformAPI`` maps status codes and drops the body, which is right for the plane's
older routes and loses what a coding tool must act on here. This subclass reads the
code first and keeps the rest of the body.

What it does *not* take over, so a caller handles Studio exactly as it handles every
other plane call:

* no usable key — 401 ``invalid_api_key`` and, on a SaaS server that holds several
  workspaces, 400 ``customer_required`` — is :class:`PlatformAuthError`;
* 429 is :class:`PlatformRateLimitError`, and a 5xx is :class:`PlatformConnectionError`
  (both from the base class).
"""

from __future__ import annotations

from typing import Any

import httpx

from fastaiagent._internal.errors import PlatformAuthError, StudioError
from fastaiagent._platform.api import PlatformAPI

#: Codes that mean "there is no usable key", whatever the status.
_NO_USABLE_KEY = {"invalid_api_key", "customer_required"}


def _detail(response: httpx.Response) -> dict[str, Any] | None:
    """The plane's ``detail`` object when the body has the Studio error shape."""
    try:
        body = response.json()
    except Exception:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict) and isinstance(detail.get("error"), str):
        return detail
    return None


class StudioTransport(PlatformAPI):
    def _handle_response(self, response: httpx.Response) -> Any:
        if response.is_success or response.status_code >= 500 or response.status_code == 429:
            return super()._handle_response(response)
        detail = _detail(response)
        if detail is None:
            return super()._handle_response(response)
        code = detail["error"]
        message = str(detail.get("message") or detail.get("detail") or code)
        if code in _NO_USABLE_KEY:
            raise PlatformAuthError(
                f"{message} Pass an API key with the Agent Studio permissions "
                "(studio:read and studio:write), or set FASTAIAGENT_API_KEY."
            )
        extras = {k: v for k, v in detail.items() if k not in ("error", "message", "detail")}
        raise StudioError(code, message, status=response.status_code, details=extras)
