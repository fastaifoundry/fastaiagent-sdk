"""Connection management for the FastAIAgent platform."""

from __future__ import annotations

import logging
from typing import Any

from fastaiagent._version import __version__

logger = logging.getLogger(__name__)


class _Connection:
    """Singleton holding platform connection state."""

    def __init__(self) -> None:
        self.api_key: str | None = None
        self.target: str = "https://app.fastaiagent.net"
        self.project: str | None = None
        self.domain_id: str | None = None
        self.project_id: str | None = None
        self.scopes: list[str] = []
        self._platform_processor: Any = None

    @property
    def is_connected(self) -> bool:
        return self.api_key is not None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key or "",
            "Content-Type": "application/json",
            "User-Agent": f"fastaiagent-sdk/{__version__}",
        }


_connection = _Connection()


def _normalize_target(target: str) -> str:
    """Normalize a platform target URL.

    Ensures a scheme is present (defaults to ``http://`` for localhost/
    private hosts, ``https://`` otherwise) and strips trailing slashes.
    httpx rejects URLs without a scheme with an opaque error, so we
    do this upfront to give users a clear, tolerant experience.
    """
    t = (target or "").strip().rstrip("/")
    if not t:
        return t
    if "://" in t:
        return t
    host = t.split("/", 1)[0].split(":", 1)[0].lower()
    is_local = host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(
        ".localhost"
    )
    return f"{'http' if is_local else 'https'}://{t}"


def connect(
    api_key: str,
    target: str = "https://app.fastaiagent.net",
    project: str | None = None,
) -> None:
    """Connect the SDK to FastAIAgent Platform for observability,
    prompt management, and evaluation services.

    All SDK features work without connect(). This adds platform
    backends alongside local storage.

    The API key carries its own domain and project scope from the
    platform. The ``project`` parameter is an optional override
    for the trace export payload.
    """
    import httpx

    from fastaiagent._internal.errors import PlatformAuthError, PlatformConnectionError

    _connection.api_key = api_key
    _connection.target = _normalize_target(target)
    _connection.project = project

    # Lightweight auth check — also captures domain/project from the key
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{_connection.target}/public/v1/auth/check",
                headers=_connection.headers,
            )
            if resp.status_code == 401:
                _connection.api_key = None
                _connection.project = None
                raise PlatformAuthError(
                    "Invalid API key. Check your key at "
                    "https://app.fastaiagent.net/settings/api-keys"
                )
            if resp.status_code == 403:
                _connection.api_key = None
                _connection.project = None
                raise PlatformAuthError(f"Forbidden: {resp.text}")
            if resp.status_code == 200:
                data = resp.json()
                _connection.domain_id = data.get("domain_id")
                _connection.project_id = data.get("project_id")
                _connection.scopes = data.get("scopes", [])
                logger.info(
                    "Connected to platform: domain=%s project=%s scopes=%s",
                    _connection.domain_id,
                    _connection.project_id,
                    _connection.scopes,
                )
    except httpx.ConnectError:
        # Allow connecting even if platform is unreachable — traces will
        # queue locally and export when the platform becomes available.
        logger.warning(
            "Could not reach platform at %s. "
            "Connection stored — traces will export when platform is reachable.",
            _connection.target,
        )
    except (PlatformAuthError, PlatformConnectionError):
        raise

    # Register platform trace exporter
    try:
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from fastaiagent.trace.otel import get_tracer_provider
        from fastaiagent.trace.platform_export import PlatformSpanExporter

        exporter = PlatformSpanExporter()
        processor = BatchSpanProcessor(exporter)
        get_tracer_provider().add_span_processor(processor)
        _connection._platform_processor = processor
    except Exception:
        logger.debug("Could not register platform trace exporter", exc_info=True)


def disconnect() -> None:
    """Disconnect from platform. Revert to local-only mode.

    Forces a flush of any pending trace spans before disconnecting.
    """
    if _connection._platform_processor is not None:
        try:
            _connection._platform_processor.force_flush(timeout_millis=5000)
            _connection._platform_processor.shutdown()
        except Exception:
            pass
        _connection._platform_processor = None
    _connection.api_key = None
    _connection.project = None
    _connection.domain_id = None
    _connection.project_id = None
    _connection.scopes = []
